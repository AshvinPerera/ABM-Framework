use std::sync::Mutex;
use types::{EntityID, ShardID, IndexID, VersionID, EntityCount, 
    ENTITY_BITS, SHARD_BITS, VERSION_BITS, INDEX_BITS, INDEX_CAP};
use error::{CapacityError, ShardBoundsError};

const ENTITY_BITS: u8 = 64;
const SHARD_BITS: u8 = 10;
const VERSION_BITS: u8 = 32;
const INDEX_BITS: u8 = ENTITY_BITS - SHARD_BITS - VERSION_BITS;

const _: () = assert!(VERSION_BITS + SHARD_BITS < ENTITY_BITS, "bit layout overflow. shrink VERSION_BITS or SHARD_BITS");
const INDEX_CAP: IndexID = (((1 as EntityID) << INDEX_BITS) - 1) as IndexID;

const INDEX_MASK: EntityID = ((1 as EntityID) << INDEX_BITS) - 1;
const SHARD_MASK: EntityID = ((1 as EntityID) << SHARD_BITS) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Entity(EntityID);

#[inline]
fn make_entity(shard: ShardID, index: IndexID, version: VersionID) -> Entity {
    debug_assert!((index as EntityID) <= INDEX_MASK);
    debug_assert!((shard as EntityID) <= SHARD_MASK);
    let id = 
        ((version as EntityID) << (SHARD_BITS + INDEX_BITS)) | 
        ((shard as EntityID) << INDEX_BITS) |  
        (index as EntityID);
    Entity(id)
}

#[inline]
fn split_entity(entity: Entity) -> (ShardID, IndexID, VersionID) {
    let id = entity.0;
    let shard = ((id >> INDEX_BITS) & SHARD_MASK) as ShardID;    
    let index = (id & INDEX_MASK) as IndexID;
    let version = (id >> (INDEX_BITS + SHARD_BITS)) as VersionID;
    (shard, index, version)
}

impl Entity {
    #[inline] pub fn components(self) -> (ShardID, IndexID, VersionID) { split_entity(self) }
    #[inline] pub fn shard(self) -> ShardID { ((self.0 >> INDEX_BITS) & SHARD_MASK) as ShardID }
    #[inline] pub fn index(self) -> IndexID { (self.0 & INDEX_MASK) as IndexID }
    #[inline] pub fn version(self) -> VersionID { (self.0 >> (INDEX_BITS + SHARD_BITS)) as VersionID }
}

#[derive(Default)]
pub struct Entities {
    versions: Vec<VersionID>,
    free_store: Vec<IndexID>,
}

impl Entities {
    pub fn new() -> Self { Self::default() }

    fn ensure_capacity(&self, additional: EntityCount) -> Result<(), CapacityError> {
        let bits_needed = (self.versions.len() as EntityID) + (additional as EntityID);
        let capacity = INDEX_CAP as EntityID + 1;
        if bits_needed > capacity {
            return Err(CapacityError { bits_needed, capacity });
        }
        Ok(())
    }

    pub fn spawn(&mut self, shard_id: ShardID) -> Result<Entity, CapacityError> {
        if let Some(i) = self.free_store.pop() {
            return Ok(make_entity(shard_id, i, self.versions[i as usize]));
        } 
        self.ensure_capacity(1)?;
        let i = self.versions.len() as IndexID;
        self.versions.push(0);
        Ok(make_entity(shard_id, i, 0))
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        let (_, i, v) = split_entity(entity);
        match self.versions.get_mut(i as usize) {
            Some(live) if *live == v => {
                *live = live.wrapping_add(1);
                self.free_store.push(i);
                true
            }
            _ => false,
        }
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let (_, i, v) = split_entity(entity);
        self.versions.get(i as usize).map_or(false, |&live| live == v)
    }

    pub fn spawn_n(&mut self, shard_id: ShardID, n: EntityCount)  -> Result<Vec<Entity>, CapacityError> {
        if n == 0 { return Ok(Vec::new()); }        
    
        let reusable_indexes = self.free_store.len().min(n as usize) as EntityCount;
        let new_indexes = n - reusable_indexes;
        self.ensure_capacity(new_indexes as IndexID)?;

        let mut spawned_entities = Vec::with_capacity(n as usize);

        for _ in 0..reusable_indexes {
            let i = self.free_store.pop().unwrap();
            spawned_entities.push(make_entity(shard_id, i, self.versions[i as usize]));
        }

        
        if new_indexes > 0 {
            let start = self.versions.len() as VersionID;
            self.versions.reserve(new_indexes as usize);
            self.versions.extend(std::iter::repeat(0).take(new_indexes as usize));
            
            for offset in 0..new_indexes {
                spawned_entities.push(make_entity(shard_id, start + offset, 0));
            }
        }
        
        Ok(spawned_entities)
    }
}

pub struct EntityShards {
    shards: Vec<Mutex<Entities>>,
}

impl EntityShards {
    pub fn new(n_shards: usize) -> Self {
        assert!(n_shards > 0 && n_shards <= (1usize << SHARD_BITS));
        let mut shards = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            shards.push(Mutex::new(Entities::default()));
        }
        Self { shards }
    }

    #[inline]
    fn pick_shard_by_thread(&self) -> ShardID {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let thread_id = std::thread::current().id();
        let mut hasher = DefaultHasher::new();
        thread_id.hash(&mut hasher);
        (hasher.finish() as usize % self.shards.len()) as ShardID
    }

    pub fn spawn(&self) -> Entity {
        let shard_index = self.pick_shard_by_thread();
        let mut shard = self.shards[shard_index as usize].lock().unwrap();
        shard.spawn(shard_index)
    }

    pub fn spawn_on(&self, shard_id: ShardID) -> Result<Entity, ShardBoundsError> {
        if (shard_id as usize) >= self.shards.len() {
            return Err(ShardBoundsError { index: shard_id, max_index: (self.shards.len() - 1) as u32 });
        }
        let mut shard = self.shards[shard_id as usize].lock().unwrap();
        Ok(shard.spawn(shard_id))
    }

    pub fn spawn_n(&self, n: usize) -> Vec<Entity> {
        use rayon::prelude::*;
    
        let shard_count = self.shards.len();
        let n_per_shard = n / shard_count;
        let extra_spawns = n % shard_count;

        (0..shard_count).into_par_iter().flat_map(|shard| {
            let spawn_count = n_per_shard + usize::from(shard < extra_spawns);
            self.spawn_n_on(shard as ShardID, spawn_count)
        }).collect()
    }

    pub fn spawn_n_on(&self, shard_id: ShardID, n: usize) -> Result<Vec<Entity>, ShardBoundsError> {
        if (shard_id as usize) >= self.shards.len() {
            return Err(ShardBoundsError { index: shard_id, max_index: (self.shards.len() - 1) as u32 });
        }
        
        let mut shard = self.shards[shard_id as usize].lock().unwrap();
        let mut spawned_entities = shard.spawn_n(shard_id as ShardID, n as EntityCount);
        Ok(spawned_entities)
    }

    pub fn despawn(&self, entity: Entity) -> bool {
        let shard_index = entity.shard();
        let mut shard = self.shards[shard_index as usize].lock().unwrap();
        shard.despawn(entity)
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let shard_index = entity.shard();
        let shard = self.shards[shard_index as usize].lock().unwrap();
        shard.is_alive(entity)
    }
}