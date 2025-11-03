use crate::types::{
    ArchetypeID, 
    ShardID,
    ChunkID,
    RowID, 
    CHUNK_CAP, 
    ComponentID, 
    COMPONENT_CAP,
    Signature
};
use crate::storage::{
    TypeErasedAttribute
};
use crate::entity::{
    Entity, 
    EntityLocation, 
    EntityShards
};
use crate::component::{ 
    component_id_of_type_id
};
use crate::error::{
    SpawnError
};


#[derive(Debug)]
struct Archetype {
    archetype_id: ArchetypeID,
    components: Vec<Option<Box<dyn TypeErasedAttribute>>>,
    signature: Signature,
    length: usize,
    entity_positions: Vec<Vec<Option<Entity>>>
}

impl Archetype {
    fn new(archetype_id: ArchetypeID) -> Self {
            Self {
                archetype_id,
                components: vec![None; COMPONENT_CAP],
                signature: Signature::default(),
                length: 0,
                entity_positions: Vec::new(),
            }
        }

    fn length(&self) -> usize {
        self.length
    }

    fn ensure_capacity(&mut self, chunk_count: usize) {
        while self.entity_positions.len() < chunk_count {
            self.entity_positions.push(vec![None; CHUNK_CAP]);
        }
    }

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    #[inline]
    pub fn component(&self, component_id: ComponentID) -> Option<&dyn TypeErasedAttribute> {
        self.components.get(component_id as usize).and_then(|o| o.as_deref())
    }

    #[inline]
    pub fn component_mut(&mut self, component_id: ComponentID) -> Option<&mut dyn TypeErasedAttribute> {
        self.components.get_mut(component_id as usize).and_then(|o| o.as_deref_mut())
    } 

    pub fn insert_empty_component(&mut self, component_id: ComponentID, component: Box<dyn TypeErasedAttribute>) {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            panic!("component_id out of range for COMPONENT_CAP.");
        }

        let slot = &mut self.components[index];
        debug_assert!(slot.is_none(), "the component is already present.");
        *slot = Some(component);
        self.signature.set(component_id);
    }

    pub fn remove_component(&mut self, component_id: ComponentID) -> Option<Box<dyn TypeErasedAttribute>> {
        if self.length > 0 {
            panic!("cannot remove a component from a non-empty archetype.");
        } 

        let index = component_id as usize;
        let taken = self.components.get_mut(index)?.take();
        if taken.is_some() {
            self.signature.clear(component_id);
        }
        taken
    }

    pub fn spawn_on(&mut self, shards: &mut EntityShards, shard_id: ShardID) -> Result<Entity, SpawnError> {
        let mut reference_position: Option<(ChunkID, RowID)> = None;

        for component in self.components.iter_mut().filter_map(|c| c.as_mut()) {
            let position = match component.push_dyn(Box::new(())) {
                Ok(p) => p,
                Err(_) => {
                    if let Some((chunk, row)) = reference_position {
                        for c in self.components.iter_mut().filter_map(|c| c.as_mut()) {
                            let _ = c.swap_remove(chunk, row);
                        }
                    }
                    return Err(SpawnError::StoragePushFailed);
                }
            };

            if let Some(rp) = reference_position {
                assert_eq!(position, rp, "attributes must stay aligned per row.");
            } else {
                reference_position = Some(position);
            }
        }

        let Some((chunk, row)) = reference_position else {
            return Err(SpawnError::EmptyArchetype);
        };

        self.ensure_capacity(chunk as usize + 1);

        debug_assert!(
            self.entity_positions[chunk as usize][row as usize].is_none(),
            "spawn_on: target entity slot is already occupied."
        );

        let location = EntityLocation { archetype: self.archetype_id, chunk, row };

        let entity = match shards.spawn_on(shard_id, location) {
            Ok(e) => e,
            Err(err) => {
                for component in self.components.iter_mut().filter_map(|c| c.as_mut()) {
                    let _ = component.swap_remove(chunk, row);
                }
                return Err(err);
            }
        };

        self.entity_positions[chunk as usize][row as usize] = Some(entity);
        self.length += 1;

        Ok(entity)
    }

    pub fn despawn_on(&mut self, shards: &mut EntityShards, entity: Entity) -> Result<(), SpawnError> {
        let Some(location) = shards.get_location(entity) else {
            return Err(SpawnError::StaleEntity);
        };

        debug_assert_eq!(
            location.archetype, self.archetype_id,
            "the entity is not in this archetype."
        );

        let entity_chunk = location.chunk;
        let entity_row = location.row;

        let ok = shards.despawn(entity);
        if !ok { return Err(SpawnError::StaleEntity); }

        let mut moved_from: Option<(ChunkID, RowID)> = None;
        for component in self.components.iter_mut().filter_map(|c| c.as_mut()) {
            let position = component.swap_remove(entity_chunk, entity_row);
            if let Some(expected) = moved_from {
                assert_eq!(position, Some(expected), "all columns must move the same row");
            } else {
                moved_from = position;
            }
        }

        if let Some((moved_chunk, moved_row)) = moved_from {
            let moved_entity = match self.entity_positions[moved_chunk as usize][moved_row as usize] {
                Some(e) => e,
                None => {
                    debug_assert!(false, "moved slot should hold an entity; storage and positions out of sync.");
                    return Err(SpawnError::StaleEntity);
                }
            };
            self.entity_positions[entity_chunk as usize][entity_row as usize] = Some(moved_entity);

            shards.set_location(
                moved_entity,
                EntityLocation {
                    archetype: self.archetype_id,
                    chunk: entity_chunk,
                    row: entity_row,
                },
            );
            self.entity_positions[moved_chunk as usize][moved_row as usize] = None;
        } else {
            self.entity_positions[entity_chunk as usize][entity_row as usize] = None;
        }

        self.length -= 1;
        Ok(())
    }

    pub fn from_components<T: IntoIterator<Item = std::any::TypeId>>(archetype_id: ArchetypeID, types: T) -> Self {
        let mut me = Self::new(archetype_id);
        for type_id in types {
            let component_id = component_id_of_type_id(type_id)
                .expect("component type must be registered before creating archetypes.");
            let component: Box<dyn TypeErasedAttribute> = make_empty_component_for(component_id);
            me.insert_empty_component(component_id, component);
        }
        me
    }
}

impl Archetype {
    pub fn signature(&self) -> &Signature { &self.signature }

    pub fn matches_all(&self, need: &Signature) -> bool {
        self.signature.contains_all(need)
    }
}

fn make_empty_component_for(_component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    Box::new(crate::storage::Attribute::<()>::new())
}
