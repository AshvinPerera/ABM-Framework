use crate::types::{
    ArchetypeID, 
    ShardID,
    ChunkID,
    RowID, 
    CHUNK_CAP, 
    ComponentID, 
    COMPONENT_CAP,
    Signature,
    DynamicBundle
};
use crate::storage::{
    TypeErasedAttribute,
    Attribute
};
use crate::entity::{
    Entity, 
    EntityLocation, 
    EntityShards
};
use crate::component::{ 
    component_id_of_type_id,
    get_component_storage_factory
};
use crate::error::{
    SpawnError
};

//! # Archetype
//!
//! The `Archetype` type represents a storage container for all entities that
//! share an identical component signature. Each archetype owns a set of
//! component attributes, arranged in a column-major layout, where every attribute
//! stores values of a single component type.
//!
//! ## Operations
//!
//! Archetypes support:
//!
//! - Spawning entities into a new row across all component attribute.
//! - Removing entities and maintaining compactness through `swap_remove`.
//! - Moving entities between archetypes when their component signatures change.
//! - Borrowing chunk views for system execution with component-level read/write access.
//!
//! These operations maintain the required alignment invariants and ensure that
//! component data remains consistent with entity metadata.
//!
//! ## Invariants
//!
//! - All component attributes in an archetype share identical row counts.
//! - Component presence is determined solely by the archetype's `Signature`.
//! - Any row movement (via `push_from`, `swap_remove`, or de-spawn) must update
//!   `entity_positions` to remain consistent with component storage.
//!
//! Violating these invariants results in undefined entity/component alignment.
//!
//! ## Error Conditions
//!
//! Archetype operations produce `SpawnError` values when invariants cannot be
//! upheld, such as:
//!
//! - Missing components in a bundle during spawn.
//! - Storage push failures.
//! - Misaligned writes during spawn or move.
//! - De-spawning stale or untracked entities.
//!
//! ## Summary
//!
//! `Archetype` provides the core high-performance storage representation for
//! entities in the ECS. Its chunked, columnar design ensures locality, fast
//! traversal, predictable memory behavior, and compatibility with CPU/GPU
//! parallelization strategies.


pub struct ChunkBorrow<'a> {
    pub length: usize,
    pub reads: Vec<&'a [u8]>,
    pub writes: Vec<*mut u8>,
    pub _marker: std::marker::PhantomData<&'a mut u8>
}

pub struct MoveResult {
    pub source_position: (ChunkID, RowID),
    pub destination_position: (ChunkID, RowID),
    pub moved_components: Vec<ComponentID>
}

pub struct RemoveResult {
    pub source_position: (ChunkID, RowID),
    pub removed_components: Vec<ComponentID>,
    pub removed_attributes: Vec<Box<dyn Any>>
}

pub struct InsertResult {
    pub destination_position: (ChunkID, RowID),
    pub inserted_components: Vec<ComponentID>
}

#[derive(Debug)]
pub struct Archetype {
    archetype_id: ArchetypeID,
    components: Vec<Option<Box<dyn TypeErasedAttribute>>>,
    signature: Signature,
    length: usize,
    entity_positions: Vec<Vec<Option<Entity>>>
}

impl Archetype {

    /// Creates a new empty `Archetype` with the given identifier.
    ///
    /// ## Purpose
    /// Initializes component column storage, the signature bitset, entity tracking
    /// buffers, and internal counters.
    ///
    /// ## Behavior
    /// - Allocates `COMPONENT_CAP` component slots, all initially empty.
    /// - Initializes an empty `Signature`.
    /// - No component columns are allocated until explicitly inserted.
    ///
    /// ## Invariants
    /// The archetype contains no entities upon creation.

    pub fn new(archetype_id: ArchetypeID) -> Self {
            Self {
                archetype_id,
                components: vec![None; COMPONENT_CAP], // fixed-size component attribute slots
                signature: Signature::default(),
                length: 0,
                entity_positions: Vec::new(), // grows chunk-by-chunk on demand
            }
        }

    /// Returns the number of active entities stored in the archetype.
    ///
    /// ## Notes
    /// This reflects logical count only; physical chunk storage may contain unused rows.

    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the `ArchetypeID` associated with this archetype.
    ///
    /// ## Notes
    /// This value is stable for the lifetime of the archetype.

    pub fn archetype_id(&self) -> ArchetypeID {
        self.archetype_id 
    }

    /// Returns a reference to the archetype's signature.
    ///
    /// ## Notes
    /// Used by query and filtering logic.

    pub fn signature(&self) -> &Signature { &self.signature }

    /// Returns `true` if this archetype contains all components described in `need`.
    ///
    /// ## Notes
    /// This performs a subset check using signature bits.

    pub fn matches_all(&self, need: &Signature) -> bool {
        self.signature.contains_all(need)
    }

    /// Ensures that `entity_positions` contains at least `chunk_count` chunks.
    ///
    /// ## Purpose
    /// Expands chunk metadata storage to match component column allocations.
    ///
    /// ## Invariants
    /// - Each added chunk contains exactly `CHUNK_CAP` rows.
    /// - Does not allocate component data; only entity metadata.

    fn ensure_capacity(&mut self, chunk_count: usize) {
        // Ensure entity_positions always has a slot for every allocated chunk.
        while self.entity_positions.len() < chunk_count {
            self.entity_positions.push(vec![None; CHUNK_CAP]);
        }
    }

    /// Guarantees that a component attribute exists for the given `component_id`.
    ///
    /// ## Behavior
    /// - Allocates a new column using the provided factory if not already present.
    /// - Marks the component bit in the signature.
    ///
    /// ## Invariants
    /// Attribute allocation and signature must remain consistent.

    #[inline]
    pub fn ensure_component(&mut self, component_id: ComponentID, factory: impl FnOnce() -> Box<dyn TypeErasedAttribute>) -> Result<(), SpawnError>{
        // Lazily creates the column for a component type.
        let index = component_id as usize;
        if index >= COMPONENT_CAP { return Err(SpawnError::InvalidComponentId); }

        if self.components[index].is_none() {
            self.components[index] = Some(factory());
            self.signature.set(component_id);
        }
        Ok(())
    }

    /// Returns `true` if the archetype contains the specified component.
    ///
    /// ## Notes
    /// This checks the signature only; does not inspect the attribute buffer.

    #[inline]
    pub fn has(&self, component_id: ComponentID) -> bool {
        self.signature.has(component_id)
    }

    /// Returns an immutable reference to the component attribute for `component_id`,
    /// if present.
    ///
    /// ## Failure
    /// Returns `None` when the component is not part of the signature.

    #[inline]
    pub fn component(&self, component_id: ComponentID) -> Option<&dyn TypeErasedAttribute> {
        self.components.get(component_id as usize).and_then(|o| o.as_deref())
    }

    /// Returns a mutable reference to the component attribute for `component_id`,
    /// if present.
    ///
    /// ## Failure
    /// Returns `None` when the component is not part of the signature.
    ///
    /// ## Safety
    /// Caller must ensure no aliasing occurs with simultaneous borrows.

    #[inline]
    pub fn component_mut(&mut self, component_id: ComponentID) -> Option<&mut dyn TypeErasedAttribute> {
        self.components.get_mut(component_id as usize).and_then(|o| o.as_deref_mut())
    } 

    /// Computes how many chunks are required to store all active rows.
    ///
    /// ## Behavior
    /// - Returns `0` if no entities exist.
    /// - Otherwise computes `(length - 1) / CHUNK_CAP + 1`.

    pub fn chunk_count(&self) -> usize {
        if self.length == 0 {
            0
        } else {
            ((self.length - 1) / CHUNK_CAP) + 1 // last chunk may be partially full
        }
    }

    /// Returns the number of valid rows in the specified chunk.
    ///
    /// ## Behavior
    /// - Returns `0` if the chunk is unused.
    /// - Returns `CHUNK_CAP` for fully populated chunks.
    /// - Returns remaining entity count for the final partial chunk.
    ///
    /// ## Invariants
    /// Must reflect row count across all component attributes.

    pub fn chunk_valid_length(&self, chunk_index: usize) -> usize {
        // Returns how many rows in a chunk contain valid entities.
        if self.length == 0 || chunk_index > (self.length - 1) / CHUNK_CAP {
            0
        } else if chunk_index < (self.length - 1) / CHUNK_CAP {
            CHUNK_CAP
        } else {
            let used = self.length % CHUNK_CAP;
            if used == 0 { CHUNK_CAP } else { used } // possibly partial last chunk
        }
    }

    /// Inserts an empty component attribute into the archetype.
    ///
    /// ## Purpose
    /// Used when constructing archetypes from predefined type lists.
    ///
    /// ## Behavior
    /// - Fails if the index exceeds `COMPONENT_CAP`.
    /// - Assumes the attribute did not previously exist.
    /// - Sets the signature bit for the component.
    ///
    /// ## Invariants
    /// Component attributes must be added only before entities are inserted.

    pub fn insert_empty_component(&mut self, component_id: ComponentID, component: Box<dyn TypeErasedAttribute>) {
        let index = component_id as usize;
        if index >= COMPONENT_CAP {
            panic!("component_id out of range for COMPONENT_CAP.");
        }

        // Only safe to insert into an empty slot; archetype signature must match storage layout.
        let slot = &mut self.components[index];
        debug_assert!(slot.is_none(), "the component is already present.");
        *slot = Some(component);
        self.signature.set(component_id);
    }

    /// Removes a component attribute from an empty archetype.
    ///
    /// ## Behavior
    /// - Panics if the archetype still contains entities.
    /// - Clears the signature bit for the component.
    /// - Returns the removed attribute if present.
    ///
    /// ## Invariants
    /// Removing attributes in a populated archetype would break row alignment.

    pub fn remove_component(&mut self, component_id: ComponentID) -> Result<Option<Box<dyn TypeErasedAttribute>>, SpawnError> {
        // Components cannot be removed while entities exist�would break row alignment.
        if self.length > 0 {
            return Err(SpawnError::ArchetypeNotEmpty);
        } 

        let index = component_id as usize;
        let taken = self.components.get_mut(index)?.take();
        if taken.is_some() {
            self.signature.clear(component_id); // signature always matches stored columns
        }
        Ok(taken)
    }

    pub fn move_row_across_shared_components(
        &mut self,
        destination: &mut Archetype,
        source_chunk: ChunkID,
        source_row: RowID,
        shared_components: Vec<ComponentID>
    ) -> Result<MoveResult, MoveError> 
    {
        let mut source_position = (source_chunk, source_row);
        let mut destination_position: Option<(ChunkID, RowID)> = None;
        let mut moved_components = Vec::new();

        shared_components.into_iter().for_each(|component_id| {
            if !self.signature.has(component_id) | !destination.signature.has(component_id) {
                continue;
            }

            let source_component = self.components[component_id as usize]
                .get_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let destination_component = destination.components[component_id as usize]
                .get_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let ((destination_chunk, destination_row), swap_information) = destination_component
                .push_from(source_component, source_chunk, source_row)
                .map_err(|e| MoveError::PushFromFailed { component_id, source_error: e})?;

            moved_components.push(component_id);

            match destination_position {
                Some(position) if position != (destination_chunk, destination_row) => {
                    
                    return Err(MoveError::RowMisalignment {
                        expected: position,
                        got: (destination_chunk, destination_row),
                        component_id,
                    });
                }
                None => destination_position = Some((destination_chunk, destination_row)),
                _ => {}
            }

            if let Some(information) = swap_information {
                match first_swap_information {
                    Some(existing) if existing != information => {
                        return Err(MoveError::InconsistentSwapInfo);
                    }
                    None => first_swap_information = Some(information),
                    _ => {}
                }
            }
        });

        let destination_position = destination_position.ok_or(MoveError::NoComponentsMoved)?;

        Ok(MoveResult {
            destination_position,
            first_swap_information,
            moved_components,
        })
    }

    pub fn insert_row_in_components_at_destination(
        &mut self,
        destination_position: (ChunkID, RowID),
        added_components: &mut Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<MoveResult, MoveError> 
    {
        let mut written = Vec::new();
        let (destination_chunk, destination_row) = destination_position;

        for component_id in self.signature.iterate_over_components() {
            if self.signature.has(component_id) {
                continue;
            }

            // ensure the added value exists
            let position = added_components
                .iter()
                .position(|(cid, _)| *cid == component_id)
                .ok_or(MoveError::MissingAddedComponent(component_id))?;

            let (_, value) = added_components.remove(position);

            // write into destination column
            let destination_component = self.components[component_id as usize]
                .as_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let (chunk, row) = destination_component.push_dyn(value);

            if (chunk, row) != (destination_chunk, destination_row) {
                // rollback everything written so far
                for &cid in written.iter().rev() {
                    if let Some(c) = self.components[cid as usize].as_mut() {
                        let _ = c.swap_remove(destination_chunk, destination_row);
                    }
                }
                return Err(MoveError::RowMisalignment {
                    expected: (dest_chunk, dest_row),
                    got: (chunk, row),
                    component_id,
                });
            }

            written.push(component_id);
        }

        Ok(MoveResult {
            destination_position,
            None,
            written
        })
    }

    pub fn remove_row_in_components_at_source(
        &mut self,
        source_chunk: ChunkID,
        source_row: RowID,
    ) -> Result<MoveResult, MoveError> 
    {
        let mut removed_components = Vec::new();
        let mut unified_swap_info: Option<(ChunkID, RowID)> = None;

        for component_id in self.signature.iterate_over_components() {
            if self.signature.has(component_id) {
                continue;
            }

            let component = self.components[component_id as usize]
                .as_mut()
                .ok_or(MoveError::InconsistentStorage)?;

            let swap_info = component
                .swap_remove(source_chunk, source_row)
                .ok_or(MoveError::SwapRemoveError)?;

            removed_components.push(component_id);

            if let Some(info) = swap_info {
                match unified_swap_info {
                    Some(exist) if exist != info => return Err(MoveError::InconsistentSwapInfo),
                    None => unified_swap_info = Some(info),
                    _ => {}
                }
            }
        }

        Ok(MoveResult {
            unified_swap_info,
            removed_components,
        })
    }

    pub fn update_entity_on_row_move(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_chunk: ChunkID,
        source_row: RowID,
        destination_position: (ChunkID, RowID),
        unified_swap_info: Option<(ChunkID, RowID)>,
    ) -> Result<(), MoveError> 
    {
        let (dest_chunk, dest_row) = destination_position;

        destination.ensure_capacity(dest_chunk as usize + 1);
        destination.entity_positions[dest_chunk as usize][dest_row as usize] = Some(entity);

        shards.set_location(
            entity,
            EntityLocation {
                archetype: destination.archetype_id,
                chunk: dest_chunk,
                row: dest_row,
            },
        );

        match unified_swap_info {
            Some((last_chunk, last_row)) => {
                self.ensure_capacity(last_chunk as usize + 1);

                let swapped_entity = self.entity_positions[last_chunk as usize][last_row as usize]
                    .ok_or(MoveError::MetadataFailure)?;

                self.entity_positions[source_chunk as usize][source_row as usize] =
                    Some(swapped_entity);

                shards.set_location(
                    swapped_entity,
                    EntityLocation {
                        archetype: self.archetype_id,
                        chunk: source_chunk,
                        row: source_row,
                    },
                );

                self.entity_positions[last_chunk as usize][last_row as usize] = None;
            }
            None => {
                self.entity_positions[source_chunk as usize][source_row as usize] = None;
            }
        }

        Ok(())
    }

    fn rollback_push_from(
        &mut self,
        destination: &mut Archetype,
        component_id: ComponentID,
        destination_chunk: ChunkID,
        destination_row: RowID,
        source_chunk: ChunkID,
        source_row: RowID,
        swap_information: Option<(ChunkID, RowID)>
    ) -> Result<(), MoveError> {
        let source_component = self.components[component_id as usize]
            .get_mut()
            .ok_or(MoveError::InconsistentStorage)?;

        let destination_component = destination.components[component_id as usize]
            .get_mut()
            .ok_or(MoveError::InconsistentStorage)?;

    }

    fn rollback_move(
        &mut self,
        destination: &mut Archetype,
        result: &MoveResult,
    ) {
        let (chunk, row) = result.destination_position;

        for &component_id in result.moved_components.iter().rev() {
            if let Some(component) = destination.components[component_id as usize].as_mut() {
                let _ = component.swap_remove(chunk, row);
            }
        }
    }

    /// Moves an entity�s component row from this archetype to another.
    ///
    /// # Purpose
    /// This operation is used when an entity transitions to a new archetype
    /// because its set of components has changed (added or removed).
    ///
    /// The function constructs a new row in the destination archetype containing
    /// exactly the components described by the destination�s signature.
    ///
    /// # Behavior
    ///
    /// For each component type:
    ///
    /// - **If the component exists in both source and destination**  
    ///   The component value at `(source_chunk, source_row)` is moved into the
    ///   destination component column via `push_from`, preserving its internal
    ///   ordering guarantees (including swap�remove semantics).
    ///
    /// - **If the component exists in the destination but not in the source**  
    ///   A value for this component **must** be supplied in `added_components`.
    ///   That value is inserted using `push_dyn`.
    ///
    /// - **If the component exists in the source but not in the destination**  
    ///   The component value at `(source_chunk, source_row)` is discarded using
    ///   `swap_remove`, removing the row compactly from the source column.
    ///
    /// The first column to receive the moved or inserted value defines the
    /// destination `(chunk, row)` for this entity. All other component columns
    /// for the entity must place their data **at exactly the same location**.
    /// This preserves strict row alignment across all component arrays.
    ///
    /// After all component values are written:
    ///
    /// - The destination archetype�s `entity_positions` entry for the final
    ///   `(chunk, row)` is updated to record the entity�s ID.
    ///
    /// - The source archetype�s row at `(source_chunk, source_row)` is cleared.
    ///
    /// - If any source component column performed a swap�remove, the function
    ///   updates `entity_positions` and the global shard registry so the moved
    ///   entity now references the correct new position.
    ///
    /// - Archetype `length` counters are updated in both source and destination.

    pub fn move_row_to_new_archetype(
        &mut self,
        destination: &mut Archetype,
        shards: &EntityShards,
        entity: Entity,
        source_chunk: ChunkID,
        source_row: RowID,
        mut added_components: Vec<(ComponentID, Box<dyn Any>)>,
    ) -> Result<(ChunkID, RowID), MoveError> {
        let shared = self.move_shared_components(destination, source_chunk, source_row)?;

        let dest_pos = shared.destination_position;

        // ---- Phase 2: destination-only components ----
        let written_dest_only = match destination.insert_destination_only_components(
            dest_pos,
            &mut added_components,
        ) {
            Ok(w) => w,
            Err(e) => {
                self.rollback_shared_components(destination, &shared);
                return Err(e);
            }
        };

        // ---- Phase 3: remove source-only components ----
        let removed_source = match self.remove_source_only_components(source_chunk, source_row) {
            Ok(r) => r,
            Err(e) => {
                destination.rollback_added_destination_components(&written_dest_only, dest_pos);
                self.rollback_shared_components(destination, &shared);
                return Err(e);
            }
        };

        let combined_swap_info = removed_source.unified_swap_info.or(shared.unified_swap_info);

        // ---- Phase 4: metadata ----
        if let Err(e) = self.apply_metadata_updates(
            destination,
            shards,
            entity,
            source_chunk,
            source_row,
            dest_pos,
            combined_swap_info,
        ) {
            self.rollback_removed_source_components(&removed_source, source_chunk, source_row);
            destination.rollback_added_destination_components(&written_dest_only, dest_pos);
            self.rollback_shared_components(destination, &shared);
            return Err(e);
        }

        // ---- Success ----
        self.length -= 1;
        destination.length += 1;

        Ok(dest_pos)
    }

    /// Spawns a new entity into this archetype using the provided component bundle.
    ///
    /// ## Purpose
    /// Writes a full row of component values and allocates an entity handle.
    ///
    /// ## Behavior
    /// - Each component in the archetype�s signature must be supplied by the bundle.
    /// - All component attributes must write to the same `(chunk, row)` location.
    /// - On failure, all partial writes are rolled back.
    ///
    /// ## Errors
    /// - `MissingComponent` when the bundle does not contain a required value.
    /// - `StoragePushFailedWith` on backend storage errors.
    /// - `MisalignedStorage` when attributes disagree on row placement.
    /// - `EmptyArchetype` if no components exist.
    ///
    /// ## Invariants
    /// Attribute alignment and entity position mappings must remain consistent.

    pub fn spawn_on(&mut self, shards: &mut EntityShards, shard_id: ShardID, mut bundle: impl DynamicBundle) -> Result<Entity, SpawnError> {
        // Keep track of columns already written so that roll back is possible on error.
        let mut written_index: Vec<usize> = Vec::new();
        let mut reference_position: Option<(ChunkID, RowID)> = None;

        for (index, component_option) in self.components.iter_mut().enumerate() {
            let Some(component) = component_option.as_mut() else { continue };

            let component_id = index as ComponentID;

            // Identify the expected type so error messages are meaningful.
            let type_id = component.element_type_id();
            let name = component.element_type_name();

            // The bundle must contain every component required by signature.
            let Some(value) = bundle.take(component_id) else {
                // Roll back already-written components.
                if let Some((c, r)) = reference_position {
                    for &j in &written_index {
                        if let Some(s) = self.components[j].as_mut() {
                            let _ = s.swap_remove(c, r);
                        }
                    }
                }
                return Err(SpawnError::MissingComponent { type_id, name });
            };

            let position = match component.push_dyn(value) {
                Ok(p) => p,
                Err(e) => {
                    // Rollback on storage failure.
                    if let Some((c, r)) = reference_position {
                        for &j in &written_index {
                            if let Some(s) = self.components[j].as_mut() {
                                let _ = s.swap_remove(c, r);
                            }
                        }
                    }
                    return Err(SpawnError::StoragePushFailedWith(e));
                }
            };

            // Ensure all components push to identical coordinates.
            if let Some(rp) = reference_position {
                debug_assert_eq!(position, rp, "attributes must stay aligned per row.");
                if position != rp {
                    // Roll back if misalignment is detected.
                    for &j in &written_index {
                        if let Some(s) = self.components[j].as_mut() {
                            let _ = s.swap_remove(rp.0, rp.1);
                        }
                    }
                    return Err(SpawnError::MisalignedStorage { expected: rp, got: position });
                }
            } else {
                // First component defines row location for this entity.
                reference_position = Some(position);
            }

            written_index.push(index);
        }

        let Some((chunk, row)) = reference_position else {
            return Err(SpawnError::EmptyArchetype); // No components existed; invalid archetype
        };

        self.ensure_capacity(chunk as usize + 1);

        debug_assert!(
            self.entity_positions[chunk as usize][row as usize].is_none(),
            "spawn_on_bundle: target entity slot is already occupied."
        );

        let location = EntityLocation { archetype: self.archetype_id, chunk, row };

        // Allocate the actual entity handle.
        let entity = shards.spawn_on(shard_id, location).map_err(|e| {
            // Roll back component writes.
            for &j in &written_index {
                if let Some(s) = self.components[j].as_mut() {
                    let _ = s.swap_remove(chunk, row);
                }
            }
            e
        })?;

        self.entity_positions[chunk as usize][row as usize] = Some(entity);
        self.length += 1;

        Ok(entity)
    }

    /// Removes an entity from this archetype and maintains row compactness.
    ///
    /// ## Purpose
    /// Ensures component attributes remain dense by using `swap_remove`.
    ///
    /// ## Behavior
    /// - Updates the entity tracker to reflect despawn.
    /// - All component attributes must agree on the swapped row, if any.
    /// - Updates `entity_positions` for any entity moved via swap.
    ///
    /// ## Errors
    /// - `StaleEntity` when the entity does not exist.
    ///
    /// ## Invariants
    /// Component storage and entity metadata must remain synchronized.

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

        // Track whether swap_remove relocated another entity.
        let mut moved_from: Option<(ChunkID, RowID)> = None;

        for component in self.components.iter_mut().filter_map(|c| c.as_mut()) {
            // swap_remove keeps columns compact; all components must agree on moved row.
            let position = component.swap_remove(entity_chunk, entity_row)
                .expect("swap_remove failed in despawn");
            if let Some(expected) = moved_from {
                debug_assert_eq!(position, Some(expected), "all components must move the same row");
            } else {
                moved_from = position;
            }
        }

        if let Some((moved_chunk, moved_row)) = moved_from {
            // Update entity_positions to reflect the swap.
            let moved_entity = self.entity_positions[moved_chunk as usize][moved_row as usize]
                .expect("moved slot should hold an entity; storage and positions out of sync.");

            // Fill hole
            self.entity_positions[entity_chunk as usize][entity_row as usize] = Some(moved_entity);
            
            shards.set_location(moved_entity, EntityLocation {
                archetype: self.archetype_id, chunk: entity_chunk, row: entity_row
            });
            
            // Clear old swapped-from slot
            self.entity_positions[moved_chunk as usize][moved_row as usize] = None;
        } else {
            // No swap occurred; simply clear the slot.
            self.entity_positions[entity_chunk as usize][entity_row as usize] = None;
        }
        
        self.length -= 1;
        if self.length == 0 {
            self.entity_positions.clear();
        }
        Ok(())
    }

    /// Borrows the specified chunk for system execution, providing read and write
    /// access to component buffers.
    ///
    /// ## Purpose
    /// Allows systems to operate on tightly packed slices of component data.
    ///
    /// ## Behavior
    /// - `read_ids` produce immutable byte slices.
    /// - `write_ids` produce raw pointers for write access.
    /// - Caller must ensure all borrow and aliasing rules are upheld.
    ///
    /// ## Invariants
    /// Returned slices must correspond to valid rows of the requested chunk.

    pub fn borrow_chunk_for(
        &self,
        chunk: ChunkID,
        read_ids: &[ComponentID],
        write_ids: &[ComponentID],
    ) -> ChunkBorrow<'_> {
        // Borrow a chunk view for system execution; rows must be contiguous and type-aligned.
        let length = self.chunk_valid_length(chunk);
        let mut reads = Vec::with_capacity(read_ids.len());
        let mut writes = Vec::with_capacity(write_ids.len());

        for &component_id in read_ids {
            let component = self.components[component_id as usize].as_ref().expect("missing read component");
            reads.push(component.chunk_bytes_ref(chunk, length));
        }

        for &component_id in write_ids {
            let component = self.components[component_id as usize].as_ref().expect("missing write component");
            let bytes = component.chunk_bytes_mut(chunk, length);
            writes.push(bytes.as_mut_ptr());
        }

        ChunkBorrow { length, reads, writes, _marker: std::marker::PhantomData }
    }

    /// Constructs a new archetype and inserts empty attributes for the provided
    /// component types.
    ///
    /// ## Behavior
    /// - Each type ID must correspond to a registered component.
    /// - Component attributes are allocated via the storage factory.
    /// - No entities are created.
    ///
    /// ## Invariants
    /// The resulting archetype is empty but has a fully defined signature.

    pub fn from_components<T: IntoIterator<Item = std::any::TypeId>>(archetype_id: ArchetypeID, types: T) -> Self {
        let mut me = Self::new(archetype_id);

        // Create empty component columns for a predefined signature.
        for type_id in types {
            let component_id = component_id_of_type_id(type_id)
                .expect("component type must be registered before creating archetypes.");
            let component: Box<dyn TypeErasedAttribute> = make_empty_component_for(component_id);
            me.insert_empty_component(component_id, component);
        }
        me
    }
}

pub struct ArchetypeMatch {
    pub archetype_id: ArchetypeID,
    pub chunks: usize,
}

fn make_empty_component_for(component_id: ComponentID) -> Box<dyn TypeErasedAttribute> {
    get_component_storage_factory(component_id)()
}
