//! ECS system scheduling and execution.
//!
//! This module is responsible for:
//! * grouping systems into execution stages based on access compatibility,
//! * running compatible systems in parallel using Rayon,
//! * enforcing structural synchronization points between stages.
//!
//! ## Scheduling model
//!
//! Systems are assigned to **stages** such that:
//! * systems within the same stage do **not** conflict on component access,
//! * all systems in a stage may run in parallel,
//! * stages are executed sequentially.
//!
//! This allows maximal parallelism while preserving safety guarantees
//! derived from declared read/write access sets.
//!
//! ## Structural synchronization
//!
//! Deferred ECS commands (spawns, despawns, component mutations) are applied:
//! * **before** each stage begins,
//! * **after** each stage completes.


use rayon::prelude::*;

use crate::engine::manager::ECSManager;
use crate::engine::systems::{System, SystemBackend};
use crate::engine::types::AccessSets;


/// An execution stage used by [`Scheduler`].
///
/// Stores *indices* into the scheduler's system list. This lets the
/// scheduler:
/// - keep systems registered for repeated ticks,
/// - rebuild plans when systems are added/removed,
/// - evolve into a CPU/GPU multi-backend dispatcher.
#[derive(Clone, Debug, Default)]
pub struct Stage {
    /// Indices of systems that can execute in parallel.
    pub system_indices: Vec<usize>,
    /// Aggregate access sets of systems in this stage (used for fast conflict checks
    /// during plan construction).
    aggregate_access: AccessSets,
}

impl Stage {
    /// Returns true if `access` does NOT conflict with anything already in this stage.
    #[inline]
    pub fn can_accept(&self, access: &AccessSets) -> bool {
        !access.conflicts_with(&self.aggregate_access)
    }

    /// Adds a system index to this stage and merges its access into the aggregate.
    #[inline]
    pub fn push(&mut self, idx: usize, access: &AccessSets) {
        self.system_indices.push(idx);
        or_signature_in_place(&mut self.aggregate_access.read, &access.read);
        or_signature_in_place(&mut self.aggregate_access.write, &access.write);
    }

    /// Returns true if this stage is acting as a boundary marker.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        self.system_indices.is_empty()
    }
}

/// Scheduler that stores systems, compiles them into conflict-free execution stages,
/// and executes stages with parallelism.
pub struct Scheduler {
    systems: Vec<Box<dyn System>>,
    /// Cached CPU stages.
    cpu_stages: Vec<Stage>,
    /// Whether `cpu_stages` needs rebuilding.
    dirty: bool,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    /// Creates an empty scheduler.
    #[inline]
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            cpu_stages: Vec::new(),
            dirty: true,
        }
    }

    /// Returns the number of registered systems.
    #[inline]
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Returns `true` if no systems are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Removes all systems and stages.
    #[inline]
    pub fn clear(&mut self) {
        self.systems.clear();
        self.cpu_stages.clear();
        self.dirty = true;
    }

    /// Registers a boxed system.
    #[inline]
    pub fn add_boxed(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
        self.dirty = true;
    }

    /// Registers a concrete system.
    #[inline]
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.add_boxed(Box::new(system));
    }

    /// Registers a function-backed system.
    pub fn add_fn_system<F>(
        &mut self,
        system: crate::engine::systems::FnSystem<F>,
    ) where
        F: Fn(crate::engine::manager::ECSReference<'_>) + Send + Sync + 'static,
    {
        self.add_system(system);
    }

    /// Convenience helper to build-and-register an [`FnSystem`](crate::engine::systems::FnSystem)
    #[inline]
    pub fn add_fn<F>(
        &mut self,
        id: crate::engine::types::SystemID,
        name: &'static str,
        access: AccessSets,
        f: F,
    ) where
        F: Fn(crate::engine::manager::ECSReference<'_>) + Send + Sync + 'static,
    {
        self.add_fn_system(crate::engine::systems::FnSystem::new(id, name, access, f));
    }

    /// Ensures stages are up to date.
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }

        // Deterministic: sort indices by system ID.
        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        indices.sort_by_key(|&i| self.systems[i].id());

        self.cpu_stages.clear();

        for idx in indices {
            let sys = &self.systems[idx];

            // For GPU scaling: keep GPU systems as hard boundaries.
            if sys.backend() == SystemBackend::GPU {
                // Flush any pending CPU stage and create a boundary stage.
                self.cpu_stages.push(Stage::default());
                continue;
            }

            let access = sys.access();

            // Greedy packing into the first compatible stage.
            let mut placed = false;
            for stage in self.cpu_stages.iter_mut() {
                // Empty stages are boundaries; don't pack across them.
                if stage.system_indices.is_empty() {
                    continue;
                }
                if stage.can_accept(&access) {
                    stage.push(idx, &access);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let mut stage = Stage::default();
                stage.push(idx, &access);
                self.cpu_stages.push(stage);
            }
        }

        self.dirty = false;
    }

    /// Runs the schedule once.
    ///
    /// This will:
    /// 1) rebuild the plan if needed,
    /// 2) execute each stage sequentially,
    /// 3) run systems within a stage in parallel,
    /// 4) apply deferred commands before and after each stage.
    pub fn run(&mut self, ecs: &ECSManager) {
        self.rebuild();

        for stage in &self.cpu_stages {
            ecs.apply_deferred_commands();

            // Boundary / future GPU stage marker.
            if stage.system_indices.is_empty() {
                // In a GPU-enabled build, this is where GPU kernels would run.
                ecs.apply_deferred_commands();
                continue;
            }

            stage.system_indices.par_iter().for_each(|&system_idx| {
                let world = ecs.world_ref();
                self.systems[system_idx].run(world);
            });

            ecs.apply_deferred_commands();
        }
    }

    /// Returns a lightweight view of the CPU stages.
    pub fn cpu_stages(&mut self) -> &[Stage] {
        self.rebuild();
        &self.cpu_stages
    }
}

#[inline]
fn or_signature_in_place(
    dst: &mut crate::engine::types::Signature,
    src: &crate::engine::types::Signature
) {
    for (d, s) in dst.components.iter_mut().zip(src.components.iter()) {
        *d |= *s;
    }
}
