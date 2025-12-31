#![cfg(feature = "gpu")]

use crate::engine::types::ComponentID;

/// Deterministic binding order.
///
/// Convention:
/// - all read buffers first (sorted by ComponentID)
/// - then all write buffers (sorted by ComponentID)
/// - then params uniform last

#[allow(dead_code)]
pub fn sort_component_ids(mut components: Vec<ComponentID>) -> Vec<ComponentID> {
    components.sort_unstable();
    components
}
