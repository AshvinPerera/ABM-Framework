use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CapacityError {
    bits_needed: u64,
    capacity: u64,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter, "entity limit reached (bits needed {}, capacity {})", 
            self.bits_needed, 
            self.capacity
        )
    }
}

impl std::error::Error for CapacityError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShardBoundsError {
    index: u32,
    max_index: u32,
}

impl fmt::Display for ShardBoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter, "shard index {} out of bounds (max {})", 
            self.index, 
            self.max_index
        )
    }
}

impl std::error::Error for ShardBoundsError {}

