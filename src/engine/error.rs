use std::fmt;

use crate::types::{ChunkID, RowID, CHUNK_CAP};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityError {
    pub entities_needed: u64,
    pub capacity: u64,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter, "entity limit reached ({} entities needed, but capacity is {})", 
            self.entities_needed, 
            self.capacity
        )
    }
}

impl std::error::Error for CapacityError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardBoundsError {
    pub index: u32,
    pub max_index: u32,
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

#[derive(Debug)]
pub enum SpawnError {
    Capacity(CapacityError),
    ShardBounds(ShardBoundsError),
    StaleEntity,
    EmptyArchetype,
    StoragePushFailed
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::Capacity(e) => write!(f, "{e}"),
            SpawnError::ShardBounds(e) => write!(f, "{e}"),
            SpawnError::StaleEntity => write!(f, "stale entity handle"),
            SpawnError::EmptyArchetype => write!(f, "cannot spawn into an empty archetype"),
            SpawnError::StoragePushFailed => write!(f, "failed to push entity into storage")
        }
    }
}

impl From<CapacityError> for SpawnError {
    fn from(e: CapacityError) -> Self { SpawnError::Capacity(e) }
}
impl From<ShardBoundsError> for SpawnError {
    fn from(e: ShardBoundsError) -> Self { SpawnError::ShardBounds(e) }
}

impl std::error::Error for SpawnError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionOutOfBoundsError {
    pub chunk: ChunkID,
    pub row: RowID,
    pub chunks: usize,
    pub capacity: usize,
    pub last_chunk_length: usize,
}

impl fmt::Display for PositionOutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "position ({}, {}) out of bounds (chunks:{}, capacity:{}, last chunk length:{})",
            self.chunk, self.row, self.chunks, self.capacity, self.last_chunk_length
        )
    }
}

impl std::error::Error for PositionOutOfBoundsError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeMismatchError {
    pub expected: &'static str,
    pub actual: &'static str,
}

impl fmt::Display for TypeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type mismatch: expected `{}`, got `{}`", self.expected, self.actual)
    }
}

impl std::error::Error for TypeMismatchError {}

#[derive(Debug)]
pub enum AttributeError {
    Position(PositionOutOfBoundsError),
    TypeMismatch(TypeMismatchError),
    IndexOverflow(&'static str),
}

impl fmt::Display for AttributeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeError::Position(e) => write!(f, "{e}"),
            AttributeError::TypeMismatch(e) => write!(f, "{e}"),
            AttributeError::IndexOverflow(what) => write!(f, "index overflow for {what}"),
        }
    }
}

impl std::error::Error for AttributeError {}
