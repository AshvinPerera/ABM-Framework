//! ECS query description and construction.
//!
//! This module defines data structures and a builder-style API for
//! *describing* ECS queries: which components are required, which are read,
//! which are written, and which must be absent.

use crate::engine::types::{QuerySignature, ComponentID, AccessSets, set_read, set_write, set_without};
use crate::engine::component::{component_id_of};


/// An immutable, fully constructed ECS query description.
///
/// `BuiltQuery` is a **data-only representation** of a query after it has been
/// assembled by `QueryBuilder`.
///
/// It contains:
/// - a structural [`QuerySignature`] used for archetype matching,
/// - an ordered list of components accessed immutably,
/// - an ordered list of components accessed mutably.

#[derive(Clone)]
pub struct BuiltQuery {
    /// Structural query signature used to match archetypes.
    pub signature: QuerySignature,

    /// Component IDs accessed in read-only mode.
    pub reads: Vec<ComponentID>,

    /// Component IDs accessed mutably.
    pub writes: Vec<ComponentID>,
}

/// Builder for constructing ECS query descriptions.
///
/// `QueryBuilder` incrementally records:
/// - which components must be present,
/// - which components are read-only,
/// - which components are written,
/// - which components must be absent.
///
/// The builder follows a *builder-style* API and is typically consumed
/// by calling [`build`](Self::build) to produce a [`BuiltQuery`].

pub struct QueryBuilder {
    /// Structural and access-level query signature.
    signature: QuerySignature,

    /// Component IDs read by the query (in declaration order).
    reads: Vec<ComponentID>,

    /// Component IDs written by the query (in declaration order).
    writes: Vec<ComponentID>,
}

impl QueryBuilder {
    /// Creates a new, empty query builder.
    pub fn new() -> Self { Self { signature: QuerySignature::default(), reads: vec![], writes: vec![] } }

    /// Declares read-only access to component `T`.
    ///
    /// This:
    /// - marks `T` as a required component in the query signature,
    /// - records `T` as read-access for conflict analysis,
    /// - appends `T`’s component ID to the read list.
    
    pub fn read<T: 'static + Send + Sync>(mut self) -> Self {
        set_read::<T>(&mut self.signature);
        self.reads.push(component_id_of::<T>());
        self
    }
    /// Declares mutable access to component `T`.
    ///
    /// This:
    /// - marks `T` as a required component in the query signature,
    /// - records `T` as write-access for conflict analysis,
    /// - appends `T`’s component ID to the write list.

    pub fn write<T: 'static + Send + Sync>(mut self) -> Self {
        set_write::<T>(&mut self.signature);
        self.writes.push(component_id_of::<T>());
        self
    }


    /// Excludes component `T` from matching archetypes. 
    pub fn without<T: 'static + Send + Sync>(mut self) -> Self {
        set_without::<T>(&mut self.signature);
        self
    }

    /// Finalizes the query description and returns an immutable [`BuiltQuery`].  
    pub fn build(self) -> BuiltQuery {
        BuiltQuery {
            signature: self.signature,
            reads: self.reads,
            writes: self.writes,
        }
    }

    /// Returns the declared read/write access sets for this query.
    ///
    /// This is used by schedulers to detect conflicts between
    /// queries before execution.

    pub fn access_sets(&self) -> AccessSets {
        AccessSets { read: self.signature.read, write: self.signature.write }
    }
}
