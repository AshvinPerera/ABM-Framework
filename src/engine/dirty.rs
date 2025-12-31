//! # Dirty Chunk Tracking
//!
//! This module provides chunk-granular dirty tracking for component columns.
//!
//! ## Purpose
//! Track which (archetype, component, chunk) ranges were written by CPU systems,
//! so the GPU backend can upload only the modified chunks.
//!
//! ## Concurrency
//! Writes occur inside parallel query execution (Rayon). This tracker is designed
//! to be thread-safe with minimal contention.
//!
//! ## Design
//! - Keyed by (ArchetypeID, ComponentID)
//! - Each entry stores a bitset of dirty chunk indices.
//! - Structural changes invalidate all entries.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::{RwLock, Arc};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::types::{ArchetypeID, ComponentID};

#[derive(Debug)]
struct Entry {
    // Number of chunks the bitset was built for.
    chunk_count: usize,
    // Bitset: 1 bit per chunk.
    words: Vec<AtomicU64>,
}

impl Entry {
    fn new(chunk_count: usize) -> Self {
        let word_count = (chunk_count + 63) / 64;
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(AtomicU64::new(0));
        }
        Self { chunk_count, words }
    }

    #[inline]
    fn mark_dirty(&self, chunk: usize) {
        let word = chunk / 64;
        let bytes = chunk % 64;
        if word >= self.words.len() { return; }
        let mask = 1u64 << bytes;
        self.words[word].fetch_or(mask, Ordering::Relaxed);
    }

    #[inline]
    fn mark_all_dirty(&self) {
        for word in &self.words {
            word.store(u64::MAX, Ordering::Relaxed);
        }
    }

    fn take_dirty_chunks_and_clear(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (word_index, word) in self.words.iter().enumerate() {
            let bits = word.swap(0, Ordering::AcqRel);
            if bits == 0 { continue; }

            let base = word_index * 64;
            for bit in 0..64 {
                if (bits >> bit) & 1 == 1 {
                    let index = base + bit;
                    if index < self.chunk_count {
                        out.push(index);
                    }
                }
            }
        }
        out
    }
}

/// Chunk-granular dirty tracker.
#[derive(Debug, Default)]
pub struct DirtyChunks {
    map: RwLock<HashMap<(ArchetypeID, ComponentID), Arc<Entry>>>,
}

impl DirtyChunks {
    /// Creates a new DirtyChunk Record
    pub fn new() -> Self {
        Self { map: RwLock::new(HashMap::new()) }
    }

    /// Called when structural changes occur (spawn/despawn/add/remove).
    /// Clears all prior entries.
    pub fn notify_world_changed(&self) {
        if let Ok(mut m) = self.map.write() {
            m.clear();
        }
    }

    /// Ensure an entry exists for (archetype, component) with the right chunk_count.
    fn ensure_entry(&self, archetype: ArchetypeID, component: ComponentID, chunk_count: usize) -> Arc<Entry> {
        // Fast path: read lock
        if let Ok(m) = self.map.read() {
            if let Some(e) = m.get(&(archetype, component)) {
                if e.chunk_count == chunk_count {
                    return e.clone();
                }
            }
        }

        // Slow path: write lock
        let mut m = self.map.write().expect("DirtyChunks lock poisoned");
        if let Some(e) = m.get(&(archetype, component)) {
            if e.chunk_count == chunk_count {
                return e.clone();
            }
        }

        let e = Arc::new(Entry::new(chunk_count));
        e.mark_all_dirty();
        m.insert((archetype, component), e.clone());
        e
    }

    /// Mark a specific chunk dirty (thread-safe).
    #[inline]
    pub fn mark_chunk_dirty(&self, archetype: ArchetypeID, component: ComponentID, chunk: usize, chunk_count: usize) {
        let e = self.ensure_entry(archetype, component, chunk_count);
        e.mark_dirty(chunk);
    }

    /// Mark all chunks dirty for a component in an archetype.
    pub fn mark_all_dirty(&self, archetype: ArchetypeID, component: ComponentID, chunk_count: usize) {
        let e = self.ensure_entry(archetype, component, chunk_count);
        e.mark_all_dirty();
    }

    /// Take and clear the dirty chunk list for this (archetype, component).
    pub fn take_dirty_chunks(&self, archetype: ArchetypeID, component: ComponentID, chunk_count: usize) -> Vec<usize> {
        let e = self.ensure_entry(archetype, component, chunk_count);
        e.take_dirty_chunks_and_clear()
    }
}
