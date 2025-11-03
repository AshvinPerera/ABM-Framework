use std::{
    ptr,
    any::{Any, TypeId},
    any::{type_name, type_name_of_val},
    mem::MaybeUninit,
    convert::TryInto
};

use crate::types::{ChunkID, RowID, CHUNK_CAP};
use crate::error::{PositionOutOfBoundsError, TypeMismatchError, AttributeError};


pub trait TypeErasedAttribute: Any + Send + Sync {
    fn chunk_count(&self) -> usize;
    fn length(&self) -> usize;
    fn last_chunk_length(&self) -> usize;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any; 

    fn element_type_id(&self) -> TypeId;
    fn element_type_name(&self) -> &'static str;

    fn swap_remove(&mut self, chunk: ChunkID, row: RowID) -> Result<Option<(ChunkID, RowID)>, AttributeError>;
    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError>;
}

/// Invariant:
/// - All chunks before the last are fully initialized (CHUNK_CAP elements).
/// - Only the last chunk may be partially initialized with `last_chunk_length`.
/// - `length` is the total number of initialized elements.
pub struct Attribute<T> {
    chunks: Vec<Box<[MaybeUninit<T>; CHUNK_CAP]>>,
    last_chunk_length: usize, // number of initialized elements in the last chunk
    length: usize
}

impl<T> Default for Attribute<T> {
    fn default() -> Self {
        Self { chunks: Vec::new(), last_chunk_length: 0, length: 0 }
    }
}

impl<T> Attribute<T> {
    #[inline]
    fn ensure_last_chunk(&mut self) {
        if self.chunks.is_empty() || self.last_chunk_length == CHUNK_CAP {
            self.chunks.push(Box::new(std::array::from_fn(|_| MaybeUninit::<T>::uninit())));
            self.last_chunk_length = 0;
        }
    }

    #[inline]
    fn get_chunk_position(&self, index: usize) -> (ChunkID, RowID) {
        let chunk = (index / CHUNK_CAP) as ChunkID;
        let row = (index % CHUNK_CAP) as RowID;
        (chunk, row)
    }

    #[inline]
    fn get_slot_unchecked(&mut self, chunk: usize, row: usize) -> &mut MaybeUninit<T> {
        debug_assert!(chunk < self.chunk_count());
        debug_assert!(row < CHUNK_CAP);
        &mut self.chunks[chunk][row]
    }

    #[inline]
    fn to_ids(chunk: usize, row: usize) -> Result<(ChunkID, RowID), AttributeError> {
        let chunk: ChunkID = chunk.try_into().map_err(|_| AttributeError::IndexOverflow("ChunkID"))?;
        let row: RowID     = row.try_into().map_err(|_| AttributeError::IndexOverflow("RowID"))?;
        Ok((chunk, row))
    }  

    #[inline]
    fn valid_position(&self, chunk: ChunkID, row: RowID) -> bool {
        let chunk = chunk as usize;
        let row = row as usize;
        if chunk >= self.chunk_count() { return false; }
        if chunk + 1 == self.chunk_count() {
            row < self.last_chunk_length
        } else {
            row < CHUNK_CAP
        }
    }

    #[inline]
    fn last_filled_position(&self) -> Option<(usize, usize)> {
        if self.length == 0 { return None; }
        let index = self.length - 1;
        Some((index / CHUNK_CAP, index % CHUNK_CAP))
    }

    pub fn get(&self, chunk: ChunkID, row: RowID) -> Option<&T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_ref() })
    }

    pub fn get_mut(&mut self, chunk: ChunkID, row: RowID) -> Option<&mut T> {
        if !self.valid_position(chunk, row) { return None; }
        Some(unsafe { self.chunks[chunk as usize][row as usize].assume_init_mut() })
    }

    pub fn push(&mut self, v: T) -> Result<(ChunkID, RowID), AttributeError> {
        self.ensure_last_chunk();

        let chunk_us = self.chunks.len() - 1;
        let row_us   = self.last_chunk_length;

        // Safety: slot exists and is currently uninitialized
        unsafe { self.get_slot_unchecked(chunk_us, row_us).as_mut_ptr().write(v); }

        self.last_chunk_length += 1;
        self.length += 1;

        let (chunk_id, row_id) = Self::to_ids(chunk_us, row_us)?;
        Ok((chunk_id, row_id))
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) -> Result<(), AttributeError> {
        for v in iterator {
            self.push(v)?;
        }
        Ok(())
    }

    fn drop_all_initialized_elements(&mut self) {
        if self.length == 0 { return; }

        let mut remaining = self.length;
        for (chunk_idx, chunk) in self.chunks.iter_mut().enumerate() {
            let init_in_chunk = if chunk_idx == self.chunks.len() - 1 {
                self.last_chunk_length
            } else {
                CHUNK_CAP
            };

            let to_drop = init_in_chunk.min(remaining);
            for i in 0..to_drop {
                unsafe { chunk[i].assume_init_drop(); }
            }
            if remaining <= init_in_chunk { break; }
            remaining -= init_in_chunk;
        }
    }

    pub fn clear(&mut self) {
        if self.length == 0 { return; }

        self.drop_all_initialized_elements();
        self.chunks.clear();
        self.last_chunk_length = 0;
        self.length = 0;
    }    
}

impl<T: 'static + Send + Sync> TypeErasedAttribute for Attribute<T> {
    fn chunk_count(&self) -> usize { self.chunks.len() }
    fn length(&self) -> usize { self.length }
    fn last_chunk_length(&self) -> usize { self.last_chunk_length }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn element_type_id(&self) -> TypeId {TypeId::of::<T>()}
    fn element_type_name(&self) -> &'static str {type_name::<T>()}

    fn swap_remove(&mut self, chunk: ChunkID, row: RowID) -> Result<Option<(ChunkID, RowID)>, AttributeError> {
        let Some((last_chunk_us, last_row_us)) = self.last_filled_position() else {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: self.chunk_count(),
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        };

        if !self.valid_position(chunk, row) {
            return Err(AttributeError::Position(PositionOutOfBoundsError {
                chunk,
                row,
                chunks: self.chunk_count(),
                capacity: CHUNK_CAP,
                last_chunk_length: self.last_chunk_length,
            }));
        }

        let c = chunk as usize;
        let r = row as usize;

        if last_chunk_us == c && last_row_us == r {
            unsafe { self.get_slot_unchecked(last_chunk_us, last_row_us).assume_init_drop(); }
            self.length -= 1;
            self.last_chunk_length -= 1;

            if self.last_chunk_length == 0 {
                self.chunks.pop();
                if !self.chunks.is_empty() {
                    self.last_chunk_length = CHUNK_CAP;
                }
            }
            return Ok(None);
        } else {
            unsafe {
                let a = self.get_slot_unchecked(c, r).as_mut_ptr();
                let b = self.get_slot_unchecked(last_chunk_us, last_row_us).as_mut_ptr();
                ptr::swap(a, b);
                self.get_slot_unchecked(last_chunk_us, last_row_us).assume_init_drop();
            }

            self.length -= 1;
            self.last_chunk_length -= 1;
            if self.last_chunk_length == 0 {
                self.chunks.pop();
                if !self.chunks.is_empty() {
                    self.last_chunk_length = CHUNK_CAP;
                }
            }

            let moved_from = Self::to_ids(last_chunk_us, last_row_us)?;
            return Ok(Some(moved_from));
        }
    }

    fn push_dyn(&mut self, value: Box<dyn Any>) -> Result<(ChunkID, RowID), AttributeError> {
        if let Ok(v) = value.downcast::<T>() {
            return Attribute::<T>::push(self, *v);
        }
        let expected = type_name::<T>();
        let actual = type_name_of_val(&*value);
        Err(AttributeError::TypeMismatch(TypeMismatchError { expected, actual }))
    }
}

impl<T> Drop for Attribute<T> {
    fn drop(&mut self) {
        self.drop_all_initialized_elements();
    }
}
