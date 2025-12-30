#![cfg(feature = "gpu")]

use std::collections::HashMap;

use crate::engine::archetype::Archetype;
use crate::engine::component::{component_description_by_component_id, iter_bits_from_words, Signature};
use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::engine::types::{ArchetypeID, ComponentID, ChunkID};

use crate::gpu::context::GPUContext;

#[derive(Debug)]
struct BufferEntry {
    buffer: wgpu::Buffer,
    bytes: usize,
}

#[derive(Debug)]
pub struct Mirror {
    buffers: HashMap<(ArchetypeID, ComponentID), BufferEntry>,
}

impl Mirror {
    pub fn new() -> Self {
        Self { buffers: HashMap::new() }
    }

    pub fn ensure_gpu_safe(&self, component_id: ComponentID) -> ECSResult<(usize, &'static str)> {
        let description = component_description_by_component_id(component_id)?
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id }))?;

        if !description.gpu_usage {
            return Err(ECSError::from(ExecutionError::GpuUnsupportedComponent {
                component_id,
                name: description.name,
            }));
        }
        Ok((description.size, description.name))
    }

    pub fn upload_signature(&mut self, context: &GPUContext, archetypes: &[Archetype], signature: &Signature) -> ECSResult<()> {
        let mut component_ids: Vec<ComponentID> = iter_bits_from_words(&signature.components).collect();
        component_ids.sort_unstable();

        for archetype in archetypes {
            for &component_id in &component_ids {
                if archetype.has(component_id) {
                    let (size, _) = self.ensure_gpu_safe(component_id)?;
                    self.upload_column(context, archetype, component_id, size)?;
                }
            }
        }
        Ok(())
    }

    pub fn download_signature(&mut self, context: &GPUContext, archetypes: &mut [Archetype], signature: &Signature) -> ECSResult<()> {
        let mut component_ids: Vec<ComponentID> = iter_bits_from_words(&signature.components).collect();
        component_ids.sort_unstable();

        for archetype in archetypes {
            for &component_id in &component_ids {
                if archetype.has(component_id) {
                    let (size, _) = self.ensure_gpu_safe(component_id)?;
                    self.download_column(context, archetype, component_id, size)?;
                }
            }
        }
        Ok(())
    }

    pub fn buffer_for(&self, archetype: ArchetypeID, component_id: ComponentID) -> Option<&wgpu::Buffer> {
        self.buffers.get(&(archetype, component_id)).map(|e| &e.buffer)
    }

    fn ensure_buffer(&mut self, context: &GPUContext, archetype: ArchetypeID, component_id: ComponentID, bytes: usize) {
        let key = (archetype, component_id);
        let create = match self.buffers.get(&key) {
            None => true,
            Some(e) => e.bytes < bytes,
        };

        if create {
            let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("abm_component_storage"),
                size: bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.buffers.insert(key, BufferEntry { buffer, bytes });
        }
    }

    fn upload_column(&mut self, context: &GPUContext, archetype: &Archetype, component_id: ComponentID, component_size: usize) -> ECSResult<()> {
        let len = archetype.length()?;
        if len == 0 { return Ok(()); }

        let bytes_total = len * component_size;
        self.ensure_buffer(context, archetype.archetype_id(), component_id, bytes_total);

        let storage = self.buffers.get(&(archetype.archetype_id(), component_id)).unwrap();

        let locked = archetype.component_locked(component_id)
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id: component_id }))?;
        let guard = locked.read()
            .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what: "attribute read lock (upload)" }))?;

        let mut host = vec![0u8; bytes_total];
        let mut offset = 0usize;

        let chunks = archetype.chunk_count()?;
        for chunk in 0..chunks {
            let valid = archetype.chunk_valid_length(chunk)?;
            if valid == 0 { continue; }

            let (pointer, bytes) = guard.chunk_bytes(chunk as ChunkID, valid)
                .ok_or_else(|| ECSError::Internal("chunk_bytes returned None".into()))?;
            let source = unsafe { std::slice::from_raw_parts(pointer, bytes) };

            host[offset..offset + bytes].copy_from_slice(source);
            offset += bytes;
        }

        context.queue.write_buffer(&storage.buffer, 0, &host);
        Ok(())
    }

    fn download_column(&mut self, context: &GPUContext, archetype: &mut Archetype, component_id: ComponentID, component_size: usize) -> ECSResult<()> {
        let len = archetype.length()?;
        if len == 0 { return Ok(()); }

        let bytes_total = len * component_size;
        self.ensure_buffer(context, archetype.archetype_id(), component_id, bytes_total);

        let storage = self.buffers.get(&(archetype.archetype_id(), component_id)).unwrap();

        let staging = context.device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("abm_readback_staging"),
                size: bytes_total as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let mut encoder = context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("abm_readback_encoder"),
            }
        );

        encoder.copy_buffer_to_buffer(&storage.buffer, 0, &staging, 0, bytes_total as u64);
        context.queue.submit(Some(encoder.finish()));

        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("wgpu poll failed after copy: {e:?}").into(),
                })
            })?;

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });

        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("wgpu poll failed during map_async: {e:?}").into(),
                })
            })?;


        receiver.recv().ok().transpose().map_err(|_| {
            ECSError::from(ExecutionError::GpuDispatchFailed { message: "failed to map readback buffer".into() })
        })?;

        let data = slice.get_mapped_range();
        let host: &[u8] = &data;

        let locked = archetype.component_locked(component_id)
            .ok_or_else(|| ECSError::from(ExecutionError::MissingComponent { component_id: component_id }))?;
        let mut guard = locked.write()
            .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what: "attribute write lock (download)" }))?;

        let mut offset = 0usize;
        let chunks = archetype.chunk_count()?;
        for chunk in 0..chunks {
            let valid = archetype.chunk_valid_length(chunk)?;
            if valid == 0 { continue; }

            let (pointer, bytes) = guard.chunk_bytes_mut(chunk as ChunkID, valid)
                .ok_or_else(|| ECSError::Internal("chunk_bytes_mut returned None".into()))?;
            let destination = unsafe { std::slice::from_raw_parts_mut(pointer, bytes) };

            destination.copy_from_slice(&host[offset..offset + bytes]);
            offset += bytes;
        }

        drop(data);
        staging.unmap();
        Ok(())
    }
}
