//! # GPU Dispatch Runtime
//!
//! This module defines the **GPU execution bridge** between the ECS scheduler
//! and the GPU backend.
//!
//! ## Purpose
//!
//! The dispatch runtime is responsible for:
//! * coordinating GPU execution of ECS systems implementing [`GpuSystem`],
//! * mirroring ECS archetype component data to GPU buffers,
//! * dispatching compute workloads per archetype,
//! * synchronizing GPU execution,
//! * and copying mutated component data back into ECS storage.
//!
//! This module is the **only location** where ECS state, GPU pipelines, and
//! command submission intersect.
//!
//! ## High-level execution flow
//!
//! For each GPU-capable system invocation:
//!
//! 1. Acquire **exclusive ECS access** (`with_exclusive`).
//! 2. Compute the union of read/write component signatures.
//! 3. Upload matching component columns to GPU buffers.
//! 4. Dispatch the GPU compute pipeline **per matching archetype**.
//! 5. Synchronize GPU execution.
//! 6. Download mutated component columns back into ECS storage.
//!
//! ## Design philosophy
//!
//! * **Single global GPU runtime**
//!   - GPU initialization and pipeline caches are shared globally.
//! * **Archetype-granular dispatch**
//!   - Each archetype is dispatched independently for predictable memory layout.
//! * **Explicit data movement**
//!   - All CPU to GPU transfers are explicit and phase-controlled.
//! * **Strict ECS invariants**
//!   - Structural mutation and parallel iteration are forbidden during GPU execution.
//!
//! ## Concurrency and safety model
//!
//! * GPU execution occurs inside `ECSReference::with_exclusive`.
//! * No ECS iteration may be active during GPU dispatch.
//! * Component borrows are enforced before upload and after download.
//! * GPU synchronization is explicit via `device.poll`.

#![cfg(feature = "gpu")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::engine::archetype::Archetype;
use crate::engine::component::Signature;
use crate::engine::error::{ECSError, ECSResult, ExecutionError};
use crate::engine::manager::ECSReference;
use crate::engine::systems::{GpuSystem, System};
use crate::engine::types::GPUAccessMode;

use crate::gpu::mirror::Mirror;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::GPUContext;
use crate::gpu::GPUResourceRegistry;

struct Runtime {
    context: GPUContext,
    mirror: Mirror,
    pipelines: PipelineCache,
    pending_download: Signature,
    params_buffer: Option<wgpu::Buffer>,
}

impl Runtime {
    #[inline]
    fn download_pending_into(
        &mut self,
        archetypes_mut: &mut [Archetype],
        pending: &Signature,
    ) -> ECSResult<()> {
        let ctx = &self.context;
        self.mirror.download_signature(ctx, archetypes_mut, pending)
    }
}

static RUNTIME: OnceLock<ECSResult<Mutex<Runtime>>> = OnceLock::new();

fn runtime() -> ECSResult<MutexGuard<'static, Runtime>> {
    let cell: &ECSResult<Mutex<Runtime>> = RUNTIME.get_or_init(|| {
        let run_time = Runtime {
            context: GPUContext::new()?,
            mirror: Mirror::new(),
            pipelines: PipelineCache::new(),
            pending_download: Signature::default(),
            params_buffer: None,
        };
        Ok(Mutex::new(run_time))
    });

    let mutex: &Mutex<Runtime> = match cell {
        Ok(matched_runtime) => matched_runtime,
        Err(e) => {
            return Err(ECSError::from(ExecutionError::GpuInitFailed {
                message: format!("{e:?}").into(),
            }));
        }
    };

    mutex
        .lock()
        .map_err(|_| ECSError::from(ExecutionError::LockPoisoned { what: "gpu runtime" }))
}

/// Synchronize any pending GPU downloads into ECS storage.
pub fn sync_pending_to_cpu(ecs: ECSReference<'_>) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let mut run_time = runtime()?;

        data.gpu_resources_mut().ensure_created(&run_time.context)?;
        data.gpu_resources_mut().download_pending(&run_time.context)?;

        let pending = {
            let p = run_time.pending_download.clone();
            if is_signature_empty(&p) {
                return Ok(());
            }
            p
        };

        let archetypes_mut: &mut [Archetype] = data.archetypes_mut();
        run_time.download_pending_into(archetypes_mut, &pending)?;

        run_time.pending_download = Signature::default();

        Ok(())
    })
}

/// Executes a single GPU-backed ECS system with exclusive world access.
pub fn execute_gpu_system(
    ecs: ECSReference<'_>,
    system: &dyn System,
    gpu: &dyn GpuSystem,
) -> ECSResult<()> {
    ecs.with_exclusive(|data| {
        let access = system.access();
        let read_signature = &access.read;
        let write_signature = &access.write;

        let union = union_signatures(read_signature, write_signature);

        let mut run_time = runtime()?;

        data.gpu_resources_mut().ensure_created(&run_time.context)?;
        data.gpu_resources_mut().upload_dirty(&run_time.context)?;

        {
            let archetypes: &[Archetype] = data.archetypes();

            {
                let Runtime { context, mirror, .. } = &mut *run_time;
                mirror.upload_signature_dirty_chunks(&*context, archetypes, &union, data.gpu_dirty_chunks())?;
            }

            dispatch_over_archetypes(
                &mut *run_time,
                system.id(),
                gpu,
                archetypes,
                &access,
                data.gpu_resources(),
            )?;
        }

        or_signature_in_place(&mut run_time.pending_download, write_signature);

        let writes = gpu.writes_resources();
        if !writes.is_empty() {
            for &resource_id in writes {
                data.gpu_resources_mut().mark_pending_download(resource_id);
            }
        } else {
            for &resource_id in gpu.uses_resources() {
                data.gpu_resources_mut().mark_pending_download(resource_id);
            }
        }

        Ok(())
    })
}

fn dispatch_over_archetypes(
    run_time: &mut Runtime,
    system_id: crate::engine::types::SystemID,
    gpu: &dyn GpuSystem,
    archetypes: &[Archetype],
    access: &crate::engine::systems::AccessSets,
    gpu_resources: &GPUResourceRegistry,
) -> ECSResult<()> {
    let mut resource_ids: Vec<_> = gpu.uses_resources().to_vec();
    resource_ids.sort_unstable();
    let resource_layout = gpu_resources.flattened_binding_descs(&resource_ids);

    let mut reads: Vec<crate::engine::types::ComponentID> =
        crate::engine::component::iter_bits_from_words(&access.read.components).collect();
    let mut writes: Vec<crate::engine::types::ComponentID> =
        crate::engine::component::iter_bits_from_words(&access.write.components).collect();
    
    writes.sort_unstable();
    reads.retain(|component_id| writes.binary_search(component_id).is_err());      
    reads.sort_unstable();

    let read_count = reads.len();
    let write_count = writes.len();

    let (pipeline, bgl0, bgl1_opt) = run_time.pipelines.get_or_create(
        &run_time.context,
        system_id,
        gpu.shader(),
        gpu.entry_point(),
        read_count,
        write_count,
        &resource_layout,
    )?;

    for archetype in archetypes {
        if !archetype.signature().contains_all(&access.read) || !archetype.signature().contains_all(&access.write) {
            continue;
        }

        let entity_len = archetype.length()? as u32;
        if entity_len == 0 {
            continue;
        }

        let mut entries0: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(read_count + write_count + 1);

        for (i, &component_id) in reads.iter().enumerate() {
            let archetype_id = archetype.archetype_id();

            let buffer = run_time
                .mirror
                .buffer_for(archetype_id, component_id)
                .ok_or_else(|| {
                    ECSError::from(ExecutionError::GpuMissingBuffer {
                        archetype_id,
                        component_id,
                        access: GPUAccessMode::Read,
                    })
                })?;

            entries0.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let base = reads.len();
        for (j, &component_id) in writes.iter().enumerate() {
            let archetype_id = archetype.archetype_id();

            let buffer = run_time
                .mirror
                .buffer_for(archetype_id, component_id)
                .ok_or_else(|| {
                    ECSError::from(ExecutionError::GpuMissingBuffer {
                        archetype_id,
                        component_id,
                        access: GPUAccessMode::Write,
                    })
                })?;

            entries0.push(wgpu::BindGroupEntry {
                binding: (base + j) as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            entity_len: u32,
            _p0: u32,
            _p1: u32,
            _p2: u32,
        }

        unsafe impl bytemuck::Pod for Params {}
        unsafe impl bytemuck::Zeroable for Params {}

        let params = Params {
            entity_len,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };

        let params_size = std::mem::size_of::<Params>() as u64;
        let recreate = match run_time.params_buffer.as_ref() {
            Some(buf) => buf.size() < params_size,
            None => true,
        };

        if recreate {
            run_time.params_buffer = Some(
                run_time.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("abm_params"),
                    size: params_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
        }

        let params_buf = run_time.params_buffer.as_ref().unwrap();

        // Update contents safely
        run_time
            .context
            .queue
            .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

        // Bind params buffer
        entries0.push(wgpu::BindGroupEntry {
            binding: (read_count + write_count) as u32,
            resource: params_buf.as_entire_binding(),
        });

        let bind_group0 = run_time.context.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("abm_bind_group_group0"),
                layout: bgl0,
                entries: &entries0,
            }
        );

        let bind_group1 = if let Some(bgl1) = bgl1_opt {
            let mut entries1: Vec<wgpu::BindGroupEntry> =
                Vec::with_capacity(resource_layout.len());

            gpu_resources.append_bind_group_entries(
                &resource_ids,
                0,
                &mut entries1,
            )?;

            println!("params_buf ptr = {:p}", params_buf);

            for e in &entries1 {
                if let wgpu::BindingResource::Buffer(b) = &e.resource {
                    println!(
                        "group1 binding={} buf_ptr={:p} usage={:?}",
                        e.binding,
                        b.buffer,
                        b.buffer.usage(),
                    );
                }
            }
            
            println!("params_buf usage = {:?}", params_buf.usage());

            Some(run_time.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("abm_bind_group_group1"),
                layout: bgl1,
                entries: &entries1,
            }))
        } else {
            None
        };

        let mut encoder = run_time
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("abm_compute_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("abm_compute_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group0, &[]);
            if let Some(bg1) = &bind_group1 {
                pass.set_bind_group(1, bg1, &[]);
            }

            let work_group_size = gpu.workgroup_size().max(1);
            let groups = (entity_len + work_group_size - 1) / work_group_size;
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let submission = run_time.context.queue.submit(Some(encoder.finish()));
        run_time
            .context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| {
                ECSError::from(ExecutionError::GpuDispatchFailed {
                    message: format!("wgpu device poll failed: {e:?}").into(),
                })
            })?;
    }

    Ok(())
}

fn union_signatures(a: &Signature, b: &Signature) -> Signature {
    let mut out = Signature::default();
    for ((o, av), bv) in out
        .components
        .iter_mut()
        .zip(a.components.iter())
        .zip(b.components.iter())
    {
        *o = *av | *bv;
    }
    out
}

#[inline]
fn or_signature_in_place(dst: &mut Signature, src: &Signature) {
    for (d, s) in dst.components.iter_mut().zip(src.components.iter()) {
        *d |= *s;
    }
}

#[inline]
fn is_signature_empty(sig: &Signature) -> bool {
    sig.components.iter().all(|&w| w == 0)
}
