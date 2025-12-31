//! # GPU Compute Pipeline Cache
//!
//! This module provides a **compute pipeline cache** for the GPU execution backend.
//! It is responsible for creating, storing, and reusing `wgpu::ComputePipeline`
//! objects and their associated `BindGroupLayout`s.
//!
//! ## Purpose
//!
//! * each ECS system is compiled into a GPU pipeline **at most once per binding layout**,
//! * pipelines are reused across frames and dispatches,
//! * bind group layouts remain stable and compatible with system access signatures.
//!
//! The cache is indexed by `(SystemID, binding_count)`, where `binding_count`
//! reflects the number of component buffers (reads + writes) plus a uniform
//! parameter buffer.
//!
//! ---
//!
//! ## Binding model
//!
//! Pipelines created by this module follow a strict binding convention:
//!
//! * Bindings `0..N-1` — storage buffers for component columns
//! * Binding `N` — uniform buffer containing per-dispatch parameters
//!
//! This layout is compatible with archetype-based dispatch, where each archetype
//! instance binds a different set of buffers while reusing the same pipeline.
//!
//! ---
//!
//! ## Safety and invariants
//!
//! * All pipelines are created for **compute-only** execution.
//! * Shader source is assumed to be valid WGSL.
//! * Binding layouts must exactly match the shader's declared bindings.
//! * Pipeline creation errors are surfaced as ECS execution errors.
//!
//! ---
//!
//! ## Usage
//!
//! The cache is owned by the GPU runtime and accessed during dispatch via
//! [`PipelineCache::get_or_create`]. Callers must ensure that:
//!
//! * `binding_count` matches the system's access signature,
//! * `shader_wgsl` and `entry_point` are consistent for a given `SystemID`.
//!

#![cfg(feature = "gpu")]

use std::collections::HashMap;

use crate::engine::error::{ECSResult, ECSError, ExecutionError};
use crate::engine::types::SystemID;

use crate::gpu::context::GPUContext;


/// Cache of GPU compute pipelines and their bind group layouts.
///
/// ## Role
/// Stores `wgpu::ComputePipeline` objects keyed by `(SystemID, binding_count)`,
/// allowing systems to reuse pipelines across dispatches.
///
/// ## Design
/// * One pipeline per system per binding layout
/// * Pipelines are created lazily on first use
/// * Layouts are stored alongside pipelines to guarantee compatibility
///
/// ## Thread safety
/// This type is not thread-safe by itself and must be externally synchronized
/// by the GPU runtime.

#[derive(Debug)]
pub struct PipelineCache {
    map: HashMap<(SystemID, usize, usize), (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}

impl PipelineCache {
    /// Creates an empty pipeline cache.
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Retrieves an existing compute pipeline or creates a new one.
    ///
    /// ## Parameters
    /// * `context` — GPU device context
    /// * `system_id` — ECS system identifier
    /// * `shader_wgsl` — WGSL compute shader source
    /// * `entry_point` — shader entry point function
    /// * `binding_count` — number of bind group entries
    ///
    /// ## Semantics
    /// * If a pipeline for `(system_id, binding_count)` exists, it is reused.
    /// * Otherwise, a new pipeline and bind group layout are created and cached.
    ///
    /// ## Errors
    /// Returns an error if pipeline creation fails, typically due to:
    /// * invalid WGSL source
    /// * binding layout mismatch
    /// * GPU driver or device errors

    pub fn get_or_create(
        &mut self,
        context: &GPUContext,
        system_id: SystemID,
        shader_wgsl: &'static str,
        entry_point: &'static str,
        read_count: usize,
        write_count: usize,
    ) -> ECSResult<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        let key = (system_id, read_count, write_count);

        if !self.map.contains_key(&key) {
            let (pipeline, bind_group_layout) =
                create_pipeline(context, shader_wgsl, entry_point, read_count, write_count)
                    .map_err(|e| {
                        ECSError::from(ExecutionError::GpuDispatchFailed { message: e.into() })
                    })?;

            self.map.insert(key, (pipeline, bind_group_layout));
        }

        let (compute_pipeline, layout) = self.map.get(&key).unwrap();
        Ok((compute_pipeline, layout))
    }
}

/// Creates a compute pipeline and its bind group layout.
///
/// ## Binding layout
/// * Storage buffers: `0..binding_count-1`
/// * Uniform buffer: `binding_count-1`
///
/// ## Parameters
/// * `context` — GPU device context
/// * `shader_wgsl` — WGSL shader source
/// * `entry_point` — compute entry point
/// * `binding_count` — total number of bindings
///
/// ## Errors
/// Returns an error string if pipeline creation fails.

fn create_pipeline(
    context: &GPUContext,
    shader_wgsl: &'static str,
    entry_point: &'static str,
    read_count: usize,
    write_count: usize,
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), String> {
    let binding_count = read_count + write_count + 1; // +1 for params
    if binding_count == 0 {
        return Err("create_pipeline: binding_count == 0".into());
    }

    let mut entries = Vec::with_capacity(binding_count);

    // Read-only storage bindings
    for i in 0..read_count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // Read-write storage bindings
    for j in 0..write_count {
        let binding = (read_count + j) as u32;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // Uniform params
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: (binding_count - 1) as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abm_bind_group_layout"),
        entries: &entries,
    });

    let pl = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("abm_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("abm_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_wgsl.into()),
    });

    let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("abm_compute_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok((pipeline, bgl))
}
