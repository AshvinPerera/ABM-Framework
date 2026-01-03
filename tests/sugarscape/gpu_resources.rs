#![cfg(feature = "gpu")]

use abm_framework::gpu::{GPUResource, GPUBindingDesc, GPUContext};
use abm_framework::engine::error::ECSResult;

use wgpu::util::DeviceExt;

pub struct SugarGrid {
    #[allow(dead_code)]
    pub w: u32,
    #[allow(dead_code)]
    pub h: u32,
    sugar: wgpu::Buffer,
    capacity: wgpu::Buffer,
    occupancy: wgpu::Buffer,
}

impl SugarGrid {
    pub fn new(ctx: &GPUContext, w: u32, h: u32, cap: &[f32]) -> Self {
        let sugar = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.sugar"),
            contents: bytemuck::cast_slice(cap),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let capacity = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.capacity"),
            contents: bytemuck::cast_slice(cap),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let zeros = vec![0u32; (w * h) as usize];
        let occupancy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.occupancy"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self { w, h, sugar, capacity, occupancy }
    }
}

impl GPUResource for SugarGrid {
    fn name(&self) -> &'static str { "SugarGrid" }
    fn create_gpu(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn upload(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn download(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }

    fn bindings(&self) -> &[GPUBindingDesc] {
        static B: [GPUBindingDesc; 3] = [
            GPUBindingDesc { read_only: false }, // sugar
            GPUBindingDesc { read_only: true  }, // capacity
            GPUBindingDesc { read_only: false }, // occupancy (atomic<u32>)
        ];
        &B
    }

    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        out.push(wgpu::BindGroupEntry { binding: base + 0, resource: self.sugar.as_entire_binding() });
        out.push(wgpu::BindGroupEntry { binding: base + 1, resource: self.capacity.as_entire_binding() });
        out.push(wgpu::BindGroupEntry { binding: base + 2, resource: self.occupancy.as_entire_binding() });
        Ok(())
    }
}

pub struct AgentIntentBuffers {
    agent_target: wgpu::Buffer,
}

impl AgentIntentBuffers {
    pub fn new(ctx: &GPUContext, agent_capacity: usize) -> Self {
        let zeros = vec![0u32; agent_capacity];
        let agent_target = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("intent.agent_target"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Self { agent_target }
    }
}

impl GPUResource for AgentIntentBuffers {
    fn name(&self) -> &'static str { "AgentIntentBuffers" }
    fn create_gpu(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn upload(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn download(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }

    fn bindings(&self) -> &[GPUBindingDesc] {
        static B: [GPUBindingDesc; 1] = [
            GPUBindingDesc { read_only: false }, // agent_target
        ];
        &B
    }

    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        out.push(wgpu::BindGroupEntry { binding: base + 0, resource: self.agent_target.as_entire_binding() });
        Ok(())
    }
}