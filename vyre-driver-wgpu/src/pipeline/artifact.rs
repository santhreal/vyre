use std::sync::Arc;

use vyre_driver::OutputBindingLayout;
use vyre_driver::{IndirectDispatch, OutputLayout};
use vyre_emit_naga::program::TrapTag;
use vyre_foundation::execution_plan::ExecutionPlan;

use super::descriptor_metadata::BufferBindingInfo;
use crate::buffer::{BindGroupCache, StagingBufferPool};

/// Target text that has already been emitted and authenticated for `program`.
///
/// Held as one value so a caller cannot supply text without the descriptor it
/// was emitted from: the descriptor decides the workgroup override the text
/// was built against.
pub(crate) struct AuthenticatedTarget<'a> {
    pub(crate) wgsl: &'a str,
    pub(crate) descriptor: &'a vyre_lower::KernelDescriptor,
    pub(crate) resource_bindings: &'a [vyre_megakernel::TargetResourceBinding],
}

/// GPU pipeline + **all** per-program dispatch metadata co-located for
/// cache hits. A hit on [`early_pipeline_cache_key`] or the WGSL hash
/// key must skip `execution_plan::plan`, output-layout derivation,
/// and fresh [`StagingBufferPool::new`] (subagent: pipeline.rs compile
/// path  -  2026-04 orchestration sweep).
#[derive(Debug)]
pub(crate) struct CachedPipelineArtifact {
    pub(crate) id: String,
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bind_group_layouts: Arc<[Arc<wgpu::BindGroupLayout>]>,
    pub(crate) bind_group_cache: Arc<BindGroupCache>,
    /// Shared across every [`WgpuPipeline`] built from this artifact.
    pub(crate) execution_plan: Arc<ExecutionPlan>,
    pub(crate) output_bindings: Arc<[OutputBindingLayout]>,
    pub(crate) buffer_bindings: Arc<[BufferBindingInfo]>,
    pub(crate) output: OutputLayout,
    pub(crate) output_word_count: usize,
    pub(crate) workgroup_shape: [u32; 3],
    pub(crate) workgroup_size: u32,
    pub(crate) indirect: Option<IndirectDispatch>,
    pub(crate) trap_tags: Arc<[TrapTag]>,
    /// Cloned per [`WgpuPipeline`]; all clones share the inner pool.
    pub(crate) staging_pool: StagingBufferPool,
}

impl CachedPipelineArtifact {
    pub(crate) fn cache_cost_bytes(&self) -> usize {
        let binding_names: usize = self
            .buffer_bindings
            .iter()
            .map(|binding| binding.name.len())
            .sum();
        let output_names: usize = self
            .output_bindings
            .iter()
            .map(|output| output.name.len())
            .sum();
        checked_cache_cost_sum(&[
            self.id.len(),
            binding_names,
            output_names,
            checked_cache_cost_product(
                self.bind_group_layouts.len(),
                std::mem::size_of::<Arc<wgpu::BindGroupLayout>>(),
            ),
            checked_cache_cost_product(
                self.buffer_bindings.len(),
                std::mem::size_of::<BufferBindingInfo>(),
            ),
            checked_cache_cost_product(
                self.output_bindings.len(),
                std::mem::size_of::<OutputBindingLayout>(),
            ),
            checked_cache_cost_product(self.trap_tags.len(), std::mem::size_of::<TrapTag>()),
            std::mem::size_of::<Self>(),
        ])
    }
}

fn checked_cache_cost_product(count: usize, element_size: usize) -> usize {
    count.saturating_mul(element_size)
}

fn checked_cache_cost_sum(parts: &[usize]) -> usize {
    let mut total = 0usize;
    for &part in parts {
        total = total.saturating_add(part);
    }
    total
}
