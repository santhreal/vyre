use super::{
    enforce_actual_output_budget, wgpu_effective_dispatch_config_for_limits, BindGroupLayoutCache,
    DispatchConfig, WgpuPipeline,
};
use vyre_driver::tuner::Mode;
use vyre_driver::validation::LaunchGeometryLimits;
use vyre_foundation::execution_plan::{self, ReadbackStrategy};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryKind, Node, Program};

use std::hash::BuildHasherDefault;
use std::sync::Arc;

use crate::buffer::BufferPool;
use crate::engine::record_and_readback::{record_and_readback, DispatchLabels, RecordAndReadback};
use crate::runtime::cache::pipeline::LruPipelineCache;
use crate::runtime::device::EnabledFeatures;
use crate::DispatchArena;
use vyre_driver::pipeline::DEFAULT_PIPELINE_CACHE_ENTRIES;
use vyre_driver::BackendError;

/// Device, queue, dispatch config and the two compile caches every pipeline
/// contract test needs. Each test used to spell this block out again.
struct PipelineHarness {
    device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
    adapter_info: wgpu::AdapterInfo,
    enabled_features: EnabledFeatures,
    config: DispatchConfig,
    pipeline_cache: Arc<LruPipelineCache>,
    layout_cache: Arc<BindGroupLayoutCache>,
}

impl PipelineHarness {
    /// `purpose` completes "Fix: GPU required for {purpose}" when no device opens.
    fn new(purpose: &str) -> Self {
        let ((device, queue), adapter_info, enabled_features) = crate::runtime::init_device()
            .unwrap_or_else(|err| panic!("Fix: GPU required for {purpose}: {err:?}"));
        Self {
            device_queue: Arc::new((device, queue)),
            adapter_info,
            enabled_features,
            config: DispatchConfig::default(),
            pipeline_cache: Arc::new(LruPipelineCache::new(DEFAULT_PIPELINE_CACHE_ENTRIES as u32)),
            layout_cache: Arc::new(BindGroupLayoutCache::with_hasher(BuildHasherDefault::<
                rustc_hash::FxHasher,
            >::default())),
        }
    }

    /// A dispatch arena over this harness's device and queue.
    fn arena(&self) -> Arc<DispatchArena> {
        Arc::new(DispatchArena::new(
            self.device_queue.0.clone(),
            self.device_queue.1.clone(),
            &self.config,
        ))
    }

    /// Compile against the shared caches, binding `pool` as the persistent pool.
    fn compile(
        &self,
        program: &Program,
        pool: BufferPool,
    ) -> Result<Arc<WgpuPipeline>, BackendError> {
        WgpuPipeline::compile_with_device_queue(
            program,
            &self.config,
            self.adapter_info.clone(),
            self.enabled_features,
            self.device_queue.clone(),
            pool,
            self.pipeline_cache.clone(),
            self.layout_cache.clone(),
            None,
        )
    }

    /// Compile against the pool `arena` owns, so buffer Arc identities match
    /// between compile-time bindings and run-time recording. A separate
    /// `BufferPool::new()` would make every dispatch a bind-group-cache miss.
    fn compile_on_arena(
        &self,
        program: &Program,
        arena: &Arc<DispatchArena>,
    ) -> Result<Arc<WgpuPipeline>, BackendError> {
        self.compile(program, arena.pool().clone())
    }
}

/// One direct dispatch through the shared record path. Every pipeline contract
/// test issues a single unprofiled 1x1x1 dispatch with no inputs over the
/// arena's own pool; only the debug labels and whether readback rings are in
/// play differ.
fn record_once(
    pipeline: &WgpuPipeline,
    arena: &DispatchArena,
    readback_rings: bool,
    labels: DispatchLabels,
) -> Result<vyre_driver::OutputBuffers, BackendError> {
    let empty_inputs: [&[u8]; 0] = [];
    record_and_readback(RecordAndReadback {
        device_queue: &pipeline.device_queue,
        pool: arena.pool(),
        readback_rings: readback_rings.then(|| arena.readback_rings()),
        pipeline: &pipeline.pipeline,
        bind_group_layouts: &pipeline.bind_group_layouts,
        bind_group_cache: Some(pipeline.bind_group_cache.as_ref()),
        buffer_bindings: &pipeline.buffer_bindings,
        inputs: &empty_inputs,
        output_bindings: Arc::clone(&pipeline.output_bindings),
        trap_tags: &pipeline.trap_tags,
        workgroup_count: [1, 1, 1],
        indirect: pipeline.indirect.as_ref(),
        labels,
        iterations: 1,
        timestamp_profile: false,
        inferred_grid_shape: None,
    })
}

#[path = "bind_group_cache_contracts.rs"]
mod bind_group_cache_contracts;
#[path = "layout_config_contracts.rs"]
mod layout_config_contracts;
#[path = "prerecorded_contracts.rs"]
mod prerecorded_contracts;
#[path = "readback_ring_contracts.rs"]
mod readback_ring_contracts;
#[path = "trap_output_contracts.rs"]
mod trap_output_contracts;
