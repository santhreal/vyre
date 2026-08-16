//! Native pipeline-mode implementation for the wgpu backend.
//!
//! Compiled pipelines and bind-group layouts are reused so repeated
//! submissions pay only resource preparation, execution, and readback costs.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::hash::BuildHasherDefault;
#[cfg(test)]
pub(crate) use vyre_driver::enforce_actual_output_budget;
use vyre_driver::{resolve_launch_workgroup_for_geometry, LaunchGeometry};
use vyre_driver::tuner::Mode;
use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::BackendLayoutFingerprint;
pub(crate) use vyre_driver::{element_size_bytes, OutputBindingLayout};
use vyre_driver::{admit_dispatch_grid, find_indirect_dispatch, infer_dispatch_grid_for_count};
pub use vyre_driver::{output_layout_from_program, IndirectDispatch, OutputLayout};
use vyre_driver::{BackendError, DispatchConfig, OutputBuffers};
use vyre_foundation::execution_plan::{self, ExecutionPlan};
use vyre_foundation::ir::Program;
use vyre_foundation::validate::ValidationOptions;

use crate::buffer::{BindGroupCache, StagingBufferPool};
use crate::pipeline::disk_cache::{
    compiled_pipeline_cache_key, create_compiled_pipeline_cache, early_pipeline_cache_key,
    load_or_compile_disk_wgsl, persist_compiled_pipeline_cache,
};
use crate::runtime;
use crate::staging_reserve::reserve_backend_vec;
use vyre_driver::allocation::reserve_hash_set_to_capacity;
use vyre_emit_naga::program::TrapTag;
use vyre_lower::{TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS};

pub(crate) use self::descriptor_metadata::BufferBindingInfo;
use self::descriptor_metadata::{
    bind_group_layout_fingerprint, create_bind_group_layouts, descriptor_buffer_bindings,
    descriptor_trap_tags,
};

pub(crate) type BindGroupLayoutCache = dashmap::DashMap<
    BackendLayoutFingerprint,
    Arc<[Arc<wgpu::BindGroupLayout>]>,
    BuildHasherDefault<rustc_hash::FxHasher>,
>;

/// Target text that has already been emitted and authenticated for `program`.
///
/// Held as one value so a caller cannot supply text without the descriptor it
/// was emitted from: the descriptor decides the workgroup override the text
/// was built against.
pub(crate) struct AuthenticatedTarget<'a> {
    pub(crate) wgsl: &'a str,
    pub(crate) descriptor: &'a vyre_lower::KernelDescriptor,
}

/// GPU pipeline + **all** per-program dispatch metadata co-located for
/// cache hits. A hit on [`early_pipeline_cache_key`] or the WGSL hash
/// key must skip `execution_plan::plan`, output-layout derivation,
/// and fresh [`StagingBufferPool::new`] (subagent: pipeline.rs compile
/// path  -  2026-04 orchestration sweep).
#[derive(Debug)]
pub(crate) struct CachedPipelineArtifact {
    id: String,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layouts: Arc<[Arc<wgpu::BindGroupLayout>]>,
    bind_group_cache: Arc<BindGroupCache>,
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

fn wgpu_effective_dispatch_config(
    program: &Program,
    config: &DispatchConfig,
    device: &wgpu::Device,
    geometry: LaunchGeometry,
) -> Result<DispatchConfig, BackendError> {
    wgpu_effective_dispatch_config_for_limits(
        program,
        config,
        wgpu_launch_limits(device),
        Mode::from_env(),
        geometry,
    )
}

fn wgpu_effective_dispatch_config_for_limits(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    mode: Mode,
    geometry: LaunchGeometry,
) -> Result<DispatchConfig, BackendError> {
    let mut effective = config.clone();
    if geometry == LaunchGeometry::Untracked && effective.workgroup_override.is_some() {
        return Ok(effective);
    }
    let element_count = wgpu_launch_element_count_for_tuning(program)?;
    let selected = resolve_launch_workgroup_for_geometry(
        program,
        &effective,
        limits,
        element_count,
        mode,
        geometry,
    );
    if selected != program.workgroup_size() {
        effective.workgroup_override = Some(selected);
    } else {
        effective.workgroup_override = None;
    }
    Ok(effective)
}

fn wgpu_launch_element_count_for_tuning(program: &Program) -> Result<u32, BackendError> {
    if program.output_buffer_indices().is_empty() {
        return Ok(0);
    }
    let layouts = vyre_driver::output_binding_layouts(program)?;
    let word_count = layouts
        .first()
        .map(|layout| layout.word_count)
        .unwrap_or_default();
    u32::try_from(word_count).map_err(|error| {
        BackendError::new(format!(
            "wgpu natural-gradient launch tuning cannot represent {word_count} output word(s) as u32: {error}. Fix: split the dispatch or provide an explicit workgroup/grid override."
        ))
    })
}

pub(crate) fn wgpu_launch_limits(device: &wgpu::Device) -> LaunchGeometryLimits {
    let limits = device.limits();
    LaunchGeometryLimits {
        backend: "wgpu",
        max_threads_per_block: limits.max_compute_invocations_per_workgroup,
        max_block_dim: [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ],
        max_grid_dim: [limits.max_compute_workgroups_per_dimension; 3],
        // WebGPU exposes no per-compute-unit thread budget, so wgpu reports
        // none. Zero keeps residency-aware launch decisions inert here rather
        // than deriving one from a number this API never supplies.
        max_threads_per_sm: 0,
    }
}

/// In-memory cache keyed by canonical program and adapter identity.
///
/// Keyed by a full program fingerprint (serialized IR + adapter fingerprint),
/// returned as `Arc` so multiple callers share one ComputePipeline.
/// `WgpuPipeline` is a thin wrapper around an `Arc<CachedPipeline>` plus
/// per-instance values (id, output_size).

/// Materialized WGPU executable state.
///
/// Holds the authenticated compute pipeline, bind-group layout, and dispatch
/// geometry used by artifact instances and lower-level driver diagnostics.
#[derive(Clone)]
pub struct WgpuPipeline {
    pub(crate) id: String,
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bind_group_layouts: Arc<[Arc<wgpu::BindGroupLayout>]>,
    pub(crate) bind_group_cache: Arc<BindGroupCache>,
    pub(crate) buffer_bindings: Arc<[BufferBindingInfo]>,
    pub(crate) output_bindings: Arc<[OutputBindingLayout]>,
    pub(crate) execution_plan: Arc<ExecutionPlan>,
    pub(crate) device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
    pub(crate) output: OutputLayout,
    pub(crate) output_word_count: usize,
    pub(crate) workgroup_shape: [u32; 3],
    pub(crate) workgroup_size: u32,
    pub(crate) indirect: Option<IndirectDispatch>,
    pub(crate) trap_tags: Arc<[TrapTag]>,
    /// Shared persistent GPU-handle pool (H1). The legacy dispatch
    /// path acquires handles from here so repeated dispatches reuse
    /// `wgpu::Buffer` allocations instead of churning the GPU
    /// allocator on every call.
    pub(crate) persistent_pool: crate::buffer::BufferPool,
    /// Staging buffer pool for readback. Hot dispatch paths reuse
    /// MAP_READ staging buffers instead of creating a fresh
    /// `wgpu::Buffer` on every readback.
    pub(crate) staging_pool: StagingBufferPool,
}

impl std::fmt::Debug for WgpuPipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuPipeline")
            .field("id", &self.id)
            .field("buffer_bindings", &self.buffer_bindings)
            .field("output_bindings", &self.output_bindings)
            .field("execution_tracks", &self.execution_plan.tracks)
            .field("output", &self.output)
            .field("output_word_count", &self.output_word_count)
            .field("workgroup_shape", &self.workgroup_shape)
            .field("workgroup_size", &self.workgroup_size)
            .field("indirect", &self.indirect)
            .field("trap_tags", &self.trap_tags)
            .finish_non_exhaustive()
    }
}

impl WgpuPipeline {
    fn from_cached_artifact(
        cached: &CachedPipelineArtifact,
        device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
        persistent_pool: crate::buffer::BufferPool,
    ) -> Self {
        Self {
            id: cached.id.clone(),
            pipeline: cached.pipeline.clone(),
            bind_group_layouts: cached.bind_group_layouts.clone(),
            bind_group_cache: cached.bind_group_cache.clone(),
            buffer_bindings: cached.buffer_bindings.clone(),
            output_bindings: cached.output_bindings.clone(),
            execution_plan: cached.execution_plan.clone(),
            device_queue,
            output: cached.output,
            output_word_count: cached.output_word_count,
            workgroup_shape: cached.workgroup_shape,
            workgroup_size: cached.workgroup_size,
            indirect: cached.indirect.clone(),
            trap_tags: cached.trap_tags.clone(),
            persistent_pool,
            staging_pool: cached.staging_pool.clone(),
        }
    }

    /// Compile `program` using backend-owned device resources.
    ///
    /// `target` supplies already authenticated target text and its descriptor;
    /// `None` lowers and emits from `program`. The two used to be independent
    /// `Option` arguments reachable through two wrapper entry points, so a
    /// half-set pair was representable and each wrapper re-listed the wiring.
    ///
    /// # Errors
    ///
    /// Returns a backend error when lowering, cache access, or pipeline
    /// creation fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_with_device_queue(
        program: &Program,
        config: &DispatchConfig,
        adapter_info: wgpu::AdapterInfo,
        enabled_features: crate::runtime::device::EnabledFeatures,
        device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
        persistent_pool: crate::buffer::BufferPool,
        pipeline_cache: Arc<runtime::cache::pipeline::LruPipelineCache>,
        bind_group_layout_cache: Arc<BindGroupLayoutCache>,
        target: Option<AuthenticatedTarget<'_>>,
    ) -> Result<Arc<Self>, BackendError> {
        let authenticated_wgsl = target.as_ref().map(|target| target.wgsl);
        let authenticated_descriptor = target.as_ref().map(|target| target.descriptor);
        let compile_program = program;
        let geometry = match authenticated_descriptor {
            Some(descriptor) => {
                LaunchGeometry::from_recorded(descriptor.dispatch.workgroup_size, "wgpu")?
            }
            None => LaunchGeometry::Untracked,
        };
        let effective_config =
            wgpu_effective_dispatch_config(compile_program, config, &device_queue.0, geometry)?;
        let config = &effective_config;
        // Authenticated target bytes cannot reuse a Program-keyed cache entry
        // that may have been built from different module bytes.
        let early_key = early_pipeline_cache_key(compile_program, &adapter_info, config);
        if authenticated_wgsl.is_none() {
            if let Some(hit) = pipeline_cache.get(&early_key) {
                return Ok(Arc::new(Self::from_cached_artifact(
                    hit.as_ref(),
                    device_queue,
                    persistent_pool,
                )));
            }
        }

        let wgsl = match authenticated_wgsl {
            Some(wgsl) => wgsl.to_string(),
            None => load_or_compile_disk_wgsl(
                compile_program,
                &adapter_info,
                config,
                &enabled_features,
            )?,
        };
        let artifact_key = compiled_pipeline_cache_key(&adapter_info, &wgsl);

        let descriptor = match authenticated_descriptor {
            Some(descriptor) => descriptor.clone(),
            None => crate::emit::descriptor_gate::validate_and_analyze(compile_program).map_err(
                |error| {
                    BackendError::new(format!(
                        "failed to derive KernelDescriptor for wgpu pipeline metadata: {error}. Fix: keep pipeline metadata on the same descriptor path as WGSL emission."
                    ))
                },
            )?,
        };
        let staging_pool = StagingBufferPool::new();
        let trap_tags_vec = descriptor_trap_tags(&descriptor)?;
        if !trap_tags_vec.is_empty()
            && !descriptor
                .bindings
                .slots
                .iter()
                .any(|slot| slot.name == TRAP_SIDECAR_NAME)
        {
            return Err(BackendError::new(format!(
                "descriptor contains trap tags but no `{TRAP_SIDECAR_NAME}` binding. Fix: lower traps through vyre-lower so the sidecar binding is inserted."
            )));
        }
        let trap_tags: Arc<[TrapTag]> = trap_tags_vec.into();
        let validation_options = ValidationOptions::default().with_backend_capabilities(
            crate::runtime::adapter_caps_probe::from_backend_profile(
                &adapter_info,
                &device_queue.0.limits(),
                &enabled_features,
            )
            .validation_capabilities(),
        );
        let execution_plan = Arc::new(
            execution_plan::plan_with_options(compile_program, validation_options).map_err(
                |error| BackendError::InvalidProgram {
                    fix: format!("Fix: wgpu pipeline planning rejected the Program: {error}"),
                },
            )?,
        );
        let output_bindings: Arc<[OutputBindingLayout]> =
            if program.output_buffer_indices().is_empty() && !trap_tags.is_empty() {
                Arc::from([])
            } else {
                vyre_driver::output_binding_layouts(program)?.into()
            };
        let output = output_bindings.first().map_or(
            OutputLayout {
                full_size: 0,
                read_size: 0,
                copy_offset: 0,
                copy_size: 0,
                trim_start: 0,
            },
            |primary_output| primary_output.layout,
        );
        let output_word_count = output_bindings
            .iter()
            .map(|binding| binding.word_count)
            .max()
            .unwrap_or(0);
        // Preserve the original workgroup shape. Without program-level
        // logical extents, dispatch paths can only derive a safe default grid
        // for 1D kernels; 2D/3D kernels must provide `grid_override`.
        let effective_wg = config
            .workgroup_override
            .unwrap_or(compile_program.workgroup_size);
        let workgroup_shape = [
            effective_wg[0].max(1),
            effective_wg[1].max(1),
            effective_wg[2].max(1),
        ];
        let workgroup_size = workgroup_shape[0]
            .checked_mul(workgroup_shape[1])
            .and_then(|xy| xy.checked_mul(workgroup_shape[2]))
            .ok_or_else(|| {
                BackendError::new(format!(
                    "workgroup_size {:?} overflows u32 when flattened. Fix: lower to a valid WGPU workgroup shape instead of saturating launch metadata.",
                    workgroup_shape
                ))
            })?;
        let indirect = find_indirect_dispatch(compile_program)?;
        let mut public_output_bindings = FxHashSet::default();
        reserve_hash_set_to_capacity(
            &mut public_output_bindings,
            output_bindings.len(),
            "WGPU pipeline binding classification",
            "public output binding",
            "split the pipeline or reduce output binding fanout before compilation",
        )?;
        public_output_bindings.extend(output_bindings.iter().map(|output| output.binding));
        let buffers = program.buffers();
        let mut explicit_output_bindings = FxHashSet::default();
        reserve_hash_set_to_capacity(
            &mut explicit_output_bindings,
            buffers.len(),
            "WGPU pipeline binding classification",
            "explicit output binding",
            "split the pipeline or reduce output binding fanout before compilation",
        )?;
        let mut pipeline_live_out_bindings = FxHashSet::default();
        reserve_hash_set_to_capacity(
            &mut pipeline_live_out_bindings,
            buffers.len(),
            "WGPU pipeline binding classification",
            "pipeline live-out binding",
            "split the pipeline or reduce live-out binding fanout before compilation",
        )?;
        for buffer in buffers {
            if buffer.is_output() {
                explicit_output_bindings.insert(buffer.binding());
            }
            if buffer.is_pipeline_live_out() {
                pipeline_live_out_bindings.insert(buffer.binding());
            }
        }

        let buffer_bindings: Arc<[BufferBindingInfo]> = descriptor_buffer_bindings(
            &descriptor,
            &public_output_bindings,
            &explicit_output_bindings,
            &pipeline_live_out_bindings,
        )?
        .into();

        for (group, binding) in bindings_reflection::declared_bindings(&wgsl) {
            if !buffer_bindings
                .iter()
                .any(|info| info.group == group && info.binding == binding)
            {
                return Err(BackendError::new(format!(
                    "lowered WGSL declares @group({group}) @binding({binding}) but pipeline metadata has no matching KernelDescriptor binding. Fix: keep Naga emission and pipeline binding derivation on the same KernelDescriptor."
                )));
            }
        }

        let max_group = buffer_bindings.iter().map(|b| b.group).max().unwrap_or(0);

        // Compile outside any lock so other threads can read the cache.
        let (device, _queue) = &*device_queue;

        let layout_fingerprint = bind_group_layout_fingerprint(&buffer_bindings)?;
        let bind_group_layouts = match bind_group_layout_cache.entry(layout_fingerprint) {
            dashmap::mapref::entry::Entry::Occupied(hit) => Arc::clone(hit.get()),
            dashmap::mapref::entry::Entry::Vacant(slot) => Arc::clone(&slot.insert(
                create_bind_group_layouts(device, &buffer_bindings, max_group)?,
            )),
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vyre P-6 pipeline layout"),
            bind_group_layouts: &bind_group_layouts
                .iter()
                .map(|l| l.as_ref())
                .collect::<SmallVec<[_; 8]>>(),
            push_constant_ranges: &[],
        });

        // Only attempt the persistent pipeline cache when the device actually
        // enabled PIPELINE_CACHE. `enabled_features_for_adapter` (device.rs)
        // requests it only on backends that implement wgpu pipeline caches
        // (Vulkan/DX12); on Metal/GL this is `false` and we compile uncached. A
        // `create_pipeline_cache` call on a device without the feature is a
        // fatal validation abort, not a recoverable error.
        let pipeline_cache_handle = if device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            Some(create_compiled_pipeline_cache(device, &artifact_key)?)
        } else {
            None
        };
        runtime::shader::dump_wgsl_if_requested("vyre P-6 cached shader module", &wgsl).map_err(
            |error| {
                BackendError::new(format!(
                    "failed to dump WGSL for compiled pipeline: {error}. Fix: set VYRE_DUMP_WGSL to a writable directory or unset it"
                ))
            },
        )?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vyre P-6 cached shader module"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vyre P-6 cached pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: pipeline_cache_handle.as_ref().map(|h| &h.cache),
        });
        if let Some(error) =
            crate::runtime::device::pop_error_scope_now(device).map_err(|message| {
                BackendError::KernelCompileFailed {
                    backend: "wgpu".to_owned(),
                    compiler_message: format!(
                        "cached WGSL pipeline validation did not complete without a host wait: {message}"
                    ),
                }
            })?
        {
            return Err(BackendError::KernelCompileFailed {
                backend: "wgpu".to_owned(),
                compiler_message: format!(
                    "cached WGSL pipeline validation failed: {error}. Fix: validate the lowered WGSL, bind-group layout, and adapter limits before compiling."
                ),
            });
        }
        if let Some(handle) = &pipeline_cache_handle {
            persist_compiled_pipeline_cache(&artifact_key, &handle.cache)?;
        }

        let compiled_artifact = Arc::new(CachedPipelineArtifact {
            id: format!("wgpu:{}", vyre_driver::hex_short(&artifact_key.hash)),
            pipeline: Arc::new(pipeline),
            bind_group_layouts,
            bind_group_cache: Arc::new(BindGroupCache::default()),
            execution_plan: execution_plan.clone(),
            output_bindings: output_bindings.clone(),
            buffer_bindings: buffer_bindings.clone(),
            output,
            output_word_count,
            workgroup_shape,
            workgroup_size,
            indirect: indirect.clone(),
            trap_tags: trap_tags.clone(),
            staging_pool: staging_pool.clone(),
        });

        if authenticated_wgsl.is_none() {
            pipeline_cache.insert(early_key, Arc::clone(&compiled_artifact));
        }

        Ok(Arc::new(Self::from_cached_artifact(
            compiled_artifact.as_ref(),
            device_queue,
            persistent_pool,
        )))
    }

    pub(crate) fn output_binding(
        &self,
        binding: u32,
    ) -> Result<&OutputBindingLayout, BackendError> {
        self.output_bindings
            .iter()
            .find(|output| output.binding == binding)
            .ok_or_else(|| {
                BackendError::new(format!(
                    "missing output layout metadata for binding {binding}. Fix: keep output_bindings synchronized with writable BufferDecls during pipeline compilation."
                ))
            })
    }

    /// The launch grid for this dispatch, judged against the device's ceiling.
    ///
    /// Every wgpu dispatch path resolves its grid here, which is why the per-axis
    /// admission belongs here too: a grid past the device ceiling is rejected by
    /// the API from inside a recorded command buffer, where the rejection is a
    /// validation abort rather than an error this call can return.
    pub(crate) fn workgroups_for_dispatch(
        &self,
        config: &DispatchConfig,
    ) -> Result<[u32; 3], BackendError> {
        let grid = self.requested_workgroups(config)?;
        admit_dispatch_grid(grid, self.max_workgroups_per_axis(), crate::WGPU_BACKEND_ID)
    }

    /// The per-axis workgroup ceiling this device reported.
    pub(crate) fn max_workgroups_per_axis(&self) -> u32 {
        self.device_queue
            .0
            .limits()
            .max_compute_workgroups_per_dimension
    }

    fn requested_workgroups(&self, config: &DispatchConfig) -> Result<[u32; 3], BackendError> {
        if let Some(grid) = config.grid_override {
            return Ok(grid);
        }
        // Non-1D workgroups have no unambiguous default grid: there's
        // no single right way to map an unknown element_count across
        // an N×M (or N×M×K) thread tile. Force the caller to set
        // grid_override explicitly rather than silently producing a
        // wrong dispatch.
        if self.workgroup_shape[1] != 1 || self.workgroup_shape[2] != 1 {
            return Err(BackendError::new(format!(
                "Fix: dispatch with non-1D workgroup_size {:?} requires DispatchConfig::grid_override. \
                 Set grid_override to the logical [x, y, z] dispatch shape you want.",
                self.workgroup_shape,
            )));
        }
        let output_word_count = u32::try_from(self.output_word_count).map_err(|error| {
            BackendError::new(format!(
                "compiled WGPU pipeline output word count {} does not fit u32: {error}. Fix: shard the dispatch before grid inference instead of saturating the launch size.",
                self.output_word_count
            ))
        })?;
        infer_dispatch_grid_for_count(output_word_count, self.workgroup_shape)
    }

    /// Substrate-neutral performance and accuracy plan computed for this
    /// compiled program.
    #[must_use]
    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.execution_plan
    }
}

impl WgpuPipeline {
    /// Read every persistent output handle back into caller slots, trimmed to
    /// each output's declared byte range.
    ///
    /// The single readback pass for the persistent pool. Both the resident and
    /// the legacy borrowed dispatch paths land here.
    fn readback_persistent_outputs(
        &self,
        output_handles: &[crate::buffer::GpuBufferHandle],
        deadline: Option<Instant>,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        let (device, queue) = &*self.device_queue;
        self::output_slots::resize_vec_with(
            outputs,
            output_handles.len(),
            Vec::new,
            "persistent pipeline output slots",
        )?;
        for ((handle, output), bytes) in output_handles
            .iter()
            .zip(self.output_bindings.iter())
            .zip(outputs.iter_mut())
        {
            crate::pipeline::output_readback::read_trimmed_output(
                handle,
                output,
                device,
                &self.staging_pool,
                queue,
                "persistent pipeline output",
                deadline,
                bytes,
            )?;
        }
        Ok(())
    }

    fn raise_if_trapped(
        &self,
        input_handles: &[crate::buffer::GpuBufferHandle],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        deadline: Option<Instant>,
    ) -> Result<(), BackendError> {
        let Some((input_index, _)) = self
            .buffer_bindings
            .iter()
            .filter(|info| info.kind != vyre_foundation::ir::MemoryKind::Shared && !info.is_output)
            .enumerate()
            .find(|(_, info)| info.internal_trap)
        else {
            return Ok(());
        };
        let Some(handle) = input_handles.get(input_index) else {
            return Err(BackendError::new(
                "internal wgpu trap buffer was not allocated. Fix: keep trap buffer binding metadata synchronized with legacy input handle allocation.",
            ));
        };
        let trap_sidecar_bytes = usize::try_from(TRAP_SIDECAR_WORDS)
            .map_err(|source| {
                BackendError::new(format!(
                    "trap sidecar word count cannot fit usize: {source}. Fix: keep TRAP_SIDECAR_WORDS within the host index ABI."
                ))
            })?
            .checked_mul(4)
            .ok_or_else(|| {
                BackendError::new(
                    "trap sidecar byte length overflowed usize. Fix: keep TRAP_SIDECAR_WORDS within the host index ABI.",
                )
            })?;
        let mut bytes = Vec::new();
        reserve_backend_vec(&mut bytes, trap_sidecar_bytes, "trap sidecar readback")?;
        handle.readback_prefix_until(
            device,
            Some(&self.staging_pool),
            queue,
            4,
            &mut bytes,
            deadline,
        )?;
        if bytes.len() < 4 {
            return Err(BackendError::new(format!(
                "internal wgpu trap flag readback returned {} bytes but 4 bytes are required. Fix: allocate the trap sidecar as {TRAP_SIDECAR_WORDS} u32 words.",
                bytes.len()
            )));
        }
        let flag = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if flag == 0 {
            return Ok(());
        }

        handle.readback_prefix_until(
            device,
            Some(&self.staging_pool),
            queue,
            u64::from(TRAP_SIDECAR_WORDS) * 4,
            &mut bytes,
            deadline,
        )?;
        trap_error_from_sidecar(&bytes, &self.trap_tags).map_or(Ok(()), Err)
    }

    fn enforce_static_output_budget(&self, config: &DispatchConfig) -> Result<(), BackendError> {
        let Some(limit) = config.max_output_bytes else {
            return Ok(());
        };
        let visible = self.execution_plan.strategy.readback.visible_bytes();
        let visible = usize::try_from(visible).map_err(|source| {
            BackendError::new(format!(
                "visible readback size cannot fit usize: {source}. Fix: split the Program output before dispatch."
            ))
        })?;
        if visible > limit {
            return Err(BackendError::new(format!(
                "visible readback size {visible} exceeds DispatchConfig.max_output_bytes {limit}. Fix: narrow BufferDecl::output_byte_range or raise max_output_bytes."
            )));
        }
        Ok(())
    }
}

/// Decode a trap sidecar readback into this backend's refusal.
///
/// The record layout and the short-readback refusal live in
/// [`vyre_driver::trap_record`]; only the wording and the tag lookup are this
/// backend's. A short readback is a refusal, not a "no trap", so the error is
/// returned rather than mapped to `None`.
pub(crate) fn trap_error_from_sidecar(bytes: &[u8], trap_tags: &[TrapTag]) -> Option<BackendError> {
    match vyre_driver::trap_record::decode_trap_record(bytes) {
        Err(error) => Some(error),
        Ok(None) => None,
        Ok(Some(record)) => {
            let detail = record.describe(|code| {
                trap_tags
                    .iter()
                    .find(|tag| tag.code == code)
                    .map(|tag| tag.tag.as_ref().to_owned())
            });
            Some(BackendError::new(format!(
                "wgpu dispatch trapped: {detail}"
            )))
        }
    }
}

/// Buffer-binding validation, usage flags, and output-clear helpers
/// shared by every wgpu pipeline mode (single-shot, persistent,
/// compound). Hosts the `usage_for_binding`, `validate_handle`, and
/// `clear_outputs_for_bound` helpers all dispatch paths consume.
pub(crate) mod binding;
/// WGSL bind-group reflection scanner  -  extracts every
/// `(group, binding)` pair declared by lowered shader source so the
/// reusable pipeline wrapper can mirror the layout exactly when
/// creating bind groups. Misalignment is a validation error.
pub(crate) mod bindings_reflection;
/// Which cached pipeline entries a rule-graph change reaches. Shared by the
/// in-memory and on-disk invalidation paths so both act on one mask.
pub(crate) mod cache_impact;
/// `CompiledPipeline` trait dispatch entrypoints. Split out so the parent
/// pipeline module does not own both compilation and execution mechanics.
pub(crate) mod compiled_dispatch;
/// Compound-resource binding (multi-program shape with shared GPU
/// resources). Used by `engine::graph` to compose pipelines without
/// re-allocating bind groups between dispatches.
pub(crate) mod compound;
/// KernelDescriptor-to-WGPU binding metadata and bind-group layout derivation.
/// Keeping this out of the parent pipeline module preserves the rule that
/// pipeline files orchestrate compile/dispatch flow rather than owning every
/// metadata transformation.
pub(crate) mod descriptor_metadata;
/// On-disk WGSL + compiled-pipeline cache. Front-end calls
/// `load_or_compile_disk_wgsl` / `compiled_pipeline_cache_key` /
/// `persist_compiled_pipeline_cache` to skip Naga + Tint + driver
/// linkage for unchanged programs across `./cargo_full test` cycles.
pub(crate) mod disk_cache;
/// Sibling of `disk_cache`  -  on-disk entry paths, entry metadata, and
/// entry removal for the disk cache.
pub(crate) mod disk_cache_entries;
/// Trimmed output readback. Owns the contract that `output_byte_range`
/// transfers only meaningful bytes instead of whole output allocations.
pub(crate) mod output_readback;
/// Fallible output slot resizing shared by persistent and batched paths.
pub(crate) mod output_slots;
/// Persistent `Resource` to GPU-handle resolution and trap sidecar allocation.
pub(crate) mod persistent_resources;
/// Persistent dispatch-item lifecycle (`DispatchItem`)  -  multi-call
/// reuse of bind groups, staging pools, and pipeline handles across
/// the same program-graph topology.
pub mod persistent;

// Inline: covers the private `wgpu_effective_dispatch_config_for_limits` and the
// `pub(crate)` `BindGroupLayoutCache` and `WgpuPipeline::compile_with_device_queue`,
// none of which an integration test can reach.
#[cfg(test)]
mod tests {
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
    use vyre_driver::BackendError;
    use vyre_driver::DEFAULT_PIPELINE_CACHE_ENTRIES;

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

    /// A one-node program storing `value` at index 0 of a `count`-element `u32`
    /// output buffer named `name`.
    ///
    /// The minimum program that produces an observable output. Six contract tests
    /// spelled it out, so a change to the fixture shape had to be applied six
    /// times or the tests stopped exercising the same program.
    fn stores_u32(name: &str, count: u32, value: u32) -> Program {
        Program::wrapped(
            vec![BufferDecl::output(name, 0, DataType::U32).with_count(count)],
            [1, 1, 1],
            vec![Node::store(name, Expr::u32(0), Expr::u32(value))],
        )
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

    mod bind_group_cache_contracts {
        use super::*;

        /// PERF-HOT-01: two WgpuPipeline instances for the same compiled shader
        /// must share one BindGroupCache (Arc identity). Different compiled
        /// shaders must have independent caches.
        #[test]
        fn bind_group_cache_shared_per_compiled_shader() {
            let harness = PipelineHarness::new("cache-sharing test");
            let pool = BufferPool::new(
                harness.device_queue.0.clone(),
                harness.device_queue.1.clone(),
                &harness.config,
            );
            let layout_cache = Arc::clone(&harness.layout_cache);

            let program1 = stores_u32("out", 4, 7);

            let p1 = harness
                .compile(&program1, pool.clone())
                .expect("Fix: first compile must succeed; restore this invariant before continuing.");
            assert_eq!(
                layout_cache.len(),
                1,
                "Fix: first compile should insert one shared bind-group layout fingerprint"
            );

            let p2 = harness
                .compile(&program1, pool.clone())
                .expect("Fix: second compile of same program must succeed; restore this invariant before continuing.");
            assert_eq!(
                layout_cache.len(),
                1,
                "Fix: recompiling the same layout must hit the shared layout cache"
            );

            assert!(
                Arc::ptr_eq(&p1.bind_group_cache, &p2.bind_group_cache),
                "Fix: same compiled shader must share BindGroupCache (HOT-01)"
            );

            let (input_handles, mut output_handles) = p1.legacy_handles_from_inputs(&[]).expect(
                "Fix: legacy handle creation must succeed; restore this invariant before continuing.",
            );
            p1.dispatch_persistent(&input_handles, &mut output_handles, None, [1, 1, 1])
                .expect("Fix: first dispatch must succeed; restore this invariant before continuing.");
            let stats_after_miss = p1.bind_group_cache_stats();
            assert_eq!(
                stats_after_miss.misses, 1,
                "Fix: first dispatch of a new signature must be a cache miss"
            );
            assert_eq!(stats_after_miss.hits, 0);

            p1.dispatch_persistent(&input_handles, &mut output_handles, None, [1, 1, 1])
                .expect("Fix: second dispatch must succeed; restore this invariant before continuing.");
            let stats_after_hit = p1.bind_group_cache_stats();
            assert_eq!(
                stats_after_hit.hits, 1,
                "Fix: second dispatch with identical handles must be a cache hit"
            );
            assert_eq!(stats_after_hit.misses, 1);

            let program2 = stores_u32("out2", 8, 42);

            let p3 = harness.compile(&program2, pool).expect(
                "Fix: compile of different program must succeed; restore this invariant before continuing.",
            );
            assert_eq!(
                layout_cache.len(),
                1,
                "Fix: compatible output-only programs must share the same wgpu bind-group layout cache entry"
            );

            assert!(
                !Arc::ptr_eq(&p1.bind_group_cache, &p3.bind_group_cache),
                "Fix: different compiled shaders must have independent BindGroupCaches"
            );
        }

        #[test]
        fn compiled_borrowed_timed_dispatch_reports_device_ns() {
            use vyre_driver::CompiledPipeline;

            let harness = PipelineHarness::new("compiled timing test");
            let device = &harness.device_queue.0;
            assert!(
                device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                    && device
                        .features()
                        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
                "Fix: WGPU compiled timing test requires timestamp query features to be negotiated."
            );
            let arena = harness.arena();

            let program = stores_u32("out", 1, 7);
            let pipeline = harness
                .compile_on_arena(&program, &arena)
                .expect("Fix: compiled timed dispatch test pipeline must compile.");

            let timed = pipeline
                .dispatch_borrowed_timed(&[], &harness.config)
                .expect("Fix: compiled borrowed timed dispatch must succeed.");
            assert_eq!(
                u32::from_le_bytes(timed.outputs[0][0..4].try_into().unwrap()),
                7
            );
            assert!(
                timed.device_ns.is_some_and(|ns| ns > 0),
                "Fix: WGPU compiled borrowed timed dispatch must report GPU device nanoseconds."
            );
            assert!(timed.enqueue_ns.is_some_and(|ns| ns > 0));
            assert!(timed.wait_ns.is_some_and(|ns| ns > 0));
        }
    }

    mod layout_config_contracts {
        use super::*;
        use vyre_driver::LaunchGeometry;

        #[test]
        fn hex_short_truncates_to_eight_bytes() {
            let hash = *blake3::hash(b"vyre-pipeline").as_bytes();
            let expected = vyre_driver::hex_encode(&hash[..8]);
            assert_eq!(vyre_driver::hex_short(&hash).len(), 16);
            assert_eq!(vyre_driver::hex_short(&hash), expected);
        }

        #[test]
        fn actual_output_budget_rejects_combined_outputs() {
            let mut config = DispatchConfig::default();
            config.max_output_bytes = Some(3);
            let err = enforce_actual_output_budget(&config, &[vec![0; 2], vec![0; 2]])
                .expect_err("combined readback over budget must fail");
            assert!(
                err.to_string().contains("max_output_bytes"),
                "Fix: budget rejection must name the violated policy, got {err}"
            );
        }

        #[test]
        fn output_layout_matches_trimmed_execution_plan() {
            let program = Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32)
                    .with_count(1024)
                    .with_output_byte_range(4..12)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            );
            let plan = execution_plan::plan(&program)
                .expect("Fix: trimmed output program must plan; restore this invariant before continuing.");
            assert_eq!(
                plan.strategy.readback,
                ReadbackStrategy::Trimmed {
                    visible_bytes: 8,
                    avoided_bytes: 4088,
                }
            );
            let layouts = vyre_driver::output_binding_layouts(&program)
                .expect("Fix: layout must derive; restore this invariant before continuing.");
            assert_eq!(layouts[0].layout.read_size, 8);
            assert_eq!(layouts[0].layout.copy_size, 8);
        }

        #[test]
        fn wgpu_compile_config_receives_natural_gradient_workgroup_before_lowering() {
            let program = Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
                [32, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            );
            let limits = LaunchGeometryLimits {
                backend: "wgpu-test",
                max_threads_per_block: 1024,
                max_block_dim: [1024, 1024, 64],
                max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
                max_threads_per_sm: 0,
            };

            let effective = super::wgpu_effective_dispatch_config_for_limits(
                &program,
                &DispatchConfig::default(),
                limits,
                Mode::NaturalGradient,
                LaunchGeometry::Untracked,
            )
            .expect("Fix: WGPU natural-gradient config derivation must be pure and valid");

            assert_eq!(
                effective.workgroup_override,
                Some([1024, 1, 1]),
                "Fix: WGPU lowering config must include the natural-gradient workgroup so WGSL @workgroup_size and dispatch metadata agree. WebGPU reports no per-compute-unit thread budget (max_threads_per_sm 0), so residency-aware cold start is inert here and this width is unchanged by it."
            );
        }

        #[test]
        fn wgpu_natural_gradient_compile_config_preserves_semantic_safety_gates() {
            let program = Program::wrapped(
                vec![
                    BufferDecl::output("out", 0, DataType::U32).with_count(4096),
                    BufferDecl::workgroup("scratch", 64, DataType::U32).with_kind(MemoryKind::Shared),
                ],
                [64, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            );
            let limits = LaunchGeometryLimits {
                backend: "wgpu-test",
                max_threads_per_block: 1024,
                max_block_dim: [1024, 1024, 64],
                max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
                max_threads_per_sm: 0,
            };
            let mut explicit = DispatchConfig::default();
            explicit.workgroup_override = Some([256, 1, 1]);

            let explicit_effective = super::wgpu_effective_dispatch_config_for_limits(
                &program,
                &explicit,
                limits,
                Mode::NaturalGradient,
                LaunchGeometry::Untracked,
            )
            .expect("Fix: explicit WGPU workgroup override must stay valid");
            assert_eq!(explicit_effective.workgroup_override, Some([256, 1, 1]));

            let shared_effective = super::wgpu_effective_dispatch_config_for_limits(
                &program,
                &DispatchConfig::default(),
                limits,
                Mode::NaturalGradient,
                LaunchGeometry::Untracked,
            )
            .expect("Fix: shared-memory WGPU config should remain valid without autotuning");
            assert_eq!(
                shared_effective.workgroup_override, None,
                "Fix: workgroup-local scratch kernels must keep the Program-declared WGPU workgroup."
            );
        }

        /// WHY: 150.15. The compiler searches the workgroup dimension and records the
        /// winning geometry in the artifact, and the authenticated module declares that
        /// shape. A launch tuner that picked another width would dispatch a kernel nobody
        /// compiled. Before this, the wgpu path pinned the descriptor width through a
        /// dispatch override, so any caller override applied first won instead.
        #[test]
        fn recorded_artifact_geometry_outranks_the_launch_tuner_and_caller_overrides() {
            let program = Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
                [32, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            );
            let limits = LaunchGeometryLimits {
                backend: "wgpu-test",
                max_threads_per_block: 1024,
                max_block_dim: [1024, 1024, 64],
                max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
                max_threads_per_sm: 0,
            };
            let mut caller_pinned = DispatchConfig::default();
            caller_pinned.workgroup_override = Some([256, 1, 1]);

            for config in [DispatchConfig::default(), caller_pinned] {
                let effective = super::wgpu_effective_dispatch_config_for_limits(
                    &program,
                    &config,
                    limits,
                    Mode::NaturalGradient,
                    LaunchGeometry::Compiled([64, 1, 1]),
                )
                .expect("Fix: a recorded compiled geometry must resolve without error");
                assert_eq!(
                    effective.workgroup_override,
                    Some([64, 1, 1]),
                    "Fix: the recorded artifact geometry must win over both the launch tuner and a caller override."
                );
            }
        }

        /// WHY: 150.15 boundary. A descriptor that records no geometry is an invalid
        /// artifact, not an invitation to choose one. A silent fall back to the declared
        /// or tuned width would launch a shape the artifact never authenticated.
        #[test]
        fn a_descriptor_without_recorded_geometry_is_an_error() {
            for absent in [[0, 1, 1], [64, 0, 1], [64, 1, 0], [0, 0, 0]] {
                let error = LaunchGeometry::from_recorded(absent, "wgpu")
                    .expect_err("Fix: an absent geometry record must fail the launch");
                assert!(
                    error.message().contains("records no workgroup geometry"),
                    "Fix: the error must name the missing record, got {error}"
                );
            }
            assert_eq!(
                LaunchGeometry::from_recorded([64, 1, 1], "wgpu").expect("a full record is valid"),
                LaunchGeometry::Compiled([64, 1, 1])
            );
        }
    }

    mod prerecorded_contracts {
        use super::*;

        /// Pre-recording a persistent dispatch builds bind groups and records the
        /// compute pass through the same code the direct persistent path uses, only
        /// under its own wgpu labels. Replaying the recorded command buffer must
        /// therefore land the same bytes in the output buffer that a direct dispatch
        /// of the same program lands.
        #[test]
        fn prerecorded_replay_writes_the_same_output_as_direct_dispatch() {
            let harness = PipelineHarness::new("pre-recorded dispatch replay test");
            let arena = harness.arena();

            let program = stores_u32("out", 4, 7);

            let pipeline = harness
                .compile_on_arena(&program, &arena)
                .expect("Fix: pre-recorded dispatch test pipeline must compile.");

            let direct = record_once(
                &pipeline,
                &arena,
                false,
                DispatchLabels {
                    bind_group: "vyre prerecord parity direct bind group",
                    encoder: "vyre prerecord parity direct",
                    compute: "vyre prerecord parity direct compute",
                },
            )
            .expect("Fix: direct persistent dispatch must succeed before comparing against replay.");

            let prerecorded = pipeline
                .prerecord_borrowed_dispatch(&[], [1, 1, 1])
                .expect("Fix: pre-recording a persistent dispatch must succeed.");
            prerecorded
                .replay(&harness.device_queue.1)
                .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
            let replayed = prerecorded
                .read_output(0)
                .expect("Fix: reading a replayed output buffer must succeed.");

            assert_eq!(
                u32::from_le_bytes(replayed[0..4].try_into().unwrap()),
                7,
                "Fix: replayed pre-recorded dispatch must write the program's stored value."
            );
            assert_eq!(
                replayed[0..16],
                direct[0][0..16],
                "Fix: pre-recorded replay and direct persistent dispatch must produce identical output bytes."
            );
        }

        /// A wgpu command buffer is single-submit. The second replay must be a
        /// structured error rather than a raw wgpu panic.
        #[test]
        fn prerecorded_second_replay_is_a_structured_error() {
            let harness = PipelineHarness::new("pre-recorded dispatch resubmit test");
            let arena = harness.arena();

            let program = stores_u32("out", 1, 3);

            let pipeline = harness
                .compile_on_arena(&program, &arena)
                .expect("Fix: pre-recorded resubmit test pipeline must compile.");
            let prerecorded = pipeline
                .prerecord_borrowed_dispatch(&[], [1, 1, 1])
                .expect("Fix: pre-recording a persistent dispatch must succeed.");

            prerecorded
                .replay(&harness.device_queue.1)
                .expect("Fix: first replay of a pre-recorded command buffer must succeed.");
            let error = prerecorded
                .replay(&harness.device_queue.1)
                .expect_err("Fix: a pre-recorded command buffer must refuse a second submission.");
            assert!(
                error.to_string().contains("already submitted"),
                "Fix: expected the single-submit diagnostic, got: {error}"
            );
        }
    }

    mod readback_ring_contracts {
        use super::*;

        #[test]
        fn direct_record_and_readback_reuses_bind_groups() {
            let harness = PipelineHarness::new("direct cache test");
            let arena = harness.arena();

            let program = stores_u32("out", 4, 7);

            let pipeline = harness
                .compile_on_arena(&program, &arena)
                .expect("Fix: compile must succeed; restore this invariant before continuing.");

            for _ in 0..2 {
                let outputs = record_once(
                    &pipeline,
                    &arena,
                    false,
                    DispatchLabels {
                        bind_group: "vyre direct cache test bind group",
                        encoder: "vyre direct cache test",
                        compute: "vyre direct cache test compute",
                    },
                )
                .expect(
                    "Fix: direct record/readback must succeed; restore this invariant before continuing.",
                );
                assert_eq!(u32::from_le_bytes(outputs[0][0..4].try_into().unwrap()), 7);
            }

            let stats = pipeline.bind_group_cache_stats();
            // The pool may or may not return the same buffer Arc across two
            // back-to-back readbacks (the prior readback's pinning, plus
            // size-class bucketing, decides). What we DO require: the cache
            // is exercised on every dispatch (misses + hits >= 2) and never
            // reports a negative-cost path (no double-build for a given Arc).
            let total = stats.misses + stats.hits;
            assert!(
                total >= 2,
                "two dispatches should each consult the bind-group cache (got misses={}, hits={})",
                stats.misses,
                stats.hits,
            );
            assert!(
                stats.misses <= 2,
                "no more than one bind-group build per distinct buffer identity (got misses={})",
                stats.misses,
            );
        }

        #[test]
        fn direct_record_and_readback_trap_uses_readback_rings_only() {
            let harness = PipelineHarness::new("trap-sidecar allocation test");
            let with_rings_arena = harness.arena();
            let with_rings_pool = with_rings_arena.pool().clone();

            let program = Program::wrapped(
                vec![],
                [1, 1, 1],
                vec![Node::trap(Expr::u32(3), "direct-readback-ring-trap")],
            );

            let pipeline = harness
                .compile_on_arena(&program, &with_rings_arena)
                .expect(
                    "Fix: trapped program compile must succeed; restore this invariant before continuing.",
                );

            let before_allocations = with_rings_pool.stats().allocations;
            let error = record_once(
                &pipeline,
                &with_rings_arena,
                true,
                DispatchLabels {
                    bind_group: "vyre direct trap readback ring test bind group",
                    encoder: "vyre direct trap readback ring test",
                    compute: "vyre direct trap readback ring test compute",
                },
            )
            .expect_err(
                "Fix: trapped dispatch with readback rings must return the underlying trap sidecar error and not succeed",
            );
            let after_allocations = with_rings_pool.stats().allocations;

            assert!(
                error.to_string().contains("wgpu dispatch trapped"),
                "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
            );
            assert!(
                error.to_string().contains("direct-readback-ring-trap"),
                "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
            );
            assert_eq!(
                after_allocations,
                before_allocations + 1,
                "Fix: readback-ring trap path must use the dedicated trap sidecar buffer only (no pooled full-sidecar readback buffer allocation).",
            );
        }

        #[test]

        fn direct_record_and_readback_trap_without_readback_rings_allocates_full_sidecar_copy() {
            let harness = PipelineHarness::new("trap-sidecar allocation delta test");
            let arena = harness.arena();
            let pool = arena.pool().clone();

            let program = Program::wrapped(
                vec![],
                [1, 1, 1],
                vec![Node::trap(Expr::u32(5), "direct-readback-no-ring-trap")],
            );

            let pipeline = harness.compile_on_arena(&program, &arena).expect(
                "Fix: trapped program compile must succeed; restore this invariant before continuing.",
            );

            let before_allocations = pool.stats().allocations;
            let error = record_once(
                &pipeline,
                &arena,
                false,
                DispatchLabels {
                    bind_group: "vyre direct trap readback no-ring test bind group",
                    encoder: "vyre direct trap readback no-ring test",
                    compute: "vyre direct trap readback no-ring test compute",
                },
            )
            .expect_err(
                "Fix: trapped dispatch without rings must still return the underlying trap sidecar error and not succeed",
            );
            let after_allocations = pool.stats().allocations;

            assert!(
                error.to_string().contains("wgpu dispatch trapped"),
                "Fix: expected trap dispatch to surface a backend trap error, got: {error}"
            );
            assert!(
                error.to_string().contains("direct-readback-no-ring-trap"),
                "Fix: expected the trap tag to be preserved across fallback sidecar decode, got: {error}"
            );
            assert_eq!(
                after_allocations,
                before_allocations + 2,
                "Fix: non-ring trap path must allocate exactly the full-sidecar pooled readback buffer plus trap sidecar allocation (before={before_allocations}, after={after_allocations})."
            );
        }
    }

    mod trap_output_contracts {
        use super::*;

        #[test]
        fn direct_record_and_readback_trap_with_output_preserves_ring_fast_path() {
            let harness = PipelineHarness::new("trap+output readback allocation contract test");
            let with_rings_arena = harness.arena();
            let without_rings_arena = harness.arena();
            let with_rings_pool = with_rings_arena.pool().clone();
            let without_rings_pool = without_rings_arena.pool().clone();

            let program = Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![
                    Node::store("out", Expr::u32(0), Expr::u32(99)),
                    Node::trap(Expr::u32(9), "mixed-output-ring-trap"),
                ],
            );

            let pipeline = harness
                .compile_on_arena(&program, &with_rings_arena)
                .expect("Fix: trapped program with output compile must succeed; restore this invariant before continuing.");

            let with_rings_before = with_rings_pool.stats().allocations;
            let with_rings_error = record_once(
                &pipeline,
                &with_rings_arena,
                true,
                DispatchLabels {
                    bind_group: "vyre mixed output ring test bind group",
                    encoder: "vyre mixed output ring test",
                    compute: "vyre mixed output ring test compute",
                },
            )
            .expect_err(
                "Fix: trapped dispatch with output and rings must still surface trap errors and not succeed",
            );
            let with_rings_after = with_rings_pool.stats().allocations;

            assert!(
                with_rings_error
                    .to_string()
                    .contains("wgpu dispatch trapped"),
                "Fix: expected trap dispatch to surface a backend trap error, got: {with_rings_error}"
            );
            assert!(
                with_rings_error.to_string().contains("mixed-output-ring-trap"),
                "Fix: expected trap tag to be preserved through mixed-output ring path, got: {with_rings_error}"
            );
            assert_eq!(
                with_rings_after,
                with_rings_before + 2,
                "Fix: ring-backed mixed output+trap path should add only output + trap buffer allocations from pool before first successful mapping.",
            );

            let without_rings_before = without_rings_pool.stats().allocations;
            let without_rings_error = record_once(
                &pipeline,
                &without_rings_arena,
                false,
                DispatchLabels {
                    bind_group: "vyre mixed output no-ring test bind group",
                    encoder: "vyre mixed output no-ring test",
                    compute: "vyre mixed output no-ring test compute",
                },
            )
            .expect_err(
                "Fix: trapped dispatch without rings should surface the trap error and not succeed",
            );
            let without_rings_after = without_rings_pool.stats().allocations;

            assert!(
                without_rings_error
                    .to_string()
                    .contains("wgpu dispatch trapped"),
                "Fix: expected trap dispatch to surface a backend trap error, got: {without_rings_error}"
            );
            assert!(
                without_rings_error.to_string().contains("mixed-output-ring-trap"),
                "Fix: expected trap tag to be preserved through mixed-output fallback path, got: {without_rings_error}"
            );
            assert_eq!(
                without_rings_after,
                without_rings_before + 4,
                "Fix: no-ring mixed output+trap path should allocate output storage, trap storage, output readback, and trap readback buffers; ring-backed dispatch must be the path that avoids the two pooled readback allocations.",
            );
        }
    }
}
