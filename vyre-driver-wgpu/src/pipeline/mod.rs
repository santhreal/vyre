//! Native pipeline-mode implementation for the wgpu backend.
//!
//! Compiled pipelines and bind-group layouts are reused so repeated
//! submissions pay only resource preparation, execution, and readback costs.

pub(crate) mod artifact;
pub(crate) mod binding;
pub(crate) mod bindings_reflection;
pub(crate) mod cache_impact;
pub(crate) mod compiled_dispatch;
pub(crate) mod compound;
pub(crate) mod descriptor_metadata;
pub(crate) mod disk_cache;
pub(crate) mod disk_cache_entries;
pub(crate) mod output_readback;
pub(crate) mod output_slots;
pub mod persistent;
pub(crate) mod persistent_resources;
#[cfg(test)]
mod tests;
pub(crate) mod tuning;

use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use vyre_driver::allocation::reserve_hash_set_to_capacity;
#[cfg(test)]
pub(crate) use vyre_driver::enforce_actual_output_budget;
use vyre_driver::BackendLayoutFingerprint;
use vyre_driver::{admit_dispatch_grid, find_indirect_dispatch, infer_dispatch_grid_for_count};
pub(crate) use vyre_driver::{element_size_bytes, OutputBindingLayout};
pub use vyre_driver::{output_layout_from_program, IndirectDispatch, OutputLayout};
use vyre_driver::{BackendError, DispatchConfig, LaunchGeometry, OutputBuffers};
use vyre_emit_naga::program::TrapTag;
use vyre_foundation::execution_plan::{self, ExecutionPlan};
use vyre_foundation::ir::Program;
use vyre_foundation::validate::ValidationOptions;
use vyre_lower::{TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS};

pub(crate) use self::artifact::AuthenticatedTarget;
use self::artifact::CachedPipelineArtifact;
pub(crate) use self::descriptor_metadata::BufferBindingInfo;
use self::descriptor_metadata::{
    bind_group_layout_fingerprint, create_bind_group_layouts, descriptor_buffer_bindings,
    descriptor_trap_tags,
};
use self::tuning::wgpu_effective_dispatch_config;
use crate::buffer::{BindGroupCache, StagingBufferPool};
use crate::pipeline::disk_cache::{
    compiled_pipeline_cache_key, create_compiled_pipeline_cache, early_pipeline_cache_key,
    load_or_compile_disk_wgsl, persist_compiled_pipeline_cache,
};
use crate::runtime;
use crate::staging_reserve::reserve_backend_vec;

pub(crate) type BindGroupLayoutCache = dashmap::DashMap<
    BackendLayoutFingerprint,
    Arc<[Arc<wgpu::BindGroupLayout>]>,
    BuildHasherDefault<rustc_hash::FxHasher>,
>;

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
        let authenticated_resource_bindings =
            target.as_ref().map(|target| target.resource_bindings);
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
        let mut host_input_bindings = FxHashSet::default();
        reserve_hash_set_to_capacity(
            &mut host_input_bindings,
            buffers.len(),
            "WGPU pipeline binding classification",
            "host input binding",
            "split the pipeline or reduce input binding fanout before compilation",
        )?;
        if let Some(resource_bindings) = authenticated_resource_bindings {
            host_input_bindings.extend(
                resource_bindings
                    .iter()
                    .filter(|binding| {
                        binding.access != vyre_megakernel::TargetResourceAccess::WriteOnly
                    })
                    .map(|binding| (binding.group, binding.slot)),
            );
        } else {
            host_input_bindings.extend(
                buffers
                    .iter()
                    .filter(|buffer| {
                        buffer.kind() != vyre_foundation::ir::MemoryKind::Shared
                            && !buffer.is_backend_allocated_output()
                    })
                    .map(|buffer| (0, buffer.binding())),
            );
        }

        let buffer_bindings: Arc<[BufferBindingInfo]> =
            descriptor_buffer_bindings(&descriptor, &public_output_bindings, &host_input_bindings)?
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
