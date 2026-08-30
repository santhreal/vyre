use std::sync::atomic::Ordering;
use std::time::Instant;

use metal::Device;
use vyre_driver::{
    output_binding_layouts, BackendError, BindingPlan, DispatchConfig, PipelineCacheIdentity,
    PipelineCacheMissReason, TimedDispatchResult,
};
use vyre_foundation::ir::Program;

use super::buffer_plan::{metal_slot_map, output_layout_map, plan_buffers};
use super::dispatch::validate_metal_dispatch_config;
use super::metrics::elapsed_ns;
use super::resident::ns_uint_to_u32_saturating;
use super::MetalBackend;
use crate::METAL_BACKEND_ID;

#[derive(Clone)]
pub(super) struct MetalCompiledPipeline {
    pub(super) identity: PipelineCacheIdentity,
    pub(super) artifact: vyre_emit_metal::MetalArtifact,
    pub(super) pipeline: metal::ComputePipelineState,
}

pub(crate) struct MetalTargetModule {
    pub(crate) artifact: vyre_emit_metal::MetalArtifact,
    pub(crate) pipeline: metal::ComputePipelineState,
}

pub(super) fn metal_pipeline_cache_key(
    program: &Program,
    config: &DispatchConfig,
    device: &Device,
) -> Result<PipelineCacheIdentity, BackendError> {
    let device_name = device.name();
    let revision_extra = format!(
        "artifact_schema={}:msl={}.{}:driver={}:device={}",
        vyre_emit_metal::METAL_ARTIFACT_SCHEMA,
        vyre_emit_metal::DEFAULT_MSL_VERSION.0,
        vyre_emit_metal::DEFAULT_MSL_VERSION.1,
        env!("CARGO_PKG_VERSION"),
        device_name
    );
    let fingerprint =
        vyre_driver::PipelineDeviceFingerprint::from_parts(0x106b, 0, "metal", &revision_extra);
    PipelineCacheIdentity::try_from_program(program, config, fingerprint).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal pipeline cache could not build shared Program/policy/device identity: {error}"
            ),
        }
    })
}

impl MetalBackend {
    pub(super) fn compile_pipeline(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<
        (
            PipelineCacheIdentity,
            vyre_emit_metal::MetalArtifact,
            metal::ComputePipelineState,
        ),
        BackendError,
    > {
        let cache_identity = metal_pipeline_cache_key(program, config, &self.device)?;
        let miss_reason = {
            let cache = self.lock_pipeline_cache("pipeline cache lookup")?;
            if let Some(hit) = cache.get(&cache_identity.digest).cloned() {
                self.metrics
                    .pipeline_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok((hit.identity, hit.artifact, hit.pipeline));
            }
            PipelineCacheMissReason::classify_identities(
                cache.values().map(|entry| &entry.identity),
                &cache_identity,
            )
        };
        self.metrics
            .pipeline_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        self.record_pipeline_cache_miss_reason(miss_reason);
        let lowered = vyre_lower::lower_physical(program).map_err(|error| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "pre-emission lowering failed before Metal compilation: {error}"
                ),
            }
        })?;
        let artifact = vyre_emit_metal::emit_artifact(lowered.descriptor()).map_err(|error| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!("MSL artifact emission failed: {error}"),
            }
        })?;
        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(&artifact.msl, &options)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!("Metal library compilation failed: {error}"),
            })?;
        let function = library
            .get_function(&artifact.entry_point, None)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal entry point `{}` lookup failed: {error}",
                    artifact.entry_point
                ),
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal compute pipeline creation failed for `{}`: {error}",
                    artifact.entry_point
                ),
            })?;
        // Cache the SIMD-group width from the first successfully compiled pipeline.
        // Metal exposes this per-pipeline-state (not per-device), so we store it
        // on first compile. If concurrent compiles race here we may call
        // compare_exchange twice; both are racing to write the same value for a
        // given device family, so the last writer wins and no correctness is lost.
        let thread_width = ns_uint_to_u32_saturating(pipeline.thread_execution_width());
        if thread_width > 0 {
            let _ = self.cached_simd_width.compare_exchange(
                0,
                thread_width,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        let compiled = MetalCompiledPipeline {
            identity: cache_identity,
            artifact,
            pipeline,
        };
        let mut cache = self.lock_pipeline_cache("pipeline cache insert")?;
        let cached = cache
            .entry(cache_identity.digest)
            .or_insert_with(|| compiled.clone());
        Ok((
            cached.identity,
            cached.artifact.clone(),
            cached.pipeline.clone(),
        ))
    }

    pub(crate) fn materialize_target_module(
        &self,
        artifact: vyre_emit_metal::MetalArtifact,
    ) -> Result<MetalTargetModule, BackendError> {
        // `main` is reserved in the Metal shading language, so the translator
        // renames the entry point it emits. The name is authenticated with the
        // rest of the artifact and looked up below, which reports a name the
        // library does not define; a name the artifact never stated is what
        // cannot be looked up at all.
        if artifact.entry_point.is_empty() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: authenticated Metal target states no entry point. Recompile the target payload from the neutral artifact.".to_string(),
            });
        }
        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(&artifact.msl, &options)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "authenticated Metal library compilation failed: {error}"
                ),
            })?;
        let function = library
            .get_function(&artifact.entry_point, None)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "authenticated Metal entry point `{}` lookup failed: {error}",
                    artifact.entry_point
                ),
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "authenticated Metal pipeline creation failed for `{}`: {error}",
                    artifact.entry_point
                ),
            })?;
        Ok(MetalTargetModule { artifact, pipeline })
    }

    pub(crate) fn dispatch_target_module(
        &self,
        module: &MetalTargetModule,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let started = Instant::now();
        validate_metal_dispatch_config(
            config,
            "Metal authenticated cooperative grid dispatch",
            "Metal authenticated repeated dispatch",
            "Metal authenticated dispatch",
        )?;
        let binding_plan = BindingPlan::from_borrowed_inputs(program, inputs)?;
        let output_layouts = output_binding_layouts(program)?;
        let output_by_binding = output_layout_map(output_layouts)?;
        let metal_slots = metal_slot_map(&module.artifact)?;
        let buffers = plan_buffers(
            &self.device,
            &binding_plan,
            inputs,
            &output_by_binding,
            &metal_slots,
            &module.artifact.bindings,
        )?;
        let result = self.dispatch_planned_buffers(
            program,
            &binding_plan,
            config,
            &module.artifact,
            &module.pipeline,
            buffers,
        )?;
        Ok(TimedDispatchResult::split_timed(
            result.outputs,
            elapsed_ns(started, "Metal authenticated timed dispatch")?,
            None,
            result.enqueue_ns,
            result.wait_ns,
        ))
    }

    fn record_pipeline_cache_miss_reason(&self, reason: PipelineCacheMissReason) {
        match reason {
            PipelineCacheMissReason::EmptyCache => {
                self.metrics
                    .pipeline_cache_miss_empty_cache
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineCacheMissReason::ProgramChanged => {
                self.metrics
                    .pipeline_cache_miss_program_changed
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineCacheMissReason::DispatchPolicyChanged => {
                self.metrics
                    .pipeline_cache_miss_dispatch_policy_changed
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineCacheMissReason::DeviceOrRuntimeChanged => {
                self.metrics
                    .pipeline_cache_miss_device_or_runtime_changed
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineCacheMissReason::KeyAbsent => {
                self.metrics
                    .pipeline_cache_miss_key_absent
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
