use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, BoundResource,
    CompiledPipeline, Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner,
    Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    fuse_selected_module, selected_modules, Artifact, ArtifactValueId, Digest, ResourceLifetime,
    TargetModuleBundle, TargetPayload, TargetPayloadFormat,
};

use crate::pipeline::WgpuPipeline;
use crate::target_compiler::{
    WgpuTargetModule, WGPU_TARGET_FORMAT_VERSION, WGPU_TARGET_MODULE_SCHEMA_VERSION,
};
use crate::{WgpuBackend, WGPU_BACKEND_ID};

struct WgpuDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
    lost: Arc<AtomicBool>,
}

impl Device for WgpuDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn is_healthy(&self) -> bool {
        !self.lost.load(Ordering::Acquire)
    }
}

pub(crate) struct WgpuMaterializer {
    backend: WgpuBackend,
    descriptor: WgpuDevice,
}

impl ArtifactMaterializer for WgpuMaterializer {
    fn device(&self) -> &dyn Device {
        &self.descriptor
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if !self.descriptor.is_healthy() {
            return Err(BackendError::DispatchFailed {
                code: None,
                message: format!(
                    "WGPU device generation {} is lost; reacquire and rematerialize the artifact",
                    self.descriptor.identity.generation
                ),
            });
        }
        if payload.neutral_artifact() != artifact.digest() {
            return Err(invalid_module(
                "target payload is not authenticated for the supplied neutral artifact",
            ));
        }
        if payload.format() != self.descriptor.target_format() {
            return Err(BackendError::UnsupportedFeature {
                name: format!("target payload format `{}`", payload.format().identity()),
                backend: WGPU_BACKEND_ID.to_string(),
            });
        }
        let bundle = TargetModuleBundle::from_bytes(payload.bytes()).map_err(compile_error)?;
        let selected = selected_modules(artifact).map_err(compile_error)?;
        if bundle.modules.len() != selected.len() {
            return Err(invalid_module(
                "target module count must equal the compiler-selected fusion-group count",
            ));
        }
        let config = DispatchConfig::default();
        let mut modules = Vec::with_capacity(selected.len());
        for (image, selected) in bundle.modules.into_iter().zip(selected) {
            if image.group != selected.group || image.stage != selected.stage {
                return Err(invalid_module(
                    "target module group/stage identity must match the neutral selected plan",
                ));
            }
            if image.entry_point != "main" {
                return Err(invalid_module(
                    "WGSL target module entry point must be `main`",
                ));
            }
            let target: WgpuTargetModule =
                serde_json::from_slice(&image.bytes).map_err(|error| {
                    invalid_module(&format!("WGSL target module is malformed: {error}"))
                })?;
            if target.schema_version != WGPU_TARGET_MODULE_SCHEMA_VERSION {
                return Err(invalid_module("WGSL target module schema is unsupported"));
            }
            if !target.wgsl.contains("@compute") || !target.wgsl.contains("fn main(") {
                return Err(invalid_module(
                    "WGSL target module does not define compute entry point `main`",
                ));
            }
            let program = Arc::new(fuse_selected_module(&selected).map_err(compile_error)?);
            let pipeline = WgpuPipeline::compile_target_with_device_queue(
                &program,
                &target.wgsl,
                &target.descriptor,
                &config,
                self.backend.adapter_info.clone(),
                self.backend.enabled_features,
                self.backend.current_device_queue(),
                self.backend.dispatch_arena_snapshot(),
                self.backend.current_persistent_pool(),
                Arc::clone(&self.backend.pipeline_cache),
                Arc::clone(&self.backend.bind_group_layout_cache),
            )?;
            modules.push(WgpuExecutableModule { program, pipeline });
        }
        Ok(Box::new(WgpuArtifactInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.descriptor.identity.clone(),
            modules,
            values: artifact
                .resources()
                .iter()
                .map(|resource| (resource.name.clone(), resource.value))
                .collect(),
            outputs: artifact
                .resources()
                .iter()
                .filter(|resource| resource.lifetime == ResourceLifetime::Output)
                .map(|resource| resource.value)
                .collect(),
        }))
    }
}

struct WgpuExecutableModule {
    program: Arc<Program>,
    pipeline: Arc<WgpuPipeline>,
}

struct WgpuArtifactInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    modules: Vec<WgpuExecutableModule>,
    values: BTreeMap<String, ArtifactValueId>,
    outputs: BTreeSet<ArtifactValueId>,
}

impl ArtifactInstance for WgpuArtifactInstance {
    fn artifact(&self) -> Digest {
        self.artifact
    }

    fn payload(&self) -> Digest {
        self.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        if bindings.artifact() != self.artifact {
            return Err(invalid_module("bindings name a different neutral artifact"));
        }
        let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        for (value, resource) in bindings.resources() {
            match resource {
                BoundResource::Host(bytes) => {
                    state.insert(*value, bytes.clone());
                }
                BoundResource::Resident(_) => {
                    return Err(BackendError::UnsupportedFeature {
                        name: "WGPU artifact resident binding".to_string(),
                        backend: WGPU_BACKEND_ID.to_string(),
                    });
                }
            }
        }
        Ok(Box::new(ReadySubmission {
            result: Some(self.execute(state)),
        }))
    }
}

impl WgpuArtifactInstance {
    fn execute(
        &self,
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Completion, BackendError> {
        let config = DispatchConfig::default();
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for module in &self.modules {
            let plan = BindingPlan::build(&module.program)?;
            let input_count = plan
                .bindings
                .iter()
                .filter_map(|binding| binding.input_index)
                .max()
                .map_or(0, |index| index + 1);
            let mut inputs = vec![&[][..]; input_count];
            for binding in &plan.bindings {
                let Some(input_index) = binding.input_index else {
                    continue;
                };
                let buffer = &module.program.buffers()[binding.buffer_index];
                let value = self.value_for_buffer(buffer.name())?;
                inputs[input_index] = state.get(&value).map(Vec::as_slice).ok_or_else(|| {
                    invalid_module(&format!(
                        "canonical artifact value {} for Program buffer `{}` is unbound",
                        value.0,
                        buffer.name()
                    ))
                })?;
            }
            let dispatched = module.pipeline.dispatch_borrowed_timed(&inputs, &config)?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            for binding in &plan.bindings {
                let Some(output_index) = binding.output_index else {
                    continue;
                };
                let buffer = &module.program.buffers()[binding.buffer_index];
                let value = self.value_for_buffer(buffer.name())?;
                let bytes = dispatched.outputs.get(output_index).ok_or_else(|| {
                    invalid_module(&format!(
                        "WGSL target module omitted output {output_index} for Program buffer `{}`",
                        buffer.name()
                    ))
                })?;
                state.insert(value, bytes.clone());
            }
        }
        let outputs = self
            .outputs
            .iter()
            .map(|value| {
                state
                    .get(value)
                    .cloned()
                    .map(|bytes| (*value, bytes))
                    .ok_or_else(|| {
                        invalid_module(&format!(
                            "selected execution did not produce canonical output value {}",
                            value.0
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Completion {
            artifact: self.artifact,
            outputs,
            device_ns: has_device_timing.then_some(device_ns),
        })
    }

    fn value_for_buffer(&self, name: &str) -> Result<ArtifactValueId, BackendError> {
        self.values.get(name).copied().ok_or_else(|| {
            invalid_module(&format!(
                "Program buffer `{name}` is absent from the canonical artifact ABI"
            ))
        })
    }
}

struct ReadySubmission {
    result: Option<Result<Completion, BackendError>>,
}

impl Submission for ReadySubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.result
            .take()
            .ok_or_else(|| invalid_module("each Submission completion may be consumed only once"))?
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let backend = WgpuBackend::acquire()?;
    let format =
        TargetPayloadFormat::new("wgsl", WGPU_TARGET_FORMAT_VERSION).map_err(compile_error)?;
    let generation = ResidentOwner::new()?.get();
    let device = backend.adapter_name.to_string();
    let lost = Arc::clone(&backend.device_lost);
    Ok(Box::new(WgpuMaterializer {
        backend,
        descriptor: WgpuDevice {
            identity: DeviceIdentity {
                backend: WGPU_BACKEND_ID,
                device,
                generation,
            },
            format,
            lost,
        },
    }))
}

fn invalid_module(reason: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!("Fix: {reason}. Recompile the target payload from the neutral artifact."),
    }
}

fn compile_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: WGPU_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}
