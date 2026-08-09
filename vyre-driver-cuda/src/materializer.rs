use std::collections::{BTreeMap, BTreeSet};
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

use crate::backend::CudaBackend;
use crate::pipeline::CudaCompiledPipeline;
use crate::{CudaBackendRegistration, CUDA_BACKEND_ID};

struct CudaDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
}

impl Device for CudaDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

pub(crate) struct CudaMaterializer {
    backend: CudaBackend,
    resident: CudaBackendRegistration,
    descriptor: CudaDevice,
}

impl ArtifactMaterializer for CudaMaterializer {
    fn device(&self) -> &dyn Device {
        &self.descriptor
    }
    fn allocate_resident(&self, byte_len: usize) -> Result<vyre_driver::Resource, BackendError> {
        vyre_driver::VyreBackend::allocate_resident(&self.resident, byte_len)
    }

    fn upload_resident(
        &self,
        resource: &vyre_driver::Resource,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::upload_resident(&self.resident, resource, bytes)
    }

    fn free_resident(&self, resource: vyre_driver::Resource) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::free_resident(&self.resident, resource)
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if payload.neutral_artifact() != artifact.digest() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: materialize only a target payload authenticated for the supplied neutral artifact.".to_string(),
            });
        }
        if payload.format() != self.descriptor.target_format() {
            return Err(BackendError::UnsupportedFeature {
                name: format!("target payload format `{}`", payload.format().identity()),
                backend: CUDA_BACKEND_ID.to_string(),
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
                    "PTX target module entry point must be `main`",
                ));
            }
            let ptx = std::str::from_utf8(&image.bytes).map_err(|error| {
                invalid_module(&format!("PTX target module is not UTF-8: {error}"))
            })?;
            if !ptx.contains(".visible .entry main(") {
                return Err(invalid_module(
                    "PTX target module does not define `.visible .entry main`",
                ));
            }
            let program = Arc::new(fuse_selected_module(&selected).map_err(compile_error)?);
            let prepared = self.backend.prepare_static_dispatch(&program, &config)?;
            let module_key = self.backend.module_cache_key_for_raw_ptx_artifact(ptx)?;
            self.backend.module_for_ptx_with_key(ptx, module_key)?;
            let ptx: Arc<str> = Arc::from(ptx);
            let pipeline = Arc::new(CudaCompiledPipeline::new_from_target_payload(
                self.backend.clone(),
                Arc::clone(&program),
                ptx,
                module_key,
                &config,
                prepared,
            )?);
            modules.push(CudaExecutableModule { program, pipeline });
        }
        Ok(Box::new(CudaArtifactInstance {
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
            retained: artifact
                .resources()
                .iter()
                .filter(|resource| resource.lifetime == ResourceLifetime::Retained)
                .map(|resource| resource.value)
                .collect(),
        }))
    }
}

struct CudaExecutableModule {
    program: Arc<Program>,
    pipeline: Arc<CudaCompiledPipeline>,
}

struct CudaArtifactInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    modules: Vec<CudaExecutableModule>,
    values: BTreeMap<String, ArtifactValueId>,
    outputs: BTreeSet<ArtifactValueId>,
    retained: BTreeSet<ArtifactValueId>,
}

impl ArtifactInstance for CudaArtifactInstance {
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
            return Err(BackendError::InvalidProgram {
                fix:
                    "Fix: bind resources against the exact artifact digest owned by this instance."
                        .to_string(),
            });
        }
        let mut host_state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        let mut resident_state = BTreeMap::<ArtifactValueId, vyre_driver::Resource>::new();
        for (value, resource) in bindings.resources() {
            match resource {
                BoundResource::Host(bytes) => {
                    host_state.insert(*value, bytes.clone());
                }
                BoundResource::Resident(resource) => {
                    resident_state.insert(*value, resource.clone());
                }
            }
        }
        if !host_state.is_empty() && !resident_state.is_empty() {
            return Err(invalid_module(
                "CUDA artifact submission cannot mix host and resident resources",
            ));
        }
        let result = if resident_state.is_empty() {
            self.execute(host_state)
        } else {
            self.execute_resident(resident_state)
        };
        Ok(Box::new(ReadySubmission {
            result: Some(result),
        }))
    }
}

impl CudaArtifactInstance {
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
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: bind canonical artifact value {} for Program buffer `{}` before submission.",
                            value.0,
                            buffer.name()
                        ),
                    }
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
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA target module omitted output {} for Program buffer `{}`.",
                            output_index,
                            buffer.name()
                        ),
                    }
                })?;
                state.insert(value, bytes.clone());
            }
        }
        let outputs = project_outputs(&self.outputs, &state)?;
        let retained = project_outputs(&self.retained, &state)?;
        Ok(Completion {
            artifact: self.artifact,
            outputs,
            retained,
            device_ns: has_device_timing.then_some(device_ns),
        })
    }
    fn execute_resident(
        &self,
        resources: BTreeMap<ArtifactValueId, vyre_driver::Resource>,
    ) -> Result<Completion, BackendError> {
        if self.modules.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "CUDA resident submission for multi-module artifacts".to_string(),
                backend: CUDA_BACKEND_ID.to_string(),
            });
        }
        let module = &self.modules[0];
        let plan = BindingPlan::build(&module.program)?;
        let mut ordered = Vec::with_capacity(plan.bindings.len());
        for binding in &plan.bindings {
            let buffer = &module.program.buffers()[binding.buffer_index];
            let value = self.value_for_buffer(buffer.name())?;
            let resource = resources
                .get(&value)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: bind canonical artifact value {} for resident Program buffer `{}`.",
                        value.0,
                        buffer.name()
                    ),
                })?;
            ordered.push(resource.clone());
        }
        let dispatched = module
            .pipeline
            .dispatch_persistent_handles_timed(&ordered, &DispatchConfig::default())?;
        let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        for binding in &plan.bindings {
            let Some(output_index) = binding.output_index else {
                continue;
            };
            let buffer = &module.program.buffers()[binding.buffer_index];
            let value = self.value_for_buffer(buffer.name())?;
            let bytes = dispatched.outputs.get(output_index).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident target module omitted output {output_index} for Program buffer `{}`.",
                        buffer.name()
                    ),
                }
            })?;
            state.insert(value, bytes.clone());
        }
        Ok(Completion {
            artifact: self.artifact,
            outputs: project_outputs(&self.outputs, &state)?,
            retained: project_outputs(&self.retained, &state)?,
            device_ns: dispatched.device_ns,
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
    let backend = CudaBackend::acquire().map_err(|message| BackendError::DispatchFailed {
        code: None,
        message: format!("CUDA artifact device acquisition failed: {message}"),
    })?;
    let format = TargetPayloadFormat::new("ptx", 1).map_err(compile_error)?;
    let generation = ResidentOwner::new()?.get();
    let device = backend.caps.name.clone();
    Ok(Box::new(CudaMaterializer {
        resident: CudaBackendRegistration {
            inner: backend.clone(),
        },
        backend,
        descriptor: CudaDevice {
            identity: DeviceIdentity {
                backend: CUDA_BACKEND_ID,
                device,
                generation,
            },
            format,
        },
    }))
}

fn project_outputs(
    expected: &BTreeSet<ArtifactValueId>,
    state: &BTreeMap<ArtifactValueId, Vec<u8>>,
) -> Result<BTreeMap<ArtifactValueId, Vec<u8>>, BackendError> {
    expected
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
        .collect()
}

fn invalid_module(reason: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!("Fix: {reason}. Recompile the target payload from the neutral artifact."),
    }
}

fn compile_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: CUDA_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}
