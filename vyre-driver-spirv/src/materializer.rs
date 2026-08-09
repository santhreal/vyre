use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, BoundResource,
    Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    fuse_selected_module, selected_modules, Artifact, ArtifactValueId, Digest, ResourceLifetime,
    TargetModuleBundle, TargetPayload, TargetPayloadFormat,
};

use crate::{vulkan, SPIRV_BACKEND_ID};

struct SpirvDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
}

impl Device for SpirvDevice {
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

pub(crate) struct SpirvMaterializer {
    device: Arc<vulkan::VulkanDevice>,
    descriptor: SpirvDevice,
}

impl ArtifactMaterializer for SpirvMaterializer {
    fn device(&self) -> &dyn Device {
        &self.descriptor
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
                backend: SPIRV_BACKEND_ID.to_string(),
            });
        }
        let bundle = TargetModuleBundle::from_bytes(payload.bytes()).map_err(compile_error)?;
        let selected = selected_modules(artifact).map_err(compile_error)?;
        if bundle.modules.len() != selected.len() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: target module count must equal the compiler-selected fusion-group count. Recompile the payload from this artifact.".to_string(),
            });
        }
        let mut modules = Vec::with_capacity(selected.len());
        for (image, selected) in bundle.modules.into_iter().zip(selected) {
            if image.group != selected.group || image.stage != selected.stage {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: target module group/stage identity must match the neutral selected plan. Recompile the payload.".to_string(),
                });
            }
            if image.bytes.len() % 4 != 0 {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: SPIR-V module byte length must be divisible by four.".to_string(),
                });
            }
            let words = image
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect::<Vec<_>>();
            if words.first().copied() != Some(0x0723_0203) {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: SPIR-V target module must begin with the SPIR-V magic word."
                        .to_string(),
                });
            }
            modules.push(SpirvExecutableModule {
                program: fuse_selected_module(&selected).map_err(compile_error)?,
                words,
            });
        }
        Ok(Box::new(SpirvArtifactInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.descriptor.identity.clone(),
            native: Arc::clone(&self.device),
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

struct SpirvExecutableModule {
    program: Program,
    words: Vec<u32>,
}

struct SpirvArtifactInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    native: Arc<vulkan::VulkanDevice>,
    modules: Vec<SpirvExecutableModule>,
    values: BTreeMap<String, ArtifactValueId>,
    outputs: BTreeSet<ArtifactValueId>,
}

impl ArtifactInstance for SpirvArtifactInstance {
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
        let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        for (value, resource) in bindings.resources() {
            match resource {
                BoundResource::Host(bytes) => {
                    state.insert(*value, bytes.clone());
                }
                BoundResource::Resident(_) => {
                    return Err(BackendError::UnsupportedFeature {
                        name: "SPIR-V artifact resident binding".to_string(),
                        backend: SPIRV_BACKEND_ID.to_string(),
                    });
                }
            }
        }
        let result = self.execute(state);
        Ok(Box::new(ReadySubmission {
            result: Some(result),
        }))
    }
}

impl SpirvArtifactInstance {
    fn execute(
        &self,
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Completion, BackendError> {
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
                let value = self.values.get(buffer.name()).ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: target Program buffer `{}` must project from the canonical artifact ABI.",
                            buffer.name()
                        ),
                    }
                })?;
                inputs[input_index] = state.get(value).map(Vec::as_slice).ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: bind canonical artifact value {} for Program buffer `{}` before submission.",
                            value.0,
                            buffer.name()
                        ),
                    }
                })?;
            }
            // SAFETY: `native` owns a live Vulkan device for the entire instance;
            // words were validated as aligned SPIR-V and Program metadata came
            // from the authenticated neutral artifact.
            let outputs = unsafe {
                vulkan::dispatch_program(
                    &self.native,
                    &module.program,
                    &module.words,
                    &inputs,
                    &DispatchConfig::default(),
                )
            }?;
            for binding in &plan.bindings {
                let Some(output_index) = binding.output_index else {
                    continue;
                };
                let buffer = &module.program.buffers()[binding.buffer_index];
                let value =
                    self.values
                        .get(buffer.name())
                        .ok_or_else(|| BackendError::InvalidProgram {
                            fix: format!(
                            "Fix: output buffer `{}` must project from the canonical artifact ABI.",
                            buffer.name()
                        ),
                        })?;
                if let Some(bytes) = outputs.get(output_index) {
                    state.insert(*value, bytes.clone());
                }
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
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: selected execution must produce canonical output value {}.",
                            value.0
                        ),
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Completion {
            artifact: self.artifact,
            outputs,
            device_ns: None,
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
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: consume each Submission completion exactly once.".to_string(),
            })?
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let native = Arc::new(vulkan::VulkanDevice::acquire()?);
    let format = TargetPayloadFormat::new("spv", 1).map_err(compile_error)?;
    let generation = ResidentOwner::new()?.get();
    Ok(Box::new(SpirvMaterializer {
        device: native,
        descriptor: SpirvDevice {
            identity: DeviceIdentity {
                backend: SPIRV_BACKEND_ID,
                device: "vulkan-compute".to_string(),
                generation,
            },
            format,
        },
    }))
}

fn compile_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: SPIRV_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}
