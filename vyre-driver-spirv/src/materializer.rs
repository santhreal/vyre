use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, BoundResource,
    Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
};
use vyre_driver::materialize;
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Artifact, ArtifactValueId, Digest, TargetPayload,
    TargetPayloadFormat, TargetProfile,
};

use crate::{vulkan, SPIRV_BACKEND_ID};

struct SpirvDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl Device for SpirvDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn target_profile(&self) -> &TargetProfile {
        &self.profile
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
        let admitted = materialize::admit(
            artifact,
            payload,
            materialize::MaterializerTarget {
                backend_id: SPIRV_BACKEND_ID,
                format: self.descriptor.target_format(),
                profile: self.descriptor.target_profile(),
            },
        )?;
        let mut modules = Vec::with_capacity(admitted.len());
        for admitted_module in admitted {
            if admitted_module.image.bytes.len() % 4 != 0 {
                return Err(materialize::invalid_module(
                    "SPIR-V module byte length must be divisible by four",
                ));
            }
            let words = admitted_module
                .image
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect::<Vec<_>>();
            if words.first().copied() != Some(0x0723_0203) {
                return Err(materialize::invalid_module(
                    "SPIR-V target module must begin with the SPIR-V magic word",
                ));
            }
            modules.push(SpirvExecutableModule {
                program: admitted_module.program,
                words,
                config: admitted_module.config,
            });
        }
        let resources = materialize::project_resources(artifact);
        Ok(Box::new(SpirvArtifactInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.descriptor.identity.clone(),
            native: Arc::clone(&self.device),
            modules,
            values: resources.values,
            outputs: resources.outputs,
            retained: resources.retained,
        }))
    }
}

struct SpirvExecutableModule {
    program: Arc<Program>,
    words: Vec<u32>,
    config: DispatchConfig,
}

struct SpirvArtifactInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    native: Arc<vulkan::VulkanDevice>,
    modules: Vec<SpirvExecutableModule>,
    values: BTreeMap<String, ArtifactValueId>,
    outputs: BTreeSet<ArtifactValueId>,
    retained: BTreeSet<ArtifactValueId>,
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
        let invocation_grid = bindings.invocation_grid();
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
        let result = self.execute(state, invocation_grid);
        Ok(Box::new(ReadySubmission {
            result: Some(result),
        }))
    }
}

impl SpirvArtifactInstance {
    fn execute(
        &self,
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        for module in &self.modules {
            let mut config = module.config.clone();
            if let Some(grid) = invocation_grid {
                config.grid_override = Some(grid);
                config.dispatch_grid = Some(grid);
            }
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
                    &config,
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
        let retained = self
            .retained
            .iter()
            .map(|value| {
                state
                    .get(value)
                    .cloned()
                    .map(|bytes| (*value, bytes))
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: selected execution must preserve retained value {}.",
                            value.0
                        ),
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Completion {
            artifact: self.artifact,
            outputs,
            retained,
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
    let format = TargetPayloadFormat::new("spv", 1).map_err(|error| materialize::compile_error(SPIRV_BACKEND_ID, error))?;
    let profile = crate::target_compiler::target_profile()?;
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
            profile,
        },
    }))
}
