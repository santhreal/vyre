use std::collections::BTreeMap;
use std::sync::Arc;

use vyre_driver::materialize::{self, InstanceCore, InstanceMessages, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, Completion,
    Device, DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, Digest, TargetPayload, TargetPayloadFormat};

use crate::{vulkan, SPIRV_BACKEND_ID};

/// SPIR-V rejection text, unchanged from when this crate owned the whole path.
const MESSAGES: InstanceMessages = InstanceMessages {
    foreign_artifact: || BackendError::InvalidProgram {
        fix: "Fix: bind resources against the exact artifact digest owned by this instance."
            .to_string(),
    },
    unmapped_buffer: |name| BackendError::InvalidProgram {
        fix: format!(
            "Fix: target Program buffer `{name}` must project from the canonical artifact ABI."
        ),
    },
    missing_output_value: |value| BackendError::InvalidProgram {
        fix: format!(
            "Fix: selected execution must produce canonical output value {}.",
            value.0
        ),
    },
    missing_retained_value: |value| BackendError::InvalidProgram {
        fix: format!(
            "Fix: selected execution must preserve retained value {}.",
            value.0
        ),
    },
    completion_consumed: || BackendError::InvalidProgram {
        fix: "Fix: consume each Submission completion exactly once.".to_string(),
    },
};

/// Rejection for a declared input whose canonical value was never bound.
fn unbound_input(value: ArtifactValueId, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: bind canonical artifact value {} for Program buffer `{name}` before submission.",
            value.0
        ),
    }
}

/// First word of every well-formed SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

pub(crate) struct SpirvMaterializer {
    device: Arc<vulkan::VulkanDevice>,
    descriptor: MaterializerDevice,
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
        let admitted =
            materialize::admit(artifact, payload, self.descriptor.target(SPIRV_BACKEND_ID))?;
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
            if words.first().copied() != Some(SPIRV_MAGIC) {
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
        Ok(Box::new(SpirvArtifactInstance {
            core: self.descriptor.instance(artifact, payload, MESSAGES),
            native: Arc::clone(&self.device),
            modules,
        }))
    }
}

struct SpirvExecutableModule {
    program: Arc<Program>,
    words: Vec<u32>,
    config: DispatchConfig,
}

struct SpirvArtifactInstance {
    core: InstanceCore,
    native: Arc<vulkan::VulkanDevice>,
    modules: Vec<SpirvExecutableModule>,
}

impl ArtifactInstance for SpirvArtifactInstance {
    fn artifact(&self) -> Digest {
        self.core.artifact
    }

    fn payload(&self) -> Digest {
        self.core.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.core.device
    }

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        self.core.accept(&bindings)?;
        let invocation_grid = bindings.invocation_grid();
        let state = materialize::host_only_bindings(
            &bindings,
            "SPIR-V artifact resident binding",
            SPIRV_BACKEND_ID,
        )?;
        Ok(self.core.ready(self.execute(state, invocation_grid)))
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
            materialize::override_grid(&mut config, invocation_grid);
            let plan = BindingPlan::build(&module.program)?;
            let inputs = self
                .core
                .gather_inputs(&plan, &module.program, &state, unbound_input)?;
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
            // A SPIR-V dispatch that returns fewer buffers than the plan declares
            // leaves the earlier value in place rather than failing the submission.
            for binding in &plan.bindings {
                let Some(output_index) = binding.output_index else {
                    continue;
                };
                let buffer = &module.program.buffers()[binding.buffer_index];
                let value = self.core.values.get(buffer.name()).ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: output buffer `{}` must project from the canonical artifact ABI.",
                            buffer.name()
                        ),
                    }
                })?;
                if let Some(bytes) = outputs.get(output_index) {
                    state.insert(*value, bytes.clone());
                }
            }
        }
        self.core.completion(&state, None)
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let native = Arc::new(vulkan::VulkanDevice::acquire()?);
    let format = TargetPayloadFormat::new("spv", 1)
        .map_err(|error| materialize::compile_error(SPIRV_BACKEND_ID, error))?;
    let profile = crate::target_compiler::target_profile()?;
    let generation = ResidentOwner::new()?.get();
    Ok(Box::new(SpirvMaterializer {
        device: native,
        descriptor: MaterializerDevice::new(
            DeviceIdentity {
                backend: SPIRV_BACKEND_ID,
                device: "vulkan-compute".to_string(),
                generation,
            },
            format,
            profile,
        ),
    }))
}
