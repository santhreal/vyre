use std::sync::Arc;
use std::time::Instant;

use vyre_driver::materialize::{
    self, DeviceSpec, ExecutableModule, InstanceCore, MaterializedInstance, MaterializerDevice,
};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, DispatchConfig, Submission,
    TimedDispatchResult,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactInputSlot, TargetPayload};

use crate::{vulkan, SPIRV_BACKEND_ID};

/// First word of every well-formed SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

pub(crate) struct SpirvMaterializer {
    device: Arc<vulkan::VulkanDevice>,
    descriptor: MaterializerDevice,
}

impl ArtifactMaterializer for SpirvMaterializer {
    vyre_driver::materializer_passthrough!();

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        let modules = self.descriptor.admit_modules(
            SPIRV_BACKEND_ID,
            artifact,
            payload,
            |admitted_module| {
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
                let input_slots = vyre_megakernel::staged_input_slots(
                    &admitted_module.image.descriptor,
                    &admitted_module.resource_bindings,
                    &admitted_module.program,
                )
                .map_err(|error| materialize::compile_error(SPIRV_BACKEND_ID, error))?;
                Ok(SpirvExecutableModule {
                    program: admitted_module.program,
                    words,
                    config: admitted_module.config,
                    input_slots,
                })
            },
        )?;
        Ok(Box::new(SpirvArtifactInstance {
            core: self
                .descriptor
                .instance(artifact, payload, materialize::NEUTRAL_MESSAGES)?,
            native: Arc::clone(&self.device),
            modules,
        }))
    }
}

struct SpirvExecutableModule {
    program: Arc<Program>,
    words: Vec<u32>,
    config: DispatchConfig,
    /// Bindings this module's launch stages bytes into, in target binding
    /// order. A fused artifact carries a value from one module to the next
    /// through a buffer the reading module's Program declares as consuming no
    /// host input, so the Program's input order is not this order.
    input_slots: Vec<ArtifactInputSlot>,
}

struct SpirvArtifactInstance {
    core: InstanceCore,
    native: Arc<vulkan::VulkanDevice>,
    modules: Vec<SpirvExecutableModule>,
}

impl ExecutableModule for SpirvExecutableModule {
    vyre_driver::executable_module!();
}

impl ArtifactInstance for SpirvArtifactInstance {
    vyre_driver::artifact_instance_identity!();
    vyre_driver::artifact_instance_unreported_resources!();

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        self.submit_host_only(&bindings, "SPIR-V artifact resident binding")
    }
}

impl MaterializedInstance for SpirvArtifactInstance {
    type Module = SpirvExecutableModule;

    fn core(&self) -> &InstanceCore {
        &self.core
    }

    fn modules(&self) -> &[Self::Module] {
        &self.modules
    }

    fn module_label(&self) -> &'static str {
        "SPIR-V target module"
    }

    /// Stage this module's inputs in target binding order.
    ///
    /// The default walks the Program's host-input order, which omits a buffer
    /// a previous module wrote and this one reads.
    fn gather<'a>(
        &'a self,
        module_index: usize,
        module: &'a Self::Module,
        _plan: &vyre_driver::BindingPlan,
        state: &'a std::collections::BTreeMap<vyre_megakernel::ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'a [u8]>, BackendError> {
        materialize::gather_artifact_inputs(&self.core, module_index, &module.input_slots, state)
    }

    fn dispatch(
        &self,
        module: &Self::Module,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let started = Instant::now();
        let input_names = module
            .input_slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect::<Vec<_>>();
        // SAFETY: `native` owns a live Vulkan device for the entire instance;
        // words were validated as aligned SPIR-V and Program metadata came
        // from the authenticated neutral artifact.
        let outputs = unsafe {
            vulkan::dispatch_program(
                &self.native,
                &module.program,
                &module.words,
                inputs,
                vulkan::InputOrder::Named(&input_names),
                config,
            )
        }?;
        Ok(TimedDispatchResult::host_timed(
            outputs,
            u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
                BackendError::DispatchFailed {
                    code: None,
                    message: "SPIR-V dispatch duration overflowed a 64-bit nanosecond count"
                        .to_string(),
                }
            })?,
        ))
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let native = vulkan::shared_device()?;
    Ok(Box::new(SpirvMaterializer {
        device: native,
        descriptor: MaterializerDevice::acquire(DeviceSpec {
            backend: SPIRV_BACKEND_ID,
            device: "vulkan-compute".to_string(),
            format_extension: "spv",
            format_version: 1,
            profile: crate::target_compiler::target_profile()?,
        })?,
    }))
}
