use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::SPIRV_BACKEND_ID;

pub(crate) const SPIRV_TARGET_FORMAT: &str = "spv";
const SPIRV_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct SpirvTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for SpirvTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            |selected, _profile| {
                let words = vyre_emit_spirv::emit(&selected.descriptor)
                    .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
                let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
                for word in words {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                let grid_size = vyre_driver::infer_dispatch_grid_for_count(
                    selected.logical_element_count,
                    selected.descriptor.dispatch.workgroup_size,
                )
                .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
                Ok(EmittedTargetModule {
                    entry_point: "main".to_string(),
                    grid_size,
                    dynamic_shared_bytes: 0,
                    workgroup_size: selected.descriptor.dispatch.workgroup_size,
                    resource_bindings: selected.canonical_bindings.clone(),
                    bytes,
                })
            },
        )
    }
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    TargetProfile::new(
        SPIRV_TARGET_FORMAT,
        u64::from(SPIRV_TARGET_FORMAT_VERSION),
        [1_024, 1_024, 64],
        1_024,
        32_768,
        0,
    )
    .map_err(|error| BackendError::KernelCompileFailed {
        backend: SPIRV_BACKEND_ID.to_string(),
        compiler_message: format!(
            "SPIR-V target profile is invalid: {error}. Fix: repair the registered profile."
        ),
    })
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new(SPIRV_TARGET_FORMAT, SPIRV_TARGET_FORMAT_VERSION).map_err(|error| {
        BackendError::KernelCompileFailed {
            backend: SPIRV_BACKEND_ID.to_string(),
            compiler_message: format!(
                "SPIR-V target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        }
    })?;
    let profile = target_profile()?;
    Ok(Box::new(SpirvTargetCompiler { format, profile }))
}
