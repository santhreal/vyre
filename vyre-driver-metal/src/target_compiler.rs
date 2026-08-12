use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::METAL_BACKEND_ID;

pub(crate) const METAL_TARGET_FORMAT: &str = "msl";
pub(crate) const METAL_TARGET_FORMAT_VERSION: u16 = 2;

pub(crate) struct MetalTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for MetalTargetCompiler {
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
                let artifact =
                    vyre_emit_metal::emit_artifact(&selected.descriptor).map_err(|error| {
                        TargetCompileError::Emission(format!("Metal emission failed: {error}"))
                    })?;
                let entry_point = artifact.entry_point.clone();
                let grid_size = vyre_driver::infer_dispatch_grid_for_count(
                    selected.logical_element_count,
                    selected.descriptor.dispatch.workgroup_size,
                )
                .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
                serde_json::to_vec(&artifact)
                    .map(|bytes| EmittedTargetModule {
                        entry_point,
                        grid_size,
                        dynamic_shared_bytes: 0,
                        workgroup_size: selected.descriptor.dispatch.workgroup_size,
                        resource_bindings: selected.canonical_bindings.clone(),
                        bytes,
                    })
                    .map_err(|error| {
                        TargetCompileError::Emission(format!(
                            "Metal target artifact serialization failed: {error}"
                        ))
                    })
            },
        )
    }
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    TargetProfile::new(
        METAL_TARGET_FORMAT,
        u64::from(METAL_TARGET_FORMAT_VERSION),
        [1_024, 1_024, 64],
        1_024,
        32_768,
        0,
    )
    .map_err(|error| BackendError::KernelCompileFailed {
        backend: METAL_BACKEND_ID.to_string(),
        compiler_message: format!(
            "Metal target profile is invalid: {error}. Fix: repair the registered profile."
        ),
    })
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format =
        TargetPayloadFormat::new(METAL_TARGET_FORMAT, METAL_TARGET_FORMAT_VERSION).map_err(|error| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal target format is invalid: {error}. Fix: repair the registered format identity."
                ),
            }
        })?;
    let profile = target_profile()?;
    Ok(Box::new(MetalTargetCompiler { format, profile }))
}
