use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_driver::BackendError;
use vyre_megakernel::{TargetCompileError, TargetCompiler, TargetProfile};

use crate::METAL_BACKEND_ID;

pub(crate) const METAL_TARGET_FORMAT: &str = "msl";
pub(crate) const METAL_TARGET_FORMAT_VERSION: u16 = 2;

const METAL_DIALECT: TargetDialect = TargetDialect {
    backend_id: METAL_BACKEND_ID,
    dialect: "Metal",
    format: METAL_TARGET_FORMAT,
    format_version: METAL_TARGET_FORMAT_VERSION,
    generation: METAL_TARGET_FORMAT_VERSION as u64,
    max_workgroup_size: [1_024, 1_024, 64],
    max_invocations_per_workgroup: 1_024,
    max_dynamic_shared_bytes: 32_768,
    subgroup_size: 0,
    emit: emit_metal_module,
};

fn emit_metal_module(
    selected: &vyre_megakernel::SelectedLowering,
    _profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    let artifact = vyre_emit_metal::emit_artifact(&selected.descriptor)
        .map_err(|error| TargetCompileError::Emission(format!("Metal emission failed: {error}")))?;
    let entry_point = artifact.entry_point.clone();
    let bytes = serde_json::to_vec(&artifact).map_err(|error| {
        TargetCompileError::Emission(format!(
            "Metal target artifact serialization failed: {error}"
        ))
    })?;
    Ok(EmittedDialectModule {
        entry_point,
        bytes,
        dynamic_shared_bytes: 0,
    })
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    METAL_DIALECT.profile()
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    METAL_DIALECT.compiler()
}
