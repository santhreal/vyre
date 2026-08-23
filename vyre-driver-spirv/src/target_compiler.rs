use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_driver::BackendError;
use vyre_megakernel::{TargetCompileError, TargetCompiler, TargetProfile};

use crate::SPIRV_BACKEND_ID;

pub(crate) const SPIRV_TARGET_FORMAT: &str = "spv";
const SPIRV_TARGET_FORMAT_VERSION: u16 = 1;

const SPIRV_DIALECT: TargetDialect = TargetDialect {
    backend_id: SPIRV_BACKEND_ID,
    dialect: "SPIR-V",
    format: SPIRV_TARGET_FORMAT,
    format_version: SPIRV_TARGET_FORMAT_VERSION,
    generation: SPIRV_TARGET_FORMAT_VERSION as u64,
    max_workgroup_size: [1_024, 1_024, 64],
    max_invocations_per_workgroup: 1_024,
    max_dynamic_shared_bytes: 32_768,
    subgroup_size: 0,
    emit: emit_spirv_module,
};

fn emit_spirv_module(
    selected: &vyre_megakernel::SelectedLowering,
    _profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    let words = vyre_emit_spirv::emit(selected.descriptor())
        .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
    let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    Ok(EmittedDialectModule {
        entry_point: "main".to_string(),
        bytes,
        dynamic_shared_bytes: 0,
    })
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    SPIRV_DIALECT.profile()
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    SPIRV_DIALECT.compiler()
}
