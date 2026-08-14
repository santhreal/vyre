use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_driver::BackendError;
use vyre_megakernel::{SelectedLowering, TargetCompileError, TargetCompiler, TargetProfile};

use crate::WGPU_BACKEND_ID;

pub(crate) const WGPU_TARGET_FORMAT: &str = "wgsl";
pub(crate) const WGPU_TARGET_FORMAT_VERSION: u16 = 2;

pub(crate) const WGPU_TARGET_MODULE_SCHEMA_VERSION: u16 = 2;

const WGPU_DIALECT: TargetDialect = TargetDialect {
    backend_id: WGPU_BACKEND_ID,
    dialect: "WGSL",
    format: WGPU_TARGET_FORMAT,
    format_version: WGPU_TARGET_FORMAT_VERSION,
    generation: WGPU_TARGET_FORMAT_VERSION as u64,
    max_workgroup_size: [256, 256, 64],
    max_invocations_per_workgroup: 256,
    max_dynamic_shared_bytes: 16_384,
    subgroup_size: 0,
    emit: emit_wgsl_module,
};

/// The payload is the WGSL text behind a schema version, because the
/// materializer accepts source it must hand to the shader compiler, not an
/// image it can submit directly.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct WgpuTargetModule {
    pub(crate) schema_version: u16,
    pub(crate) wgsl: String,
}

fn emit_wgsl_module(
    selected: &SelectedLowering,
    _profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    let module = crate::emit::emit_naga_module_for_descriptor(&selected.descriptor)
        .map_err(|error| TargetCompileError::Emission(format!("WGSL emission failed: {error}")))?;
    let wgsl = crate::emit::write_wgsl(&module)
        .map_err(|error| TargetCompileError::Emission(format!("WGSL writing failed: {error}")))?;
    let bytes = serde_json::to_vec(&WgpuTargetModule {
        schema_version: WGPU_TARGET_MODULE_SCHEMA_VERSION,
        wgsl,
    })
    .map_err(|error| {
        TargetCompileError::Emission(format!("WGSL target module serialization failed: {error}"))
    })?;
    Ok(EmittedDialectModule {
        entry_point: "main".to_string(),
        bytes,
        dynamic_shared_bytes: 0,
    })
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    WGPU_DIALECT.profile()
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    WGPU_DIALECT.compiler()
}
