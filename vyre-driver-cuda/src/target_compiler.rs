use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_driver::BackendError;
use vyre_megakernel::{TargetCompileError, TargetCompiler, TargetProfile};

use crate::CUDA_BACKEND_ID;

pub(crate) const CUDA_TARGET_FORMAT: &str = "ptx";
const CUDA_TARGET_FORMAT_VERSION: u16 = 1;

const CUDA_DIALECT: TargetDialect = TargetDialect {
    backend_id: CUDA_BACKEND_ID,
    dialect: "PTX",
    format: CUDA_TARGET_FORMAT,
    format_version: CUDA_TARGET_FORMAT_VERSION,
    generation: 80,
    max_workgroup_size: [1_024, 1_024, 64],
    max_invocations_per_workgroup: 1_024,
    max_dynamic_shared_bytes: 49_152,
    subgroup_size: 32,
    emit: emit_ptx_module,
};

fn emit_ptx_module(
    selected: &vyre_megakernel::SelectedLowering,
    profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    let source = vyre_emit_ptx::emit_with_options(
        selected.descriptor(),
        vyre_emit_ptx::PtxEmitOptions {
            target: vyre_emit_ptx::ComputeCapability {
                major: profile.generation() as u32 / 10,
                minor: profile.generation() as u32 % 10,
            },
            subgroup_size: profile.subgroup_size().max(1),
            // No caller setting reaches this route, so the exact form of
            // `InverseSqrt` and `Reciprocal` is the one to prefer. The ops PTX
            // can only approximate no longer consult this field.
            ulp_budget: None,
            cooperative_grid_sync: true,
        },
    )
    .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
    Ok(EmittedDialectModule {
        entry_point: "main".to_string(),
        bytes: source.into_bytes(),
        dynamic_shared_bytes: 0,
    })
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    CUDA_DIALECT.profile()
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    CUDA_DIALECT.compiler()
}
