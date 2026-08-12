use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::CUDA_BACKEND_ID;

pub(crate) const CUDA_TARGET_FORMAT: &str = "ptx";
const CUDA_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct CudaTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for CudaTargetCompiler {
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
            |selected, profile| {
                let source = vyre_emit_ptx::emit_with_options(
                    &selected.descriptor,
                    vyre_emit_ptx::PtxEmitOptions {
                        target: vyre_emit_ptx::ComputeCapability {
                            major: profile.generation() as u32 / 10,
                            minor: profile.generation() as u32 % 10,
                        },
                        subgroup_size: profile.subgroup_size().max(1),
                        ulp_budget: None,
                        cooperative_grid_sync: true,
                    },
                )
                .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
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
                    bytes: source.into_bytes(),
                })
            },
        )
    }
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    TargetProfile::new(
        CUDA_TARGET_FORMAT,
        80,
        [1_024, 1_024, 64],
        1_024,
        49_152,
        32,
    )
    .map_err(|error| BackendError::KernelCompileFailed {
        backend: CUDA_BACKEND_ID.to_string(),
        compiler_message: format!(
            "PTX target profile is invalid: {error}. Fix: repair the registered profile."
        ),
    })
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new(CUDA_TARGET_FORMAT, CUDA_TARGET_FORMAT_VERSION).map_err(
        |error| BackendError::KernelCompileFailed {
            backend: CUDA_BACKEND_ID.to_string(),
            compiler_message: format!(
                "PTX target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        },
    )?;
    let profile = target_profile()?;
    Ok(Box::new(CudaTargetCompiler { format, profile }))
}
