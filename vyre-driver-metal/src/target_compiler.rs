use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, TargetCompileError, TargetCompiler, TargetPayload,
    TargetPayloadFormat,
};

use crate::METAL_BACKEND_ID;

const METAL_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct MetalTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for MetalTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(artifact, self.format.clone(), |program| {
            let lowered = vyre_lower::lower_verified(program).map_err(|error| {
                TargetCompileError::Emission(format!(
                    "verified lowering failed before Metal emission: {error}"
                ))
            })?;
            vyre_emit_metal::emit(&lowered.descriptor)
                .map(String::into_bytes)
                .map_err(|error| {
                    TargetCompileError::Emission(format!("Metal emission failed: {error}"))
                })
        })
    }
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format =
        TargetPayloadFormat::new("msl", METAL_TARGET_FORMAT_VERSION).map_err(|error| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal target format is invalid: {error}. Fix: repair the registered format identity."
                ),
            }
        })?;
    Ok(Box::new(MetalTargetCompiler { format }))
}
