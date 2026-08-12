use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat,
};

use crate::{backend::SpirvBackend, SPIRV_BACKEND_ID};

pub(crate) const SPIRV_TARGET_FORMAT: &str = "spv";
const SPIRV_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct SpirvTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for SpirvTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(artifact, self.format.clone(), |program| {
            let words =
                SpirvBackend::program_to_spv(program).map_err(TargetCompileError::Emission)?;
            let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
            for word in words {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            Ok(EmittedTargetModule {
                entry_point: "main".to_string(),
                bytes,
            })
        })
    }
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
    Ok(Box::new(SpirvTargetCompiler { format }))
}
