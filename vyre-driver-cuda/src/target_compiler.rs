use vyre_driver::{BackendError, DispatchConfig};
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat,
};

use crate::CUDA_BACKEND_ID;

const CUDA_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct CudaTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for CudaTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(artifact, self.format.clone(), |program| {
            crate::codegen::program_to_ptx(program, &DispatchConfig::default())
                .map(|source| EmittedTargetModule {
                    entry_point: "main".to_string(),
                    bytes: source.into_bytes(),
                })
                .map_err(TargetCompileError::Emission)
        })
    }
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new("ptx", CUDA_TARGET_FORMAT_VERSION).map_err(|error| {
        BackendError::KernelCompileFailed {
            backend: CUDA_BACKEND_ID.to_string(),
            compiler_message: format!(
                "PTX target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        }
    })?;
    Ok(Box::new(CudaTargetCompiler { format }))
}
