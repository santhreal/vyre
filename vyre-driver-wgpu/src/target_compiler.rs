use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, TargetCompileError, TargetCompiler, TargetPayload,
    TargetPayloadFormat,
};

const WGPU_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct WgpuTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for WgpuTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(artifact, self.format.clone(), |program| {
            crate::emit::lower(program)
                .map(String::into_bytes)
                .map_err(|error| {
                    TargetCompileError::Emission(format!("WGSL emission failed: {error}"))
                })
        })
    }
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new("wgsl", WGPU_TARGET_FORMAT_VERSION).map_err(|error| {
        BackendError::KernelCompileFailed {
            backend: "wgpu".to_string(),
            compiler_message: format!(
                "WGSL target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        }
    })?;
    Ok(Box::new(WgpuTargetCompiler { format }))
}
