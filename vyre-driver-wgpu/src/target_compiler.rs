use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat,
};

pub(crate) const WGPU_TARGET_FORMAT: &str = "wgsl";
pub(crate) const WGPU_TARGET_FORMAT_VERSION: u16 = 2;

pub(crate) const WGPU_TARGET_MODULE_SCHEMA_VERSION: u16 = 1;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct WgpuTargetModule {
    pub(crate) schema_version: u16,
    pub(crate) descriptor: vyre_lower::KernelDescriptor,
    pub(crate) wgsl: String,
}

pub(crate) struct WgpuTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for WgpuTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(artifact, self.format.clone(), |program| {
            let mut descriptor = vyre_lower::lower_verified(program)
                .map_err(|error| {
                    TargetCompileError::Emission(format!(
                        "verified lowering failed before WGSL emission: {error}"
                    ))
                })?
                .descriptor;
            descriptor.dispatch.workgroup_size = crate::emit::optimal_workgroup_size(
                program,
                &crate::runtime::device::EnabledFeatures::default(),
            );
            let module =
                crate::emit::emit_naga_module_for_descriptor(&descriptor).map_err(|error| {
                    TargetCompileError::Emission(format!("WGSL emission failed: {error}"))
                })?;
            let wgsl = crate::emit::write_wgsl(&module).map_err(|error| {
                TargetCompileError::Emission(format!("WGSL writing failed: {error}"))
            })?;
            let target = WgpuTargetModule {
                schema_version: WGPU_TARGET_MODULE_SCHEMA_VERSION,
                descriptor,
                wgsl,
            };
            serde_json::to_vec(&target)
                .map(|bytes| EmittedTargetModule {
                    entry_point: "main".to_string(),
                    bytes,
                })
                .map_err(|error| {
                    TargetCompileError::Emission(format!(
                        "WGSL target module serialization failed: {error}"
                    ))
                })
        })
    }
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new(WGPU_TARGET_FORMAT, WGPU_TARGET_FORMAT_VERSION).map_err(|error| {
        BackendError::KernelCompileFailed {
            backend: "wgpu".to_string(),
            compiler_message: format!(
                "WGSL target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        }
    })?;
    Ok(Box::new(WgpuTargetCompiler { format }))
}
