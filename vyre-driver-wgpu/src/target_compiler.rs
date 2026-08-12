use vyre_driver::BackendError;
use vyre_megakernel::{
    compile_selected_modules, Artifact, EmittedTargetModule, TargetCompileError, TargetCompiler,
    TargetPayload, TargetPayloadFormat, TargetProfile,
};

pub(crate) const WGPU_TARGET_FORMAT: &str = "wgsl";
pub(crate) const WGPU_TARGET_FORMAT_VERSION: u16 = 2;

pub(crate) const WGPU_TARGET_MODULE_SCHEMA_VERSION: u16 = 2;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct WgpuTargetModule {
    pub(crate) schema_version: u16,
    pub(crate) wgsl: String,
}

pub(crate) struct WgpuTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for WgpuTargetCompiler {
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
            |selected, _profile| {
                let descriptor = selected.descriptor.clone();
                let module =
                    crate::emit::emit_naga_module_for_descriptor(&descriptor).map_err(|error| {
                        TargetCompileError::Emission(format!("WGSL emission failed: {error}"))
                    })?;
                let wgsl = crate::emit::write_wgsl(&module).map_err(|error| {
                    TargetCompileError::Emission(format!("WGSL writing failed: {error}"))
                })?;
                let target = WgpuTargetModule {
                    schema_version: WGPU_TARGET_MODULE_SCHEMA_VERSION,
                    wgsl,
                };
                let grid_size = vyre_driver::infer_dispatch_grid_for_count(
                    selected.logical_element_count,
                    selected.descriptor.dispatch.workgroup_size,
                )
                .map_err(|error| TargetCompileError::Emission(error.to_string()))?;
                serde_json::to_vec(&target)
                    .map(|bytes| EmittedTargetModule {
                        entry_point: "main".to_string(),
                        grid_size,
                        dynamic_shared_bytes: 0,
                        workgroup_size: selected.descriptor.dispatch.workgroup_size,
                        resource_bindings: selected.canonical_bindings.clone(),
                        bytes,
                    })
                    .map_err(|error| {
                        TargetCompileError::Emission(format!(
                            "WGSL target module serialization failed: {error}"
                        ))
                    })
            },
        )
    }
}

pub(crate) fn target_profile() -> Result<TargetProfile, BackendError> {
    TargetProfile::new(
        WGPU_TARGET_FORMAT,
        u64::from(WGPU_TARGET_FORMAT_VERSION),
        [256, 256, 64],
        256,
        16_384,
        0,
    )
    .map_err(|error| BackendError::KernelCompileFailed {
        backend: "wgpu".to_string(),
        compiler_message: format!(
            "WGSL target profile is invalid: {error}. Fix: repair the registered profile."
        ),
    })
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
    let profile = target_profile()?;
    Ok(Box::new(WgpuTargetCompiler { format, profile }))
}
