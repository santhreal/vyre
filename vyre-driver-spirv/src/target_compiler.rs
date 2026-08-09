use vyre_driver::BackendError;
use vyre_megakernel::{
    fuse_selected_module, selected_modules, AbiAccess, Artifact, TargetCompileError,
    TargetCompiler, TargetEntryPoint, TargetModuleBundle, TargetModuleImage, TargetPayload,
    TargetPayloadFormat, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};

use crate::{backend::SpirvBackend, SPIRV_BACKEND_ID};

const SPIRV_TARGET_FORMAT_VERSION: u16 = 1;

pub(crate) struct SpirvTargetCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for SpirvTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        let modules = selected_modules(artifact)?;
        let bindings = resource_bindings(artifact);
        let mut images = Vec::with_capacity(modules.len());
        let mut entries = Vec::with_capacity(modules.len());
        for module in modules {
            let program = fuse_selected_module(&module)?;
            let words =
                SpirvBackend::program_to_spv(&program).map_err(TargetCompileError::Emission)?;
            let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
            for word in words {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            let entry_point = format!("vyre_group_{}", module.group.0);
            let node = *module.nodes.first().ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "fusion group {} has no member node",
                    module.group.0
                ))
            })?;
            entries.push(TargetEntryPoint {
                name: entry_point.clone(),
                node,
                grid_size: dispatch_grid(artifact, program.workgroup_size),
                dynamic_shared_bytes: 0,
                resource_bindings: bindings.clone(),
            });
            images.push(TargetModuleImage {
                group: module.group,
                stage: module.stage,
                entry_point,
                bytes,
            });
        }
        let bytes = TargetModuleBundle::new(images).to_bytes()?;
        TargetPayload::new(artifact, self.format.clone(), entries, bytes).map_err(Into::into)
    }
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    let format = TargetPayloadFormat::new("spv", SPIRV_TARGET_FORMAT_VERSION).map_err(|error| {
        BackendError::KernelCompileFailed {
            backend: SPIRV_BACKEND_ID.to_string(),
            compiler_message: format!(
                "SPIR-V target format is invalid: {error}. Fix: repair the registered format identity."
            ),
        }
    })?;
    Ok(Box::new(SpirvTargetCompiler { format }))
}

fn resource_bindings(artifact: &Artifact) -> Vec<TargetResourceBinding> {
    artifact
        .abi()
        .resources
        .iter()
        .map(|resource| TargetResourceBinding {
            resource: resource.value,
            slot: resource.slot,
            memory: TargetResourceMemory::Global,
            access: match resource.access {
                AbiAccess::ReadOnly => TargetResourceAccess::ReadOnly,
                AbiAccess::WriteOnly => TargetResourceAccess::WriteOnly,
                AbiAccess::ReadWrite => TargetResourceAccess::ReadWrite,
                AbiAccess::Uniform => TargetResourceAccess::ReadOnly,
            },
        })
        .collect()
}

fn dispatch_grid(artifact: &Artifact, workgroup: [u32; 3]) -> [u32; 3] {
    let elements = artifact
        .resources()
        .iter()
        .map(|resource| resource.element_count)
        .max()
        .unwrap_or(1)
        .max(1);
    let threads = u64::from(workgroup[0])
        .saturating_mul(u64::from(workgroup[1]))
        .saturating_mul(u64::from(workgroup[2]))
        .max(1);
    let groups = elements.saturating_add(threads - 1) / threads;
    [u32::try_from(groups).unwrap_or(u32::MAX), 1, 1]
}

const _: &str = SPIRV_BACKEND_ID;
