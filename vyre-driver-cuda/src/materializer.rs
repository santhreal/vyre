use std::collections::BTreeMap;
use std::sync::Arc;

use vyre_driver::materialize::{self, InstanceCore, InstanceMessages, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingRole, BindingSet,
    CompiledPipeline, Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner,
    Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, Digest, TargetPayload, TargetPayloadFormat};

use crate::backend::CudaBackend;
use crate::pipeline::CudaCompiledPipeline;
use crate::{CudaBackendRegistration, CUDA_BACKEND_ID};

/// CUDA rejection text. Two strings differ from the neutral wording: this
/// backend names the digest in a foreign-artifact rejection, and reports an
/// unpreserved retained value as an unproduced output.
const MESSAGES: InstanceMessages = InstanceMessages {
    foreign_artifact: || BackendError::InvalidProgram {
        fix: "Fix: bind resources against the exact artifact digest owned by this instance."
            .to_string(),
    },
    missing_retained_value: |value| {
        materialize::invalid_module(&format!(
            "selected execution did not produce canonical output value {}",
            value.0
        ))
    },
    ..materialize::NEUTRAL_MESSAGES
};

/// Rejection for a declared input whose canonical value was never bound.
fn unbound_input(value: ArtifactValueId, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: bind canonical artifact value {} for Program buffer `{name}` before submission.",
            value.0
        ),
    }
}

/// Rejection for a host dispatch that skipped a declared output slot.
fn omitted_output(output_index: usize, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: CUDA target module omitted output {output_index} for Program buffer `{name}`."
        ),
    }
}

/// Rejection for a resident dispatch that skipped a declared output slot.
fn omitted_resident_output(output_index: usize, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: CUDA resident target module omitted output {output_index} for Program buffer `{name}`."
        ),
    }
}

pub(crate) struct CudaMaterializer {
    backend: CudaBackend,
    resident: CudaBackendRegistration,
    descriptor: MaterializerDevice,
}

impl ArtifactMaterializer for CudaMaterializer {
    fn device(&self) -> &dyn Device {
        &self.descriptor
    }
    fn allocate_resident(&self, byte_len: usize) -> Result<vyre_driver::Resource, BackendError> {
        vyre_driver::VyreBackend::allocate_resident(&self.resident, byte_len)
    }

    fn upload_resident(
        &self,
        resource: &vyre_driver::Resource,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::upload_resident(&self.resident, resource, bytes)
    }

    fn upload_resident_at(
        &self,
        resource: &vyre_driver::Resource,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::upload_resident_at(&self.resident, resource, offset_bytes, bytes)
    }

    fn free_resident(&self, resource: vyre_driver::Resource) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::free_resident(&self.resident, resource)
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        let admitted =
            materialize::admit(artifact, payload, self.descriptor.target(CUDA_BACKEND_ID))?;
        let mut modules = Vec::with_capacity(admitted.len());
        for module in admitted {
            let ptx = std::str::from_utf8(&module.image.bytes).map_err(|error| {
                materialize::invalid_module(&format!("PTX target module is not UTF-8: {error}"))
            })?;
            if !ptx.contains(".visible .entry main(") {
                return Err(materialize::invalid_module(
                    "PTX target module does not define `.visible .entry main`",
                ));
            }
            let prepared = self
                .backend
                .prepare_static_dispatch(&module.program, &module.config)?;
            let module_key = self.backend.module_cache_key_for_raw_ptx_artifact(ptx)?;
            self.backend.module_for_ptx_with_key(ptx, module_key)?;
            let ptx: Arc<str> = Arc::from(ptx);
            let pipeline = Arc::new(CudaCompiledPipeline::new_from_target_payload(
                self.backend.clone(),
                Arc::clone(&module.program),
                ptx,
                module_key,
                &module.config,
                prepared,
            )?);
            modules.push(CudaExecutableModule {
                program: module.program,
                pipeline,
                config: module.config,
            });
        }
        Ok(Box::new(CudaArtifactInstance {
            core: self.descriptor.instance(artifact, payload, MESSAGES),
            modules,
        }))
    }
}

struct CudaExecutableModule {
    program: Arc<Program>,
    pipeline: Arc<CudaCompiledPipeline>,
    config: DispatchConfig,
}

struct CudaArtifactInstance {
    core: InstanceCore,
    modules: Vec<CudaExecutableModule>,
}

impl ArtifactInstance for CudaArtifactInstance {
    fn artifact(&self) -> Digest {
        self.core.artifact
    }

    fn payload(&self) -> Digest {
        self.core.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.core.device
    }

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        self.core.accept(&bindings)?;
        let invocation_grid = bindings.invocation_grid();
        let bound = materialize::partition_bindings(&bindings);
        if !bound.host.is_empty() && !bound.resident.is_empty() {
            return Err(materialize::invalid_module(
                "CUDA artifact submission cannot mix host and resident resources",
            ));
        }
        let result = if bound.resident.is_empty() {
            self.execute(bound.host, invocation_grid)
        } else {
            self.execute_resident(&bound.resident, invocation_grid)
        };
        Ok(self.core.ready(result))
    }
}

impl CudaArtifactInstance {
    fn execute(
        &self,
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for module in &self.modules {
            let mut config = module.config.clone();
            materialize::override_grid(&mut config, invocation_grid);
            let plan = BindingPlan::build(&module.program)?;
            let inputs = self
                .core
                .gather_inputs(&plan, &module.program, &state, unbound_input)?;
            let dispatched = module.pipeline.dispatch_borrowed_timed(&inputs, &config)?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            self.core.absorb_outputs(
                &plan,
                &module.program,
                &dispatched.outputs,
                &mut state,
                omitted_output,
            )?;
        }
        self.core
            .completion(&state, has_device_timing.then_some(device_ns))
    }

    fn execute_resident(
        &self,
        resources: &BTreeMap<ArtifactValueId, vyre_driver::Resource>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        if self.modules.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "CUDA resident submission for multi-module artifacts".to_string(),
                backend: CUDA_BACKEND_ID.to_string(),
            });
        }
        let module = &self.modules[0];
        let plan = BindingPlan::build(&module.program)?;
        let mut ordered = Vec::with_capacity(plan.bindings.len());
        for binding in resident_resource_bindings(&plan) {
            let buffer = &module.program.buffers()[binding.buffer_index];
            let value = self.core.value_for_buffer(buffer.name())?;
            let resource = resources
                .get(&value)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: bind canonical artifact value {} for resident Program buffer `{}`.",
                        value.0,
                        buffer.name()
                    ),
                })?;
            ordered.push(resource.clone());
        }
        let dispatched = module
            .pipeline
            .dispatch_artifact_resident_timed(&ordered, invocation_grid)?;
        let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        self.core.absorb_outputs(
            &plan,
            &module.program,
            &dispatched.outputs,
            &mut state,
            omitted_resident_output,
        )?;
        self.core.completion(&state, dispatched.device_ns)
    }
}

fn resident_resource_bindings(plan: &BindingPlan) -> impl Iterator<Item = &vyre_driver::Binding> {
    plan.bindings
        .iter()
        .filter(|binding| binding.role != BindingRole::Shared)
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let backend = CudaBackend::acquire().map_err(|message| BackendError::DispatchFailed {
        code: None,
        message: format!("CUDA artifact device acquisition failed: {message}"),
    })?;
    let format = TargetPayloadFormat::new("ptx", 1)
        .map_err(|error| materialize::compile_error(CUDA_BACKEND_ID, error))?;
    let profile = crate::target_compiler::target_profile()?;
    let generation = ResidentOwner::new()?.get();
    let device = backend.caps.name.clone();
    Ok(Box::new(CudaMaterializer {
        resident: CudaBackendRegistration {
            inner: backend.clone(),
        },
        backend,
        descriptor: MaterializerDevice::new(
            DeviceIdentity {
                backend: CUDA_BACKEND_ID,
                device,
                generation,
            },
            format,
            profile,
        ),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType};

    /// WHY: workgroup scratch is module-internal memory, not an artifact value. Resident
    /// materialization must bind every host-visible role while excluding shared scratch.
    #[test]
    fn resident_resource_projection_excludes_workgroup_scratch() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32)
                    .with_count(16),
                BufferDecl::workgroup("scratch", 16, DataType::F32),
                BufferDecl::output("output", 1, DataType::F32).with_count(16),
            ],
            [16, 1, 1],
            Vec::new(),
        );
        let plan = BindingPlan::build(&program)
            .expect("Fix: resident resource projection fixture must build a binding plan.");
        let names = resident_resource_bindings(&plan)
            .map(|binding| binding.name.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(names, ["input", "output"]);
    }
}
