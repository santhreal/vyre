use std::collections::BTreeMap;
use std::sync::Arc;

use vyre_driver::materialize::{
    self, ExecutableModule, InstanceCore, InstanceMessages, MaterializerDevice,
};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet,
    CompiledPipeline, Completion, DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, TargetPayload, TargetPayloadFormat};

use crate::backend::CudaBackend;
use crate::pipeline::CudaCompiledPipeline;
use crate::{CudaBackendRegistration, CUDA_BACKEND_ID};

/// CUDA rejection text. One string differs from the neutral wording: this
/// backend reports an unpreserved retained value as an unproduced output.
const MESSAGES: InstanceMessages = InstanceMessages {
    missing_retained_value: |value| {
        materialize::invalid_module(&format!(
            "selected execution did not produce canonical output value {}",
            value.0
        ))
    },
    ..materialize::NEUTRAL_MESSAGES
};

/// Rejection for a host dispatch that skipped a declared output slot.
fn omitted_output(output_index: usize, name: &str) -> BackendError {
    materialize::omitted_output("CUDA target module", output_index, name)
}

/// Rejection for a resident dispatch that skipped a declared output slot.
fn omitted_resident_output(output_index: usize, name: &str) -> BackendError {
    materialize::omitted_output("CUDA resident target module", output_index, name)
}

pub(crate) struct CudaMaterializer {
    backend: CudaBackend,
    resident: CudaBackendRegistration,
    descriptor: MaterializerDevice,
}

impl ArtifactMaterializer for CudaMaterializer {
    vyre_driver::materializer_passthrough!(resident);

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        let modules =
            self.descriptor
                .admit_modules(CUDA_BACKEND_ID, artifact, payload, |module| {
                    let ptx = std::str::from_utf8(&module.image.bytes).map_err(|error| {
                        materialize::invalid_module(&format!(
                            "PTX target module is not UTF-8: {error}"
                        ))
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
                    Ok(CudaExecutableModule {
                        program: module.program,
                        pipeline,
                        config: module.config,
                    })
                })?;
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

impl ExecutableModule for CudaExecutableModule {
    vyre_driver::executable_module!();
}

impl ArtifactInstance for CudaArtifactInstance {
    vyre_driver::artifact_instance_identity!();

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        self.core.route_submission(
            &bindings,
            || {
                materialize::invalid_module(
                    "CUDA artifact submission cannot mix host and resident resources",
                )
            },
            |state, invocation_grid| self.execute(state, invocation_grid),
            |resources, invocation_grid| self.execute_resident(resources, invocation_grid),
        )
    }
}

impl CudaArtifactInstance {
    fn execute(
        &self,
        state: BTreeMap<ArtifactValueId, Vec<u8>>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        self.core.execute_modules(
            &self.modules,
            state,
            invocation_grid,
            omitted_output,
            |module, plan, config, state| {
                let inputs = self.core.gather_inputs(
                    plan,
                    &module.program,
                    state,
                    materialize::unbound_input,
                )?;
                module.pipeline.dispatch_borrowed_timed(&inputs, config)
            },
        )
    }

    fn execute_resident(
        &self,
        resources: &BTreeMap<ArtifactValueId, vyre_driver::Resource>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        let module = self.core.single_resident_module(
            &self.modules,
            "CUDA resident submission for multi-module artifacts",
        )?;
        let plan = BindingPlan::build(&module.program)?;
        let ordered = self.core.ordered_resident_resources(
            materialize::resident_buffer_names(&plan, &module.program),
            resources,
            materialize::unbound_resident_buffer,
        )?;
        let dispatched = module
            .pipeline
            .dispatch_artifact_resident_timed(&ordered, invocation_grid)?;
        self.core.resident_completion(
            &plan,
            &module.program,
            dispatched,
            omitted_resident_output,
            &self.core.messages,
        )
    }
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
