//! Artifact materializer for the reference oracle.
//!
//! The oracle is a backend, so it admits an authenticated artifact and
//! submits bindings against it through the same lifecycle every device driver
//! uses. Without one, `ArtifactSession::compile` refused `cpu-ref` by name and
//! every conformance case that routes through canonical artifact submission
//! failed on the reference backend before it ever reached the interpreter.

use std::sync::Arc;
use std::time::Instant;

use vyre_driver::materialize::{
    self, DeviceSpec, ExecutableModule, InstanceCore, MaterializedInstance, MaterializerDevice,
};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, DispatchConfig, Submission,
    TimedDispatchResult,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, TargetPayload};

use crate::{interpret, program_dispatch, CPU_REF_BACKEND_ID};

/// Leading bytes every reference target module carries.
const REFERENCE_MODULE_TAG: &[u8] = b"vyre-reference-graph-v1\0";

struct ReferenceMaterializer {
    descriptor: MaterializerDevice,
}

impl ArtifactMaterializer for ReferenceMaterializer {
    vyre_driver::materializer_passthrough!();

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        let modules =
            self.descriptor
                .admit_modules(CPU_REF_BACKEND_ID, artifact, payload, |admitted| {
                    if !admitted.image.bytes.starts_with(REFERENCE_MODULE_TAG) {
                        return Err(materialize::invalid_module(
                            "reference target module must begin with the reference graph tag",
                        ));
                    }
                    Ok(ReferenceExecutableModule {
                        program: admitted.program,
                        config: admitted.config,
                    })
                })?;
        Ok(Box::new(ReferenceArtifactInstance {
            core: self
                .descriptor
                .instance(artifact, payload, materialize::NEUTRAL_MESSAGES)?,
            modules,
        }))
    }
}

struct ReferenceExecutableModule {
    program: Arc<Program>,
    config: DispatchConfig,
}

struct ReferenceArtifactInstance {
    core: InstanceCore,
    modules: Vec<ReferenceExecutableModule>,
}

impl ExecutableModule for ReferenceExecutableModule {
    vyre_driver::executable_module!();
}

impl ArtifactInstance for ReferenceArtifactInstance {
    vyre_driver::artifact_instance_identity!();
    vyre_driver::artifact_instance_unreported_resources!();

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        self.submit_host_only(&bindings, "reference artifact resident binding")
    }
}

impl MaterializedInstance for ReferenceArtifactInstance {
    type Module = ReferenceExecutableModule;

    fn core(&self) -> &InstanceCore {
        &self.core
    }

    fn modules(&self) -> &[Self::Module] {
        &self.modules
    }

    fn module_label(&self) -> &'static str {
        "reference target module"
    }

    fn dispatch(
        &self,
        module: &Self::Module,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let started = Instant::now();
        let outputs = interpret(&module.program, inputs, config)?;
        Ok(TimedDispatchResult::host_timed(
            outputs,
            u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
                BackendError::DispatchFailed {
                    code: None,
                    message: "reference dispatch duration overflowed a 64-bit nanosecond count"
                        .to_string(),
                }
            })?,
        ))
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    Ok(Box::new(ReferenceMaterializer {
        descriptor: MaterializerDevice::acquire(DeviceSpec {
            backend: CPU_REF_BACKEND_ID,
            device: "reference-interpreter".to_string(),
            format_extension: program_dispatch::REFERENCE_TARGET_FORMAT,
            format_version: program_dispatch::REFERENCE_TARGET_FORMAT_VERSION,
            profile: program_dispatch::target_profile()?,
        })?,
    }))
}
