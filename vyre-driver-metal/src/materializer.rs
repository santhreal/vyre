use vyre_driver::{ArtifactMaterializer, BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod native {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use vyre_driver::materialize::{self, ExecutableModule, InstanceCore, MaterializerDevice};
    use vyre_driver::{
        ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, Completion,
        DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
    };
    use vyre_foundation::ir::Program;
    use vyre_megakernel::{Artifact, ArtifactValueId, TargetPayload, TargetPayloadFormat};

    use crate::runtime::{MetalBackend, MetalTargetModule};
    use crate::target_compiler::METAL_TARGET_FORMAT_VERSION;
    use crate::METAL_BACKEND_ID;

    /// Rejection for a dispatch that skipped a declared output slot.
    fn omitted_output(output_index: usize, name: &str) -> BackendError {
        materialize::omitted_output("Metal target module", output_index, name)
    }

    pub(super) struct MetalMaterializer {
        backend: Arc<MetalBackend>,
        descriptor: MaterializerDevice,
    }

    impl ArtifactMaterializer for MetalMaterializer {
        vyre_driver::materializer_passthrough!();

        fn materialize(
            &self,
            artifact: &Artifact,
            payload: &TargetPayload,
        ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
            let modules = self.descriptor.admit_modules(
                METAL_BACKEND_ID,
                artifact,
                payload,
                |admitted_module| {
                    let target: vyre_emit_metal::MetalArtifact =
                        serde_json::from_slice(&admitted_module.image.bytes).map_err(|error| {
                            materialize::invalid_module(&format!(
                                "Metal target artifact is malformed: {error}"
                            ))
                        })?;
                    if admitted_module.image.entry_point != target.entry_point {
                        return Err(materialize::invalid_module(
                            "module bundle and Metal artifact entry points disagree",
                        ));
                    }
                    let program = admitted_module.program;
                    let config = admitted_module.config;
                    if target.workgroup_size != program.workgroup_size {
                        return Err(materialize::invalid_module(
                            "Metal artifact workgroup geometry disagrees with the selected Program",
                        ));
                    }
                    let module = self.backend.materialize_target_module(target)?;
                    Ok(MetalExecutableModule {
                        program,
                        module,
                        config,
                    })
                },
            )?;
            Ok(Box::new(MetalArtifactInstance {
                core: self
                    .descriptor
                    .instance(artifact, payload, materialize::NEUTRAL_MESSAGES),
                backend: Arc::clone(&self.backend),
                modules,
            }))
        }
    }

    struct MetalExecutableModule {
        program: Arc<Program>,
        module: MetalTargetModule,
        config: DispatchConfig,
    }

    struct MetalArtifactInstance {
        core: InstanceCore,
        backend: Arc<MetalBackend>,
        modules: Vec<MetalExecutableModule>,
    }

    impl ExecutableModule for MetalExecutableModule {
        vyre_driver::executable_module!();
    }

    impl ArtifactInstance for MetalArtifactInstance {
        vyre_driver::artifact_instance_identity!();

        fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
            self.core.submit_host_only(
                &bindings,
                "Metal artifact resident binding",
                |state, invocation_grid| self.execute(state, invocation_grid),
            )
        }
    }

    impl MetalArtifactInstance {
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
                |executable, plan, config, state| {
                    let inputs = self.core.gather_inputs(
                        plan,
                        &executable.program,
                        state,
                        materialize::unbound_input,
                    )?;
                    self.backend.dispatch_target_module(
                        &executable.module,
                        &executable.program,
                        &inputs,
                        config,
                    )
                },
            )
        }
    }

    pub(super) fn factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
        let backend = Arc::new(MetalBackend::acquire()?);
        let format = TargetPayloadFormat::new("msl", METAL_TARGET_FORMAT_VERSION)
            .map_err(|error| materialize::compile_error(METAL_BACKEND_ID, error))?;
        let profile = crate::target_compiler::target_profile()?;
        let generation = ResidentOwner::new()?.get();
        let device = backend.artifact_device_name();
        Ok(Box::new(MetalMaterializer {
            backend,
            descriptor: MaterializerDevice::new(
                DeviceIdentity {
                    backend: METAL_BACKEND_ID,
                    device,
                    generation,
                },
                format,
                profile,
            ),
        }))
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        native::factory()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Err(BackendError::UnsupportedFeature {
            name: "Apple Metal.framework artifact materialization".to_string(),
            backend: crate::METAL_BACKEND_ID.to_string(),
        })
    }
}
