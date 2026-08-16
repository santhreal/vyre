use vyre_driver::{ArtifactMaterializer, BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod native {
    use std::sync::Arc;

    use vyre_driver::materialize::{
        self, DeviceSpec, ExecutableModule, InstanceCore, MaterializedInstance, MaterializerDevice,
        ResidentInstance,
    };
    use vyre_driver::{
        ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, DispatchConfig, Resource,
        Submission, TimedDispatchResult, VyreBackend,
    };
    use vyre_foundation::ir::Program;
    use vyre_megakernel::{Artifact, TargetPayload};

    use crate::runtime::{MetalBackend, MetalTargetModule};
    use crate::target_compiler::METAL_TARGET_FORMAT_VERSION;
    use crate::METAL_BACKEND_ID;

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
                    .instance(artifact, payload, materialize::NEUTRAL_MESSAGES)?,
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
            self.submit_routed(&bindings, || {
                materialize::invalid_module(
                    "Metal artifact submission cannot mix host and resident resources",
                )
            })
        }
    }

    impl MaterializedInstance for MetalArtifactInstance {
        type Module = MetalExecutableModule;

        fn core(&self) -> &InstanceCore {
            &self.core
        }

        fn modules(&self) -> &[Self::Module] {
            &self.modules
        }

        fn module_label(&self) -> &'static str {
            "Metal target module"
        }

        fn dispatch(
            &self,
            module: &Self::Module,
            inputs: &[&[u8]],
            config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            self.backend
                .dispatch_target_module(&module.module, &module.program, inputs, config)
        }
    }

    /// `MetalBackend` holds a resident buffer table and dispatches against it,
    /// so refusing every resident binding here made the artifact path the one
    /// caller that could not use it: a chained pipeline had to round trip each
    /// stage through the host to reach the next.
    impl ResidentInstance for MetalArtifactInstance {
        fn multi_module_feature(&self) -> &str {
            "Metal resident submission for multi-module artifacts"
        }

        fn resident_module_label(&self) -> &'static str {
            "Metal resident target module"
        }

        fn launch_resident(
            &self,
            module: &Self::Module,
            ordered: &[Resource],
            config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            VyreBackend::dispatch_resident_timed(
                self.backend.as_ref(),
                &module.program,
                ordered,
                config,
            )
        }
    }

    pub(super) fn factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
        let backend = Arc::new(MetalBackend::acquire()?);
        let device = backend.artifact_device_name();
        Ok(Box::new(MetalMaterializer {
            backend,
            descriptor: MaterializerDevice::acquire(DeviceSpec {
                backend: METAL_BACKEND_ID,
                device,
                format_extension: "msl",
                format_version: METAL_TARGET_FORMAT_VERSION,
                profile: crate::target_compiler::target_profile()?,
            })?,
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
