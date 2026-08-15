use vyre_driver::{ArtifactMaterializer, BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod native {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use vyre_driver::materialize::{self, ExecutableModule, InstanceCore, MaterializerDevice};
    use vyre_driver::{
        ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingRole, BindingSet,
        Completion, DeviceIdentity, DispatchConfig, Resource, ResidentOwner, Submission,
        VyreBackend,
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

    /// Rejection for a resident dispatch that skipped a declared output slot.
    fn omitted_resident_output(output_index: usize, name: &str) -> BackendError {
        materialize::omitted_output("Metal resident target module", output_index, name)
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
            self.core.route_submission(
                &bindings,
                || {
                    materialize::invalid_module(
                        "Metal artifact submission cannot mix host and resident resources",
                    )
                },
                |state, invocation_grid| self.execute(state, invocation_grid),
                |resources, invocation_grid| self.execute_resident(resources, invocation_grid),
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

        /// Launch the single module over caller-owned resident resources.
        ///
        /// `MetalBackend` holds a resident buffer table and dispatches against
        /// it, so refusing every resident binding here made the artifact path
        /// the one caller that could not use it: a chained pipeline had to round
        /// trip each stage through the host to reach the next. The resident
        /// order is the binding plan's non-shared roles, which is the order
        /// `dispatch_resident_timed` reads.
        fn execute_resident(
            &self,
            resources: &BTreeMap<ArtifactValueId, Resource>,
            invocation_grid: Option<[u32; 3]>,
        ) -> Result<Completion, BackendError> {
            let module = self.core.single_resident_module(
                &self.modules,
                "Metal resident submission for multi-module artifacts",
            )?;
            let plan = BindingPlan::build(&module.program)?;
            let ordered = self.core.ordered_resident_resources(
                resident_resource_bindings(&plan)
                    .map(|binding| module.program.buffers()[binding.buffer_index].name()),
                resources,
                |value, name| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: bind canonical artifact value {} for resident Program buffer `{name}`.",
                        value.0
                    ),
                },
            )?;
            let mut config = module.config.clone();
            if let Some(grid) = invocation_grid {
                config.grid_override = Some(grid);
                config.dispatch_grid = Some(grid);
            }
            let dispatched = VyreBackend::dispatch_resident_timed(
                self.backend.as_ref(),
                &module.program,
                &ordered,
                &config,
            )?;
            self.core.resident_completion(
                &plan,
                &module.program,
                dispatched,
                omitted_resident_output,
                &self.core.messages,
            )
        }
    }

    /// The bindings a resident launch takes a resource for, in binding order.
    fn resident_resource_bindings(
        plan: &BindingPlan,
    ) -> impl Iterator<Item = &vyre_driver::Binding> {
        plan.bindings
            .iter()
            .filter(|binding| binding.role != BindingRole::Shared)
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
