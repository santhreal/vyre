use vyre_driver::{ArtifactMaterializer, BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod native {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use vyre_driver::{
        ArtifactInstance, BindingPlan, BindingSet, BoundResource, Completion, Device,
        DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
    };
    use vyre_driver::materialize;
    use vyre_foundation::ir::Program;
    use vyre_megakernel::{
        Artifact, ArtifactValueId, Digest, TargetPayload,
        TargetPayloadFormat, TargetProfile,
    };

    use crate::runtime::{MetalBackend, MetalTargetModule};
    use crate::target_compiler::METAL_TARGET_FORMAT_VERSION;
    use crate::METAL_BACKEND_ID;

    struct MetalDevice {
        identity: DeviceIdentity,
        format: TargetPayloadFormat,
        profile: TargetProfile,
    }

    impl Device for MetalDevice {
        fn identity(&self) -> &DeviceIdentity {
            &self.identity
        }

        fn target_format(&self) -> &TargetPayloadFormat {
            &self.format
        }

        fn target_profile(&self) -> &TargetProfile {
            &self.profile
        }

        fn is_healthy(&self) -> bool {
            true
        }
    }

    pub(super) struct MetalMaterializer {
        backend: Arc<MetalBackend>,
        descriptor: MetalDevice,
    }

    impl ArtifactMaterializer for MetalMaterializer {
        fn device(&self) -> &dyn Device {
            &self.descriptor
        }

        fn materialize(
            &self,
            artifact: &Artifact,
            payload: &TargetPayload,
        ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
            let admitted = materialize::admit(
                artifact,
                payload,
                materialize::MaterializerTarget {
                    backend_id: METAL_BACKEND_ID,
                    format: self.descriptor.target_format(),
                    profile: self.descriptor.target_profile(),
                },
            )?;
            let mut modules = Vec::with_capacity(admitted.len());
            for admitted_module in admitted {
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
                modules.push(MetalExecutableModule {
                    program,
                    module,
                    config,
                });
            }
            let resources = materialize::project_resources(artifact);
            Ok(Box::new(MetalArtifactInstance {
                artifact: artifact.digest(),
                payload: payload.digest(),
                device: self.descriptor.identity.clone(),
                backend: Arc::clone(&self.backend),
                modules,
                values: resources.values,
                outputs: resources.outputs,
                retained: resources.retained,
            }))
        }
    }

    struct MetalExecutableModule {
        program: Arc<Program>,
        module: MetalTargetModule,
        config: DispatchConfig,
    }

    struct MetalArtifactInstance {
        artifact: Digest,
        payload: Digest,
        device: DeviceIdentity,
        backend: Arc<MetalBackend>,
        modules: Vec<MetalExecutableModule>,
        values: BTreeMap<String, ArtifactValueId>,
        outputs: BTreeSet<ArtifactValueId>,
        retained: BTreeSet<ArtifactValueId>,
    }

    impl ArtifactInstance for MetalArtifactInstance {
        fn artifact(&self) -> Digest {
            self.artifact
        }

        fn payload(&self) -> Digest {
            self.payload
        }

        fn device(&self) -> &DeviceIdentity {
            &self.device
        }

        fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
            if bindings.artifact() != self.artifact {
                return Err(materialize::invalid_module("bindings name a different neutral artifact"));
            }
            let invocation_grid = bindings.invocation_grid();
            let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
            for (value, resource) in bindings.resources() {
                match resource {
                    BoundResource::Host(bytes) => {
                        state.insert(*value, bytes.clone());
                    }
                    BoundResource::Resident(_) => {
                        return Err(BackendError::UnsupportedFeature {
                            name: "Metal artifact resident binding".to_string(),
                            backend: METAL_BACKEND_ID.to_string(),
                        });
                    }
                }
            }
            Ok(Box::new(ReadySubmission {
                result: Some(self.execute(state, invocation_grid)),
            }))
        }
    }

    impl MetalArtifactInstance {
        fn execute(
            &self,
            mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
            invocation_grid: Option<[u32; 3]>,
        ) -> Result<Completion, BackendError> {
            for executable in &self.modules {
                let mut config = executable.config.clone();
                if let Some(grid) = invocation_grid {
                    config.grid_override = Some(grid);
                    config.dispatch_grid = Some(grid);
                }
                let plan = BindingPlan::build(&executable.program)?;
                let input_count = plan
                    .bindings
                    .iter()
                    .filter_map(|binding| binding.input_index)
                    .max()
                    .map_or(0, |index| index + 1);
                let mut inputs = vec![&[][..]; input_count];
                for binding in &plan.bindings {
                    let Some(input_index) = binding.input_index else {
                        continue;
                    };
                    let buffer = &executable.program.buffers()[binding.buffer_index];
                    let value = self.value_for_buffer(buffer.name())?;
                    inputs[input_index] =
                        state.get(&value).map(Vec::as_slice).ok_or_else(|| {
                            materialize::invalid_module(&format!(
                                "canonical artifact value {} for Program buffer `{}` is unbound",
                                value.0,
                                buffer.name()
                            ))
                        })?;
                }
                let dispatched = self.backend.dispatch_target_module(
                    &executable.module,
                    &executable.program,
                    &inputs,
                    &config,
                )?;
                for binding in &plan.bindings {
                    let Some(output_index) = binding.output_index else {
                        continue;
                    };
                    let buffer = &executable.program.buffers()[binding.buffer_index];
                    let value = self.value_for_buffer(buffer.name())?;
                    let bytes = dispatched.outputs.get(output_index).ok_or_else(|| {
                        materialize::invalid_module(&format!(
                            "Metal target module omitted output {output_index} for Program buffer `{}`",
                            buffer.name()
                        ))
                    })?;
                    state.insert(value, bytes.clone());
                }
            }
            let outputs = self
                .outputs
                .iter()
                .map(|value| {
                    state
                        .get(value)
                        .cloned()
                        .map(|bytes| (*value, bytes))
                        .ok_or_else(|| {
                            materialize::invalid_module(&format!(
                                "selected execution did not produce canonical output value {}",
                                value.0
                            ))
                        })
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            let retained = self
                .retained
                .iter()
                .map(|value| {
                    state
                        .get(value)
                        .cloned()
                        .map(|bytes| (*value, bytes))
                        .ok_or_else(|| {
                            materialize::invalid_module(&format!(
                                "selected execution did not preserve retained value {}",
                                value.0
                            ))
                        })
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            Ok(Completion {
                artifact: self.artifact,
                outputs,
                retained,
                device_ns: None,
            })
        }

        fn value_for_buffer(&self, name: &str) -> Result<ArtifactValueId, BackendError> {
            self.values.get(name).copied().ok_or_else(|| {
                materialize::invalid_module(&format!(
                    "Program buffer `{name}` is absent from the canonical artifact ABI"
                ))
            })
        }
    }

    struct ReadySubmission {
        result: Option<Result<Completion, BackendError>>,
    }

    impl Submission for ReadySubmission {
        fn is_ready(&self) -> bool {
            true
        }

        fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
            self.result.take().ok_or_else(|| {
                materialize::invalid_module("each Submission completion may be consumed only once")
            })?
        }
    }

    pub(super) fn factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
        let backend = Arc::new(MetalBackend::acquire()?);
        let format =
            TargetPayloadFormat::new("msl", METAL_TARGET_FORMAT_VERSION).map_err(|error| materialize::compile_error(METAL_BACKEND_ID, error))?;
        let profile = crate::target_compiler::target_profile()?;
        let generation = ResidentOwner::new()?.get();
        let device = backend.artifact_device_name();
        Ok(Box::new(MetalMaterializer {
            backend,
            descriptor: MetalDevice {
                identity: DeviceIdentity {
                    backend: METAL_BACKEND_ID,
                    device,
                    generation,
                },
                format,
                profile,
            },
        }))
    }

    }

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
