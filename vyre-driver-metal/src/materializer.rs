use vyre_driver::{ArtifactMaterializer, BackendError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod native {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use vyre_driver::{
        ArtifactInstance, BindingPlan, BindingSet, BoundResource, Completion, Device,
        DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
    };
    use vyre_foundation::ir::Program;
    use vyre_megakernel::{
        Artifact, ArtifactValueId, Digest, ResourceLifetime, TargetModuleBundle, TargetPayload,
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
            if payload.neutral_artifact() != artifact.digest() {
                return Err(invalid_module(
                    "target payload is not authenticated for the supplied neutral artifact",
                ));
            }
            if payload.format() != self.descriptor.target_format() {
                return Err(BackendError::UnsupportedFeature {
                    name: format!("target payload format `{}`", payload.format().identity()),
                    backend: METAL_BACKEND_ID.to_string(),
                });
            }
            if payload.profile() != self.descriptor.target_profile() {
                return Err(invalid_module(
                    "target payload profile does not match the acquired materializer profile",
                ));
            }
            let bundle = TargetModuleBundle::from_bytes(payload.bytes()).map_err(compile_error)?;
            let selected = artifact.fusion();
            if bundle.modules.len() != selected.len() {
                return Err(invalid_module(
                    "target module count must equal the compiler-selected fusion-group count",
                ));
            }
            if payload.entries().len() != selected.len() {
                return Err(invalid_module(
                    "target entry count must equal the compiler-selected fusion-group count",
                ));
            }
            let mut modules = Vec::with_capacity(selected.len());
            for ((image, selected), entry) in bundle
                .modules
                .into_iter()
                .zip(selected)
                .zip(payload.entries())
            {
                if image.group != selected.id
                    || image.stage != selected.stage
                    || image.nodes != selected.members
                {
                    return Err(invalid_module(
                        "target module group/stage/node identity must match the neutral selected plan",
                    ));
                }
                let target: vyre_emit_metal::MetalArtifact = serde_json::from_slice(&image.bytes)
                    .map_err(|error| {
                    invalid_module(&format!("Metal target artifact is malformed: {error}"))
                })?;
                if image.entry_point != target.entry_point {
                    return Err(invalid_module(
                        "module bundle and Metal artifact entry points disagree",
                    ));
                }
                if entry.name != image.entry_point {
                    return Err(invalid_module(
                        "target entry metadata must name the emitted Metal entry point",
                    ));
                }
                let mut config = DispatchConfig::default();
                config.grid_override = Some(entry.grid_size);
                config.dispatch_grid = Some(entry.grid_size);
                let program = Arc::new(Program::from_wire(&image.program).map_err(|error| {
                    invalid_module(&format!("selected Program is malformed: {error}"))
                })?);
                if target.workgroup_size != program.workgroup_size {
                    return Err(invalid_module(
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
            Ok(Box::new(MetalArtifactInstance {
                artifact: artifact.digest(),
                payload: payload.digest(),
                device: self.descriptor.identity.clone(),
                backend: Arc::clone(&self.backend),
                modules,
                values: artifact
                    .resources()
                    .iter()
                    .map(|resource| (resource.name.clone(), resource.value))
                    .collect(),
                outputs: artifact
                    .resources()
                    .iter()
                    .filter(|resource| resource.lifetime == ResourceLifetime::Output)
                    .map(|resource| resource.value)
                    .collect(),
                retained: artifact
                    .resources()
                    .iter()
                    .filter(|resource| resource.lifetime == ResourceLifetime::Retained)
                    .map(|resource| resource.value)
                    .collect(),
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
                return Err(invalid_module("bindings name a different neutral artifact"));
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
                            invalid_module(&format!(
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
                        invalid_module(&format!(
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
                            invalid_module(&format!(
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
                            invalid_module(&format!(
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
                invalid_module(&format!(
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
                invalid_module("each Submission completion may be consumed only once")
            })?
        }
    }

    pub(super) fn factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
        let backend = Arc::new(MetalBackend::acquire()?);
        let format =
            TargetPayloadFormat::new("msl", METAL_TARGET_FORMAT_VERSION).map_err(compile_error)?;
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

    fn invalid_module(reason: &str) -> BackendError {
        BackendError::InvalidProgram {
            fix: format!("Fix: {reason}. Recompile the target payload from the neutral artifact."),
        }
    }

    fn compile_error(error: impl std::fmt::Display) -> BackendError {
        BackendError::KernelCompileFailed {
            backend: METAL_BACKEND_ID.to_string(),
            compiler_message: format!(
                "{error}. Fix: rebuild the target payload from the neutral artifact."
            ),
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
