use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_driver::materialize::{
    self, ExecutableModule, InstanceCore, InstanceMessages, MaterializedInstance,
    MaterializerDevice, ResidentInstance,
};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, CompiledPipeline,
    Device, DeviceIdentity, DispatchConfig, Resource, ResidentOwner, Submission,
    TimedDispatchResult,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, TargetPayload, TargetPayloadFormat};

use crate::descriptor_mapping::descriptor_bind_group;
use crate::pipeline::WgpuPipeline;
use crate::target_compiler::{
    WgpuTargetModule, WGPU_TARGET_FORMAT_VERSION, WGPU_TARGET_MODULE_SCHEMA_VERSION,
};
use crate::{WgpuBackend, WGPU_BACKEND_ID};
use vyre_lower::TRAP_SIDECAR_NAME;

/// Rejection for a host dispatch that skipped a declared output slot.
fn omitted_output(output_index: usize, name: &str) -> BackendError {
    materialize::omitted_output("WGSL target module", output_index, name)
}

/// Rejection for a resident dispatch that skipped a declared output slot.
fn omitted_resident_output(output_index: usize, name: &str) -> BackendError {
    materialize::omitted_output("WGPU resident target module", output_index, name)
}

/// Resident-path rejection text. This backend names an unproduced or
/// unpreserved resident value without its lifetime class, where the host path
/// names the class; every other rejection is the neutral wording.
const RESIDENT_MESSAGES: InstanceMessages = InstanceMessages {
    missing_output_value: |value| {
        materialize::invalid_module(&format!(
            "selected execution did not produce canonical value {}",
            value.0
        ))
    },
    missing_retained_value: |value| {
        materialize::invalid_module(&format!(
            "selected execution did not preserve canonical value {}",
            value.0
        ))
    },
    ..materialize::NEUTRAL_MESSAGES
};

pub(crate) struct WgpuMaterializer {
    backend: WgpuBackend,
    descriptor: MaterializerDevice,
    lost: Arc<AtomicBool>,
}

impl ArtifactMaterializer for WgpuMaterializer {
    vyre_driver::materializer_passthrough!(backend);

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if !self.descriptor.is_healthy() {
            return Err(device_lost_error(self.descriptor.identity()));
        }
        let modules =
            self.descriptor
                .admit_modules(WGPU_BACKEND_ID, artifact, payload, |module| {
                    let target: WgpuTargetModule = serde_json::from_slice(&module.image.bytes)
                        .map_err(|error| {
                            materialize::invalid_module(&format!(
                                "WGSL target module is malformed: {error}"
                            ))
                        })?;
                    if target.schema_version != WGPU_TARGET_MODULE_SCHEMA_VERSION {
                        return Err(materialize::invalid_module(
                            "WGSL target module schema is unsupported",
                        ));
                    }
                    if !target.wgsl.contains("@compute") || !target.wgsl.contains("fn main(") {
                        return Err(materialize::invalid_module(
                            "WGSL target module does not define compute entry point `main`",
                        ));
                    }
                    let program = module.program;
                    let config = module.config;
                    let binding_plan = BindingPlan::build(&program)?;
                    let input_slots = module
                        .image
                        .descriptor
                        .bindings
                        .slots
                        .iter()
                        .filter(|slot| {
                            descriptor_bind_group(slot.memory_class).is_some()
                                && slot.name != TRAP_SIDECAR_NAME
                        })
                        .map(|slot| {
                            let required = binding_plan
                                .bindings
                                .iter()
                                .find(|binding| {
                                    program.buffers()[binding.buffer_index].name() == slot.name
                                })
                                .is_none_or(|binding| binding.input_index.is_some());
                            ArtifactInputSlot {
                                name: slot.name.clone(),
                                required,
                            }
                        })
                        .collect();
                    let pipeline = self.backend.compile_pipeline(
                        &program,
                        &config,
                        Some(crate::pipeline::AuthenticatedTarget {
                            wgsl: &target.wgsl,
                            descriptor: &module.image.descriptor,
                        }),
                    )?;
                    let resident_slots = pipeline
                        .persistent_resource_names()
                        .map(str::to_owned)
                        .collect();
                    Ok(WgpuExecutableModule {
                        program,
                        pipeline,
                        input_slots,
                        resident_slots,
                        config,
                    })
                })?;
        Ok(Box::new(WgpuArtifactInstance {
            core: self
                .descriptor
                .instance(artifact, payload, materialize::NEUTRAL_MESSAGES),
            lost: Arc::clone(&self.lost),
            modules,
        }))
    }
}

struct ArtifactInputSlot {
    name: String,
    required: bool,
}

struct WgpuExecutableModule {
    program: Arc<Program>,
    pipeline: Arc<WgpuPipeline>,
    input_slots: Vec<ArtifactInputSlot>,
    resident_slots: Vec<String>,
    config: DispatchConfig,
}

struct WgpuArtifactInstance {
    core: InstanceCore,
    lost: Arc<AtomicBool>,
    modules: Vec<WgpuExecutableModule>,
}

impl ExecutableModule for WgpuExecutableModule {
    vyre_driver::executable_module!();
}

impl ArtifactInstance for WgpuArtifactInstance {
    vyre_driver::artifact_instance_identity!();

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        if self.lost.load(Ordering::Acquire) {
            return Err(device_lost_error(&self.core.device));
        }
        self.submit_routed(&bindings, || {
            materialize::invalid_module(
                "WGPU artifact submission cannot mix host and resident resources",
            )
        })
    }
}

impl MaterializedInstance for WgpuArtifactInstance {
    type Module = WgpuExecutableModule;

    fn core(&self) -> &InstanceCore {
        &self.core
    }

    fn modules(&self) -> &[Self::Module] {
        &self.modules
    }

    fn omitted_output(&self) -> fn(usize, &str) -> BackendError {
        omitted_output
    }

    fn launch(
        &self,
        module: &Self::Module,
        _plan: &BindingPlan,
        config: &DispatchConfig,
        state: &BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<TimedDispatchResult, BackendError> {
        let inputs = self.gather_slot_inputs(module, state)?;
        match module.pipeline.dispatch_borrowed_timed(&inputs, config) {
            Err(_) if self.lost.load(Ordering::Acquire) => {
                Err(device_lost_error(&self.core.device))
            }
            result => result,
        }
    }
}

impl ResidentInstance for WgpuArtifactInstance {
    fn multi_module_feature(&self) -> &str {
        "WGPU resident submission for multi-module artifacts"
    }

    fn omitted_resident_output(&self) -> fn(usize, &str) -> BackendError {
        omitted_resident_output
    }

    fn resident_messages(&self) -> &InstanceMessages {
        &RESIDENT_MESSAGES
    }

    /// Resolve resident handles into the order the emitted target module
    /// declares, which is the order its pipeline reports rather than the
    /// binding plan's.
    fn ordered_resident(
        &self,
        module: &Self::Module,
        _plan: &BindingPlan,
        resources: &BTreeMap<ArtifactValueId, Resource>,
    ) -> Result<Vec<Resource>, BackendError> {
        self.core.ordered_resident_resources(
            module.resident_slots.iter().map(String::as_str),
            resources,
            |value, name| {
                materialize::invalid_module(&format!(
                    "canonical artifact value {} for resident target binding `{name}` is unbound",
                    value.0
                ))
            },
        )
    }

    fn launch_resident(
        &self,
        module: &Self::Module,
        ordered: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        CompiledPipeline::dispatch_persistent_handles_timed(
            module.pipeline.as_ref(),
            ordered,
            config,
        )
    }
}

impl WgpuArtifactInstance {
    /// Borrow bound bytes into the order this backend's target bindings declare.
    ///
    /// The input order comes from the emitted descriptor slots rather than the
    /// binding plan, because a slot the target module declares but the plan does
    /// not require is bound empty instead of rejected.
    fn gather_slot_inputs<'state>(
        &self,
        module: &WgpuExecutableModule,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
        let mut inputs = Vec::with_capacity(module.input_slots.len());
        for slot in &module.input_slots {
            let value = self.core.value_for_buffer(&slot.name)?;
            match state.get(&value) {
                Some(bytes) => inputs.push(bytes.as_slice()),
                None if !slot.required => inputs.push(&[]),
                None => {
                    return Err(materialize::invalid_module(&format!(
                        "canonical artifact value {} for target binding `{}` is unbound",
                        value.0, slot.name
                    )));
                }
            }
        }
        Ok(inputs)
    }
}

fn device_lost_error(identity: &DeviceIdentity) -> BackendError {
    BackendError::DeviceLost {
        backend: identity.backend.to_string(),
        device: identity.device.clone(),
        generation: identity.generation,
        message: "the WGPU device-loss callback invalidated this generation".to_string(),
    }
}

pub(crate) fn materializer_for_backend(
    backend: WgpuBackend,
) -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let format = TargetPayloadFormat::new("wgsl", WGPU_TARGET_FORMAT_VERSION)
        .map_err(|error| materialize::compile_error(WGPU_BACKEND_ID, error))?;
    let profile = crate::target_compiler::target_profile()?;
    let generation = ResidentOwner::new()?.get();
    let device = backend.adapter_name.to_string();
    let lost = Arc::clone(&backend.device_lost);
    Ok(Box::new(WgpuMaterializer {
        backend,
        descriptor: MaterializerDevice::revocable(
            DeviceIdentity {
                backend: WGPU_BACKEND_ID,
                device,
                generation,
            },
            format,
            profile,
            Arc::clone(&lost),
        ),
        lost,
    }))
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    materializer_for_backend(WgpuBackend::acquire()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use vyre_megakernel::Digest;

    /// WHY: runtime recovery must receive a stable device-loss class, never text to parse.
    #[test]
    fn lost_instance_submission_is_structured() {
        let digest = Digest([7; 32]);
        let instance = WgpuArtifactInstance {
            core: InstanceCore {
                artifact: digest,
                payload: Digest([8; 32]),
                device: DeviceIdentity {
                    backend: WGPU_BACKEND_ID,
                    device: "fault-injection".to_string(),
                    generation: 11,
                },
                values: BTreeMap::new(),
                outputs: BTreeSet::new(),
                retained: BTreeSet::new(),
                messages: materialize::NEUTRAL_MESSAGES,
            },
            lost: Arc::new(AtomicBool::new(true)),
            modules: Vec::new(),
        };

        let error = match instance.submit(BindingSet::new(digest)) {
            Ok(_) => panic!("a lost device generation must reject submission"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BackendError::DeviceLost { generation: 11, .. }
        ));
    }
}
