use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_driver::materialize::{self, InstanceCore, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, CompiledPipeline,
    Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner, Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, Digest, TargetPayload, TargetPayloadFormat};

use crate::descriptor_mapping::descriptor_bind_group;
use crate::pipeline::WgpuPipeline;
use crate::target_compiler::{
    WgpuTargetModule, WGPU_TARGET_FORMAT_VERSION, WGPU_TARGET_MODULE_SCHEMA_VERSION,
};
use crate::{WgpuBackend, WGPU_BACKEND_ID};
use vyre_lower::TRAP_SIDECAR_NAME;

/// Rejection for a host dispatch that skipped a declared output slot.
fn omitted_output(output_index: usize, name: &str) -> BackendError {
    materialize::invalid_module(&format!(
        "WGSL target module omitted output {output_index} for Program buffer `{name}`"
    ))
}

/// Rejection for a resident dispatch that skipped a declared output slot.
fn omitted_resident_output(output_index: usize, name: &str) -> BackendError {
    materialize::invalid_module(&format!(
        "WGPU resident target module omitted output {output_index} for Program buffer `{name}`"
    ))
}

/// Rejection for an unproduced output value on the resident path, which names
/// the value without its lifetime class.
fn unproduced_resident_value(value: ArtifactValueId) -> BackendError {
    materialize::invalid_module(&format!(
        "selected execution did not produce canonical value {}",
        value.0
    ))
}

/// Rejection for an unpreserved retained value on the resident path, which
/// names the value without its lifetime class.
fn unpreserved_resident_value(value: ArtifactValueId) -> BackendError {
    materialize::invalid_module(&format!(
        "selected execution did not preserve canonical value {}",
        value.0
    ))
}

pub(crate) struct WgpuMaterializer {
    backend: WgpuBackend,
    descriptor: MaterializerDevice,
    lost: Arc<AtomicBool>,
}

impl ArtifactMaterializer for WgpuMaterializer {
    fn device(&self) -> &dyn Device {
        &self.descriptor
    }
    fn allocate_resident(&self, byte_len: usize) -> Result<vyre_driver::Resource, BackendError> {
        vyre_driver::VyreBackend::allocate_resident(&self.backend, byte_len)
    }

    fn upload_resident(
        &self,
        resource: &vyre_driver::Resource,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::upload_resident(&self.backend, resource, bytes)
    }

    fn upload_resident_at(
        &self,
        resource: &vyre_driver::Resource,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::upload_resident_at(&self.backend, resource, offset_bytes, bytes)
    }

    fn free_resident(&self, resource: vyre_driver::Resource) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::free_resident(&self.backend, resource)
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if !self.descriptor.is_healthy() {
            return Err(device_lost_error(self.descriptor.identity()));
        }
        let admitted = materialize::admit(
            artifact,
            payload,
            self.descriptor.target(WGPU_BACKEND_ID),
        )?;
        let mut modules = Vec::with_capacity(admitted.len());
        for module in admitted {
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
                        .find(|binding| program.buffers()[binding.buffer_index].name() == slot.name)
                        .is_none_or(|binding| binding.input_index.is_some());
                    ArtifactInputSlot {
                        name: slot.name.clone(),
                        required,
                    }
                })
                .collect();
            let pipeline = WgpuPipeline::compile_target_with_device_queue(
                &program,
                &target.wgsl,
                &module.image.descriptor,
                &config,
                self.backend.adapter_info.clone(),
                self.backend.enabled_features,
                self.backend.current_device_queue(),
                self.backend.dispatch_arena_snapshot(),
                self.backend.current_persistent_pool(),
                Arc::clone(&self.backend.pipeline_cache),
                Arc::clone(&self.backend.bind_group_layout_cache),
            )?;
            let resident_slots = pipeline
                .persistent_resource_names()
                .map(str::to_owned)
                .collect();
            modules.push(WgpuExecutableModule {
                program,
                pipeline,
                input_slots,
                resident_slots,
                config,
            });
        }
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

impl ArtifactInstance for WgpuArtifactInstance {
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
        if self.lost.load(Ordering::Acquire) {
            return Err(device_lost_error(&self.core.device));
        }
        self.core.accept(&bindings)?;
        let invocation_grid = bindings.invocation_grid();
        let bound = materialize::partition_bindings(&bindings);
        if !bound.host.is_empty() && !bound.resident.is_empty() {
            return Err(materialize::invalid_module(
                "WGPU artifact submission cannot mix host and resident resources",
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

impl WgpuArtifactInstance {
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
            let dispatched = match module.pipeline.dispatch_borrowed_timed(&inputs, &config) {
                Err(_) if self.lost.load(Ordering::Acquire) => {
                    return Err(device_lost_error(&self.core.device));
                }
                result => result?,
            };
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
                name: "WGPU resident submission for multi-module artifacts".to_string(),
                backend: WGPU_BACKEND_ID.to_string(),
            });
        }
        let module = &self.modules[0];
        let mut ordered = Vec::with_capacity(module.resident_slots.len());
        for name in &module.resident_slots {
            let value = self.core.value_for_buffer(name)?;
            let resource = resources.get(&value).ok_or_else(|| {
                materialize::invalid_module(&format!(
                    "canonical artifact value {} for resident target binding `{name}` is unbound",
                    value.0
                ))
            })?;
            ordered.push(resource.clone());
        }
        let mut config = module.config.clone();
        materialize::override_grid(&mut config, invocation_grid);
        let dispatched = module
            .pipeline
            .dispatch_persistent_handles_timed(&ordered, &config)?;
        let plan = BindingPlan::build(&module.program)?;
        let mut state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        self.core.absorb_outputs(
            &plan,
            &module.program,
            &dispatched.outputs,
            &mut state,
            omitted_resident_output,
        )?;
        Ok(Completion {
            artifact: self.core.artifact,
            outputs: self
                .core
                .project(&self.core.outputs, &state, unproduced_resident_value)?,
            retained: self
                .core
                .project(&self.core.retained, &state, unpreserved_resident_value)?,
            device_ns: dispatched.device_ns,
        })
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
