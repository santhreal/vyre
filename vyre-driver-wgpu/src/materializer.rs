use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet, BoundResource,
    CompiledPipeline, Completion, Device, DeviceIdentity, DispatchConfig, ResidentOwner,
    Submission,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    fuse_selected_module, selected_modules, Artifact, ArtifactValueId, Digest, ResourceLifetime,
    TargetModuleBundle, TargetPayload, TargetPayloadFormat,
};

use crate::descriptor_mapping::descriptor_bind_group;
use crate::pipeline::WgpuPipeline;
use crate::target_compiler::{
    WgpuTargetModule, WGPU_TARGET_FORMAT_VERSION, WGPU_TARGET_MODULE_SCHEMA_VERSION,
};
use crate::{WgpuBackend, WGPU_BACKEND_ID};
use vyre_lower::TRAP_SIDECAR_NAME;

struct WgpuDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
    lost: Arc<AtomicBool>,
}

impl Device for WgpuDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn is_healthy(&self) -> bool {
        !self.lost.load(Ordering::Acquire)
    }
}

pub(crate) struct WgpuMaterializer {
    backend: WgpuBackend,
    descriptor: WgpuDevice,
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

    fn free_resident(&self, resource: vyre_driver::Resource) -> Result<(), BackendError> {
        vyre_driver::VyreBackend::free_resident(&self.backend, resource)
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if !self.descriptor.is_healthy() {
            return Err(device_lost_error(&self.descriptor.identity));
        }
        if payload.neutral_artifact() != artifact.digest() {
            return Err(invalid_module(
                "target payload is not authenticated for the supplied neutral artifact",
            ));
        }
        if payload.format() != self.descriptor.target_format() {
            return Err(BackendError::UnsupportedFeature {
                name: format!("target payload format `{}`", payload.format().identity()),
                backend: WGPU_BACKEND_ID.to_string(),
            });
        }
        let bundle = TargetModuleBundle::from_bytes(payload.bytes()).map_err(compile_error)?;
        let selected = selected_modules(artifact).map_err(compile_error)?;
        if bundle.modules.len() != selected.len() {
            return Err(invalid_module(
                "target module count must equal the compiler-selected fusion-group count",
            ));
        }
        let config = DispatchConfig::default();
        let mut modules = Vec::with_capacity(selected.len());
        for (image, selected) in bundle.modules.into_iter().zip(selected) {
            if image.group != selected.group || image.stage != selected.stage {
                return Err(invalid_module(
                    "target module group/stage identity must match the neutral selected plan",
                ));
            }
            if image.entry_point != "main" {
                return Err(invalid_module(
                    "WGSL target module entry point must be `main`",
                ));
            }
            let target: WgpuTargetModule =
                serde_json::from_slice(&image.bytes).map_err(|error| {
                    invalid_module(&format!("WGSL target module is malformed: {error}"))
                })?;
            if target.schema_version != WGPU_TARGET_MODULE_SCHEMA_VERSION {
                return Err(invalid_module("WGSL target module schema is unsupported"));
            }
            if !target.wgsl.contains("@compute") || !target.wgsl.contains("fn main(") {
                return Err(invalid_module(
                    "WGSL target module does not define compute entry point `main`",
                ));
            }
            let program = Arc::new(fuse_selected_module(&selected).map_err(compile_error)?);
            let binding_plan = BindingPlan::build(&program)?;
            let input_slots = target
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
                &target.descriptor,
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
            });
        }
        Ok(Box::new(WgpuArtifactInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.descriptor.identity.clone(),
            lost: Arc::clone(&self.descriptor.lost),
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

struct ArtifactInputSlot {
    name: String,
    required: bool,
}

struct WgpuExecutableModule {
    program: Arc<Program>,
    pipeline: Arc<WgpuPipeline>,
    input_slots: Vec<ArtifactInputSlot>,
    resident_slots: Vec<String>,
}

struct WgpuArtifactInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    lost: Arc<AtomicBool>,
    modules: Vec<WgpuExecutableModule>,
    values: BTreeMap<String, ArtifactValueId>,
    outputs: BTreeSet<ArtifactValueId>,
    retained: BTreeSet<ArtifactValueId>,
}

impl ArtifactInstance for WgpuArtifactInstance {
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
        if self.lost.load(Ordering::Acquire) {
            return Err(device_lost_error(&self.device));
        }
        if bindings.artifact() != self.artifact {
            return Err(invalid_module("bindings name a different neutral artifact"));
        }
        let mut host_state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        let mut resident_state = BTreeMap::<ArtifactValueId, vyre_driver::Resource>::new();
        for (value, resource) in bindings.resources() {
            match resource {
                BoundResource::Host(bytes) => {
                    host_state.insert(*value, bytes.clone());
                }
                BoundResource::Resident(resource) => {
                    resident_state.insert(*value, resource.clone());
                }
            }
        }
        if !host_state.is_empty() && !resident_state.is_empty() {
            return Err(invalid_module(
                "WGPU artifact submission cannot mix host and resident resources",
            ));
        }
        let result = if resident_state.is_empty() {
            self.execute(host_state)
        } else {
            self.execute_resident(resident_state)
        };
        Ok(Box::new(ReadySubmission {
            result: Some(result),
        }))
    }
}

impl WgpuArtifactInstance {
    fn execute(
        &self,
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Completion, BackendError> {
        let config = DispatchConfig::default();
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for module in &self.modules {
            let plan = BindingPlan::build(&module.program)?;
            let mut inputs = Vec::with_capacity(module.input_slots.len());
            for slot in &module.input_slots {
                let value = self.value_for_buffer(&slot.name)?;
                match state.get(&value) {
                    Some(bytes) => inputs.push(bytes.as_slice()),
                    None if !slot.required => inputs.push(&[]),
                    None => {
                        return Err(invalid_module(&format!(
                            "canonical artifact value {} for target binding `{}` is unbound",
                            value.0, slot.name
                        )));
                    }
                }
            }
            let dispatched = match module.pipeline.dispatch_borrowed_timed(&inputs, &config) {
                Err(_) if self.lost.load(Ordering::Acquire) => {
                    return Err(device_lost_error(&self.device));
                }
                result => result?,
            };
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            for binding in &plan.bindings {
                let Some(output_index) = binding.output_index else {
                    continue;
                };
                let buffer = &module.program.buffers()[binding.buffer_index];
                let value = self.value_for_buffer(buffer.name())?;
                let bytes = dispatched.outputs.get(output_index).ok_or_else(|| {
                    invalid_module(&format!(
                        "WGSL target module omitted output {output_index} for Program buffer `{}`",
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
            device_ns: has_device_timing.then_some(device_ns),
        })
    }
    fn execute_resident(
        &self,
        resources: BTreeMap<ArtifactValueId, vyre_driver::Resource>,
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
            let value = self.value_for_buffer(name)?;
            let resource = resources.get(&value).ok_or_else(|| {
                invalid_module(&format!(
                    "canonical artifact value {} for resident target binding `{name}` is unbound",
                    value.0
                ))
            })?;
            ordered.push(resource.clone());
        }
        let dispatched = module
            .pipeline
            .dispatch_persistent_handles_timed(&ordered, &DispatchConfig::default())?;
        let plan = BindingPlan::build(&module.program)?;
        let mut output_state = BTreeMap::<ArtifactValueId, Vec<u8>>::new();
        for binding in &plan.bindings {
            let Some(output_index) = binding.output_index else {
                continue;
            };
            let buffer = &module.program.buffers()[binding.buffer_index];
            let value = self.value_for_buffer(buffer.name())?;
            let bytes = dispatched.outputs.get(output_index).ok_or_else(|| {
                invalid_module(&format!(
                    "WGPU resident target module omitted output {output_index} for Program buffer `{}`",
                    buffer.name()
                ))
            })?;
            output_state.insert(value, bytes.clone());
        }
        let outputs = self.project_values(&output_state, &self.outputs, "produce")?;
        let retained = self.project_values(&output_state, &self.retained, "preserve")?;
        Ok(Completion {
            artifact: self.artifact,
            outputs,
            retained,
            device_ns: dispatched.device_ns,
        })
    }

    fn project_values(
        &self,
        state: &BTreeMap<ArtifactValueId, Vec<u8>>,
        values: &BTreeSet<ArtifactValueId>,
        action: &str,
    ) -> Result<BTreeMap<ArtifactValueId, Vec<u8>>, BackendError> {
        values
            .iter()
            .map(|value| {
                state
                    .get(value)
                    .cloned()
                    .map(|bytes| (*value, bytes))
                    .ok_or_else(|| {
                        invalid_module(&format!(
                            "selected execution did not {action} canonical value {}",
                            value.0
                        ))
                    })
            })
            .collect()
    }

    fn value_for_buffer(&self, name: &str) -> Result<ArtifactValueId, BackendError> {
        self.values.get(name).copied().ok_or_else(|| {
            invalid_module(&format!(
                "Program buffer `{name}` is absent from the canonical artifact ABI"
            ))
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

struct ReadySubmission {
    result: Option<Result<Completion, BackendError>>,
}

impl Submission for ReadySubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.result
            .take()
            .ok_or_else(|| invalid_module("each Submission completion may be consumed only once"))?
    }
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    let backend = WgpuBackend::acquire()?;
    let format =
        TargetPayloadFormat::new("wgsl", WGPU_TARGET_FORMAT_VERSION).map_err(compile_error)?;
    let generation = ResidentOwner::new()?.get();
    let device = backend.adapter_name.to_string();
    let lost = Arc::clone(&backend.device_lost);
    Ok(Box::new(WgpuMaterializer {
        backend,
        descriptor: WgpuDevice {
            identity: DeviceIdentity {
                backend: WGPU_BACKEND_ID,
                device,
                generation,
            },
            format,
            lost,
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
        backend: WGPU_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: runtime recovery must receive a stable device-loss class, never text to parse.
    #[test]
    fn lost_instance_submission_is_structured() {
        let digest = Digest([7; 32]);
        let instance = WgpuArtifactInstance {
            artifact: digest,
            payload: Digest([8; 32]),
            device: DeviceIdentity {
                backend: WGPU_BACKEND_ID,
                device: "fault-injection".to_string(),
                generation: 11,
            },
            lost: Arc::new(AtomicBool::new(true)),
            modules: Vec::new(),
            values: BTreeMap::new(),
            outputs: BTreeSet::new(),
            retained: BTreeSet::new(),
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
