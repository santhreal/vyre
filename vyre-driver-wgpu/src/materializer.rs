use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_driver::materialize::{
    self, DeviceSpec, ExecutableModule, InstanceCore, InstanceMessages, MaterializedInstance,
    MaterializerDevice, ResidentInstance,
};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingPlan, BindingSet,
    CompiledPipeline, Device, DeviceIdentity, DispatchConfig, Resource, Submission,
    TimedDispatchResult,
};
use vyre_foundation::ir::Program;
use vyre_lower::TRAP_SIDECAR_NAME;
use vyre_megakernel::{Artifact, ArtifactValueId, TargetPayload, TargetResourceAccess};

use crate::descriptor_mapping::descriptor_bind_group;
use crate::pipeline::WgpuPipeline;
use crate::target_compiler::{
    WgpuTargetModule, WGPU_TARGET_FORMAT_VERSION, WGPU_TARGET_MODULE_SCHEMA_VERSION,
};
use crate::{WgpuBackend, WGPU_BACKEND_ID};

/// Resident-path rejection text. Both value-shaped rejections still say which
/// event failed, produce or preserve, and omit the lifetime word the host path
/// spells out; every other rejection is the neutral wording.
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
                    let mut input_slots = Vec::new();
                    for slot in &module.image.descriptor.bindings.slots {
                        let Some(group) = descriptor_bind_group(slot.memory_class) else {
                            continue;
                        };
                        if slot.name == TRAP_SIDECAR_NAME {
                            continue;
                        }
                        let canonical = module
                            .resource_bindings
                            .iter()
                            .find(|binding| binding.group == group && binding.slot == slot.slot)
                            .ok_or_else(|| {
                                materialize::invalid_module(&format!(
                                    "target binding `{}` at group {group}, slot {} has no canonical directional metadata",
                                    slot.name, slot.slot
                                ))
                            })?;
                        let buffer = program
                            .buffers()
                            .iter()
                            .find(|buffer| buffer.name() == slot.name)
                            .ok_or_else(|| {
                                materialize::invalid_module(&format!(
                                    "target binding `{}` has no selected Program buffer",
                                    slot.name
                                ))
                            })?;
                        if canonical.access != TargetResourceAccess::WriteOnly {
                            let expected_max = usize::try_from(buffer.count())
                                .ok()
                                .and_then(|count| count.checked_mul(buffer.element().min_bytes()))
                                .filter(|_| buffer.count() != 0);
                            input_slots.push(ArtifactInputSlot {
                                name: slot.name.clone(),
                                group,
                                slot: slot.slot,
                                expected_max,
                            });
                        }
                    }
                    let pipeline = self.backend.compile_pipeline(
                        &program,
                        &config,
                        Some(crate::pipeline::AuthenticatedTarget {
                            wgsl: &target.wgsl,
                            descriptor: &module.image.descriptor,
                            resource_bindings: &module.resource_bindings,
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
                .instance(artifact, payload, materialize::NEUTRAL_MESSAGES)?,
            lost: Arc::clone(&self.lost),
            modules,
        }))
    }
}

struct ArtifactInputSlot {
    name: String,
    group: u32,
    slot: u32,
    expected_max: Option<usize>,
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

    fn module_label(&self) -> &'static str {
        "WGSL target module"
    }

    fn gather<'state>(
        &self,
        module_index: usize,
        module: &Self::Module,
        _plan: &BindingPlan,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
        let mut inputs = Vec::with_capacity(module.input_slots.len());
        for slot in &module.input_slots {
            let value = self.core.value_for_module_slot(
                &self.core.module_inputs,
                module_index,
                slot.group,
                slot.slot,
                &slot.name,
            )?;
            let bytes = state.get(&value).ok_or_else(|| {
                materialize::invalid_module(&format!(
                    "canonical artifact value {} for target binding `{}` is unbound",
                    value.0, slot.name
                ))
            })?;
            if slot
                .expected_max
                .is_some_and(|expected| bytes.len() > expected)
            {
                let canonical_name = self
                    .core
                    .values
                    .iter()
                    .find_map(|(name, candidate)| (*candidate == value).then_some(name.as_str()))
                    .unwrap_or("<unnamed>");
                return Err(materialize::invalid_module(&format!(
                    "canonical artifact value {} (`{canonical_name}`) supplied {} byte(s) to target binding `{}`, whose static limit is {} byte(s)",
                    value.0,
                    bytes.len(),
                    slot.name,
                    slot.expected_max.unwrap_or_default(),
                )));
            }
            inputs.push(bytes.as_slice());
        }
        Ok(inputs)
    }

    fn dispatch(
        &self,
        module: &Self::Module,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        match module.pipeline.dispatch_borrowed_timed(inputs, config) {
            Err(_) if self.lost.load(Ordering::Acquire) => {
                Err(device_lost_error(&self.core.device))
            }
            result => result,
        }
    }
}

impl ResidentInstance for WgpuArtifactInstance {
    vyre_driver::resident_pipeline_launch!();

    fn resident_module_label(&self) -> &'static str {
        "WGPU resident target module"
    }

    fn resident_messages(&self) -> &InstanceMessages {
        &RESIDENT_MESSAGES
    }

    /// Resolve resident handles into the order the emitted target module
    /// declares, which is the order its pipeline reports rather than the
    /// binding plan's.
    fn ordered_resident(
        &self,
        module_index: usize,
        module: &Self::Module,
        _plan: &BindingPlan,
        resources: &BTreeMap<ArtifactValueId, Resource>,
    ) -> Result<Vec<Resource>, BackendError> {
        self.core.ordered_resident_resources(
            module_index,
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
    let device = backend.adapter_name.to_string();
    let lost = Arc::clone(&backend.device_lost);
    Ok(Box::new(WgpuMaterializer {
        backend,
        descriptor: MaterializerDevice::acquire_revocable(
            DeviceSpec {
                backend: WGPU_BACKEND_ID,
                device,
                format_extension: "wgsl",
                format_version: WGPU_TARGET_FORMAT_VERSION,
                profile: crate::target_compiler::target_profile()?,
            },
            Arc::clone(&lost),
        )?,
        lost,
    }))
}

pub(crate) fn materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    materializer_for_backend(WgpuBackend::acquire()?)
}

// Inline: covers `WgpuArtifactInstance`, `core`, `materialize`, `modules`, which no integration
// test can name.
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
                module_inputs: Vec::new(),
                module_outputs: Vec::new(),
                module_resources: Vec::new(),
                module_named_resources: Vec::new(),
                module_buffer_slots: Vec::new(),
                retained_predecessors: BTreeMap::new(),
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

    /// WHY: `RESIDENT_MESSAGES` is a crate-private const, so no integration test
    /// can name it, and the distinctness that `vyre-driver` pins for the neutral
    /// record has to be asserted beside every record that overrides one. A record
    /// whose two value-shaped rejections read alike turns an unproduced output and
    /// an unpreserved retained value into one sentence, which is the defect CUDA
    /// shipped until its override was deleted. The record is destructured, so a
    /// sixth rejection stops this file compiling until it is compared too.
    ///
    /// Does not catch: a wording that is distinct from its siblings and still
    /// wrong about which event happened. The produce and preserve assertions are
    /// what cover that, and only for these two rejections.
    #[test]
    fn the_resident_record_keeps_five_distinct_rejections() {
        let InstanceMessages {
            foreign_artifact,
            unmapped_buffer,
            missing_output_value,
            missing_retained_value,
            completion_consumed,
        } = RESIDENT_MESSAGES;
        let value = ArtifactValueId(9);
        let texts = [
            ("foreign_artifact", foreign_artifact().to_string()),
            ("unmapped_buffer", unmapped_buffer("scratch").to_string()),
            (
                "missing_output_value",
                missing_output_value(value).to_string(),
            ),
            (
                "missing_retained_value",
                missing_retained_value(value).to_string(),
            ),
            ("completion_consumed", completion_consumed().to_string()),
        ];

        for (index, (left, left_text)) in texts.iter().enumerate() {
            for (right, right_text) in texts.iter().skip(index + 1) {
                assert_ne!(
                    left_text, right_text,
                    "Fix: resident rejections `{left}` and `{right}` render the same sentence, so a reader cannot tell which contract broke."
                );
            }
        }
        for (field, event) in [
            ("missing_output_value", "produce"),
            ("missing_retained_value", "preserve"),
        ] {
            let text = texts
                .iter()
                .find(|(name, _)| *name == field)
                .map(|(_, text)| text.as_str())
                .expect("every rejection this assertion names is rendered above");
            assert!(
                text.contains(event) && text.contains("9"),
                "Fix: resident `{field}` reads `{text}`, which must name the `{event}` event and value 9."
            );
        }
    }
}
