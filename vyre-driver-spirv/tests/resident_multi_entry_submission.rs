//! Resident submission of a multi-entry artifact, over a real compiled payload.
//!
//! WHY: the shared resident path refused every artifact with more than one
//! entry point, so a compiler that had already selected a two-stage plan, frozen
//! both launches, and recorded which value passes between them could only run it
//! by routing the intermediate value through host memory. The refusal was the
//! only reason multi-entry residency did not work, and nothing proved the loop
//! that replaced it submits every entry, in the recorded order, over each
//! entry's own resources.
//!
//! This exercises the neutral loop in `vyre_driver::materialize` through the
//! SPIR-V backend, which can produce a genuine two-entry payload; nothing here
//! is SPIR-V specific. The launch itself is recorded rather than executed, so no
//! device is required.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use vyre_driver::materialize::{
    self, ExecutableModule, InstanceCore, MaterializedInstance, MaterializerTarget,
    ResidentInstance, NEUTRAL_MESSAGES,
};
use vyre_driver::{
    BackendError, DeviceIdentity, DispatchConfig, LaunchDirective, ResidentOwner, Resource,
    TimedDispatchResult,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{Artifact, ArtifactValueId, TargetPayload};

mod target_artifacts;
use target_artifacts::spirv;

/// One admitted module, as a backend records it.
struct RecordedModule {
    index: usize,
    program: Arc<Program>,
    config: DispatchConfig,
}

impl ExecutableModule for RecordedModule {
    fn program(&self) -> &Program {
        &self.program
    }

    fn config(&self) -> &DispatchConfig {
        &self.config
    }
}

/// What one resident launch was handed.
#[derive(Debug, PartialEq, Eq)]
struct RecordedLaunch {
    module: usize,
    resources: Vec<Resource>,
    launch: LaunchDirective,
}

/// An instance whose launch records its arguments instead of running them.
struct RecordingInstance {
    core: InstanceCore,
    modules: Vec<RecordedModule>,
    launches: RefCell<Vec<RecordedLaunch>>,
}

impl MaterializedInstance for RecordingInstance {
    type Module = RecordedModule;

    fn core(&self) -> &InstanceCore {
        &self.core
    }

    fn modules(&self) -> &[Self::Module] {
        &self.modules
    }

    fn module_label(&self) -> &'static str {
        "recording target module"
    }

    fn dispatch(
        &self,
        _module: &Self::Module,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        Err(BackendError::UnsupportedFeature {
            name: "host dispatch".to_string(),
            backend: "recording".to_string(),
        })
    }
}

impl ResidentInstance for RecordingInstance {
    fn resident_module_label(&self) -> &'static str {
        "recording resident target module"
    }

    fn launch_resident(
        &self,
        module: &Self::Module,
        ordered: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let launch = config.launch.ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: an admitted module must carry the frozen launch.".to_string(),
        })?;
        self.launches.borrow_mut().push(RecordedLaunch {
            module: module.index,
            resources: ordered.to_vec(),
            launch,
        });
        let outputs = self.core.module_outputs[module.index]
            .iter()
            .map(|value| vec![u8::try_from(value.0 % 251).unwrap_or(0), 0, 0, 0])
            .collect();
        Ok(TimedDispatchResult::device_timed(
            outputs,
            17,
            Some(11 * (module.index as u64 + 1)),
        ))
    }
}

fn device() -> DeviceIdentity {
    DeviceIdentity {
        backend: "recording",
        device: "recording-device".to_string(),
        generation: 1,
    }
}

/// A recording instance over the two-stage artifact and its real payload.
fn two_stage_instance() -> (Artifact, TargetPayload, RecordingInstance) {
    let (artifact, payload) = spirv().compiled_two_stage();
    let admitted = materialize::admit(
        &artifact,
        &payload,
        MaterializerTarget {
            backend_id: vyre_driver_spirv::SPIRV_BACKEND_ID,
            format: payload.format(),
            profile: payload.profile(),
        },
    )
    .expect("the two-stage payload must admit");
    assert!(
        admitted.len() > 1,
        "Fix: the fixture must admit more than one module, or nothing here is a multi-entry proof."
    );
    let core = InstanceCore::new(&artifact, &payload, device(), NEUTRAL_MESSAGES)
        .expect("the two-stage payload must project an instance core");
    let modules = admitted
        .into_iter()
        .enumerate()
        .map(|(index, module)| RecordedModule {
            index,
            program: module.program,
            config: module.config,
        })
        .collect();
    let instance = RecordingInstance {
        core,
        modules,
        launches: RefCell::new(Vec::new()),
    };
    (artifact, payload, instance)
}

/// Every canonical resource bound to a distinct resident handle.
fn resident_resources(artifact: &Artifact) -> BTreeMap<ArtifactValueId, Resource> {
    let owner = ResidentOwner::new().expect("the process must mint a resident owner");
    artifact
        .resources()
        .iter()
        .enumerate()
        .map(|(index, resource)| {
            (
                resource.value,
                Resource::Resident(owner.handle(index as u64)),
            )
        })
        .collect()
}

/// WHY: a two-entry artifact used to be refused outright. Now every entry must
/// launch, once, in the order the artifact recorded, and the completion must
/// carry the final output the second entry produced from the first entry's
/// value. An implementation that launched only the first module, or launched
/// them in bundle order, satisfies neither.
#[test]
fn every_recorded_entry_launches_once_in_plan_order() {
    let (artifact, _payload, instance) = two_stage_instance();
    let resources = resident_resources(&artifact);

    let completion = instance
        .execute_resident(&resources)
        .expect("a multi-entry resident submission must run every entry");

    let launches = instance.launches.borrow();
    assert_eq!(
        launches
            .iter()
            .map(|launch| launch.module)
            .collect::<Vec<_>>(),
        (0..instance.modules.len()).collect::<Vec<_>>(),
        "every module launches exactly once, in recorded plan order"
    );
    for launch in launches.iter() {
        let recorded = artifact
            .geometry()
            .iter()
            .find(|record| record.node == artifact.abi().entries[launch.module].node)
            .expect("the artifact records geometry for every entry");
        assert_eq!(launch.launch.grid(), recorded.grid);
        assert_eq!(launch.launch.workgroup(), recorded.workgroup_size);
        assert!(
            !launch.resources.is_empty(),
            "a resident launch takes the resources its own entry declares"
        );
    }
    assert_eq!(
        completion.device_ns,
        Some(11 + 22),
        "device time is the sum over the entries that reported one"
    );
    assert!(
        !completion.outputs.is_empty(),
        "the last entry's output must reach the completion"
    );
}

/// WHY: each entry reads the resources its own entry ABI declares. Resolving
/// every module against entry 0's projection is the defect the module index
/// closes: it binds the first entry's buffers to the second entry's launch, and
/// a launch handed the wrong handles reads the wrong device memory without
/// failing.
#[test]
fn each_entry_reads_the_resources_its_own_abi_declares() {
    let (artifact, _payload, instance) = two_stage_instance();
    let resources = resident_resources(&artifact);

    instance
        .execute_resident(&resources)
        .expect("a multi-entry resident submission must run every entry");

    let launches = instance.launches.borrow();
    for launch in launches.iter() {
        let entry = &artifact.abi().entries[launch.module];
        let declared = entry
            .inputs
            .iter()
            .chain(entry.outputs.iter())
            .map(|value| {
                resources
                    .get(value)
                    .expect("every declared value is bound")
                    .clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            launch.resources, declared,
            "entry {} must be handed exactly the values its own ABI declares, in that order",
            launch.module
        );
    }
    assert_ne!(
        artifact.abi().entries[0].outputs,
        artifact.abi().entries[1].outputs,
        "Fix: the fixture entries must declare different values, or resolving every module \
         against entry 0 would look correct here."
    );
}

/// WHY: an unbound value must be refused, not defaulted. Dropping the value the
/// second entry reads is exactly the case a host round trip used to hide.
#[test]
fn a_missing_resource_for_a_later_entry_is_refused() {
    let (artifact, _payload, instance) = two_stage_instance();
    let mut resources = resident_resources(&artifact);
    let entries = &artifact.abi().entries;
    let earlier = entries[..entries.len() - 1]
        .iter()
        .flat_map(|entry| entry.inputs.iter().chain(entry.outputs.iter()))
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let last = *entries
        .last()
        .expect("the fixture carries entries")
        .outputs
        .iter()
        .find(|value| !earlier.contains(value))
        .expect("the last entry declares a value no earlier entry touches");
    resources.remove(&last);

    let error = instance
        .execute_resident(&resources)
        .expect_err("an unbound resident value must be refused");

    match error {
        BackendError::InvalidProgram { fix } => assert!(
            fix.contains(&format!("bind canonical artifact value {}", last.0)),
            "the refusal must name the unbound value; got `{fix}`"
        ),
        other => panic!("expected InvalidProgram, got {other:?}"),
    }
}
