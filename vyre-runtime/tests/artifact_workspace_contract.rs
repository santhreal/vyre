//! Runtime allocation and binding of the workspace an artifact recorded.
//!
//! WHY: a multi-entry artifact records one workspace region per value it
//! produces for itself, with the offset and byte count the selected schedule
//! assigned. Nothing read that plan: the runtime allocated whatever a caller
//! asked for and bound whatever a caller supplied, so a cross-entry value could
//! be sized by the caller, shared with an unrelated value, or replaced by a
//! buffer the compiler never planned for. These contracts pin the plan as the
//! only authority over that storage.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vyre_driver::materialize::{DeviceSpec, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, Device, DeviceIdentity, ResidentOwner, Resource, Submission,
    VyreBackend,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, ArtifactValueId, Digest, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};
use vyre_runtime::artifact_admission::ArtifactSession;

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

const FORMAT: &str = "workspace.target";
const MIDDLE: &str = "middle";
const OUTPUT: &str = "out";

/// A two-stage artifact: the first entry's output is the second entry's input.
///
/// The intermediate value is the whole point. It is produced inside the artifact
/// and read inside the artifact, so nothing outside owns it and the compiler
/// records a workspace region for it.
fn two_stage_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();
    let (_, produced) = graph
        .add_node(
            "first",
            Program::wrapped(
                vec![BufferDecl::output(OUTPUT, 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store(OUTPUT, Expr::u32(0), Expr::u32(1))],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: OUTPUT.into(),
                name: MIDDLE.into(),
                contract: value(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("the workspace fixture must accept its producer");
    let middle = *produced
        .first()
        .expect("the producer declares one output value");
    graph
        .add_node(
            "second",
            Program::wrapped(
                vec![
                    BufferDecl::storage(MIDDLE, 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                    BufferDecl::output(OUTPUT, 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    OUTPUT,
                    Expr::u32(0),
                    Expr::load(MIDDLE, Expr::u32(0)),
                )],
            ),
            vec![GraphInput {
                buffer: MIDDLE.into(),
                value: middle,
                contract: value(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: OUTPUT.into(),
                name: OUTPUT.into(),
                contract: value(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("the workspace fixture must accept its consumer");
    artifact_fixtures::compile_graph(graph, 0)
}

fn value(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(1)],
        access,
        lifetime,
    }
}

fn profile() -> TargetProfile {
    TargetProfile::new(FORMAT, 1, [64, 1, 1], 64, 1_024, 0).expect("fixture profile must be valid")
}

/// A payload with one entry per recorded node, binding that entry's own values.
///
/// The geometry is read out of the artifact: admission accepts only the
/// geometry the compiler selected, so a stated shape would never seal.
fn two_stage_payload(artifact: &Artifact) -> TargetPayload {
    let entries = artifact
        .abi()
        .entries
        .iter()
        .map(|entry| {
            let recorded = artifact
                .geometry()
                .iter()
                .find(|record| record.node == entry.node)
                .expect("the artifact records geometry for every entry");
            let resource_bindings = entry
                .inputs
                .iter()
                .chain(entry.outputs.iter())
                .enumerate()
                .map(|(slot, resource)| TargetResourceBinding {
                    resource: *resource,
                    group: 0,
                    slot: u32::try_from(slot).expect("fixture entries bind few resources"),
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                })
                .collect();
            TargetEntryPoint {
                name: format!("entry{}", entry.node.0),
                node: entry.node,
                workgroup_size: recorded.workgroup_size,
                grid_size: recorded.grid,
                dynamic_shared_bytes: recorded.dynamic_shared_bytes,
                resource_bindings,
            }
        })
        .collect();
    TargetPayload::new(
        artifact,
        TargetPayloadFormat::new(FORMAT, 1).expect("fixture format must be valid"),
        profile(),
        entries,
        vec![1, 2, 3, 4],
    )
    .expect("the two-stage fixture payload must seal")
}

/// A materializer that records every resident allocation and release.
struct WorkspaceMaterializer {
    device: MaterializerDevice,
    owner: ResidentOwner,
    next: AtomicU64,
    allocated: Mutex<Vec<usize>>,
    freed: Mutex<Vec<Resource>>,
}

impl WorkspaceMaterializer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            device: MaterializerDevice::acquire(DeviceSpec {
                backend: "workspace-artifact",
                device: "workspace-device".to_string(),
                format_extension: FORMAT,
                format_version: 1,
                profile: profile(),
            })
            .expect("the fixture device must acquire"),
            owner: ResidentOwner::new().expect("the process must mint a resident owner"),
            next: AtomicU64::new(0),
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        })
    }
}

impl ArtifactMaterializer for WorkspaceMaterializer {
    fn device(&self) -> &dyn Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        Ok(Box::new(WorkspaceInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.device.identity().clone(),
        }))
    }

    fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.allocated
            .lock()
            .expect("the fixture allocation log must not be poisoned")
            .push(byte_len);
        let id = self.next.fetch_add(1, Ordering::AcqRel);
        Ok(Resource::Resident(self.owner.handle(id)))
    }

    fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
        self.freed
            .lock()
            .expect("the fixture release log must not be poisoned")
            .push(resource);
        Ok(())
    }
}

struct WorkspaceInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
}

impl ArtifactInstance for WorkspaceInstance {
    fn artifact(&self) -> Digest {
        self.artifact
    }

    fn payload(&self) -> Digest {
        self.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    fn submit(&self, _bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        Ok(Box::new(WorkspaceSubmission(Some(Completion {
            artifact: self.artifact,
            outputs: BTreeMap::new(),
            retained: BTreeMap::new(),
            device_ns: None,
        }))))
    }
}

struct WorkspaceSubmission(Option<Completion>);

impl Submission for WorkspaceSubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.0.take().ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: consume the fixture submission once.".to_string(),
        })
    }
}

fn workspace_backend_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "raw Program backend".to_string(),
        backend: "workspace-artifact".to_string(),
    })
}

fn workspace_supported_ops() -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    static OPS: std::sync::LazyLock<std::collections::HashSet<vyre_foundation::ir::OpId>> =
        std::sync::LazyLock::new(std::collections::HashSet::new);
    &OPS
}

static WORKSPACE_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "workspace-artifact",
    target_id: vyre_foundation::operation::TargetId::expect_valid("workspace-artifact"),
    payload_format: None,
    reference_oracle: false,
    factory: workspace_backend_factory,
    supported_ops: workspace_supported_ops,
    semantic_operations: workspace_supported_ops,
    target_compiler: None,
    materializer: None,
};

/// A session over the two-stage artifact and the recording materializer.
fn session() -> (Artifact, Arc<WorkspaceMaterializer>, ArtifactSession) {
    let artifact = two_stage_artifact();
    assert!(
        !artifact.workspace().regions.is_empty(),
        "Fix: the fixture must record at least one workspace region, or every contract here is \
         vacuous."
    );
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope
        .attach_target_payload(two_stage_payload(&artifact))
        .expect("the fixture payload must attach");
    let materializer = WorkspaceMaterializer::new();
    let session = ArtifactSession::from_envelope_with_materializer(
        &WORKSPACE_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("the two-stage envelope must materialize");
    (artifact, materializer, session)
}

/// Canonical value the fixture artifact produces for itself.
fn workspace_value(artifact: &Artifact) -> ArtifactValueId {
    artifact
        .workspace()
        .regions
        .first()
        .expect("the fixture records a workspace region")
        .value
}

/// WHY: the plan states one region per cross-entry value, with the byte count
/// the schedule assigned. A runtime that rounded, merged, or padded an
/// allocation binds a buffer of a size nothing compiled against.
#[test]
fn the_runtime_allocates_exactly_the_recorded_workspace() {
    let (artifact, materializer, session) = session();

    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");

    let plan = artifact.workspace();
    assert_eq!(workspace.total_bytes(), plan.total_bytes);
    assert_eq!(
        materializer
            .allocated
            .lock()
            .expect("the allocation log must not be poisoned")
            .as_slice(),
        plan.regions
            .iter()
            .map(|region| usize::try_from(region.bytes).expect("fixture regions are small"))
            .collect::<Vec<_>>()
            .as_slice(),
        "one allocation per recorded region, of exactly the recorded byte count, in recorded order"
    );
    assert_eq!(workspace.regions().len(), plan.regions.len());
    for region in &plan.regions {
        assert!(
            workspace.owns(region.value),
            "the workspace must own every value the plan records"
        );
    }

    let allocated = workspace.regions().values().cloned().collect::<Vec<_>>();
    session
        .free_workspace(workspace)
        .expect("the workspace must release");
    assert_eq!(
        materializer
            .freed
            .lock()
            .expect("the release log must not be poisoned")
            .as_slice(),
        allocated.as_slice(),
        "every allocated region is released, and nothing else"
    );
}

/// WHY: the artifact allocated its own storage for the values its entries pass
/// between themselves. A caller buffer in that place is a wrong bind, not a
/// substitution, and silently preferring either side is how the compiler stops
/// owning cross-entry storage.
#[test]
fn a_caller_cannot_rebind_a_workspace_owned_value() {
    let (artifact, _materializer, session) = session();
    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");
    let owned = workspace_value(&artifact);
    let name = artifact
        .resources()
        .iter()
        .find(|resource| resource.value == owned)
        .expect("the workspace value is a canonical resource")
        .name
        .clone();
    let caller = session
        .allocate_resident(4)
        .expect("the fixture materializer must allocate");

    let error = session
        .resident_bindings_with_workspace(&workspace, [(name.as_str(), &caller)])
        .expect_err("a caller must not rebind a workspace-owned value");

    let text = error.to_string();
    assert!(
        text.contains("workspace-owned") && text.contains(&format!("canonical value {}", owned.0)),
        "the refusal must name the value and why it is refused; got `{text}`"
    );
}

/// WHY: binding over a workspace must still supply every value the entries
/// declare. The workspace covers what the artifact produces for itself and
/// nothing else, so a graph input or a public output left out is refused rather
/// than launched unbound.
#[test]
fn workspace_bindings_cover_the_workspace_and_demand_the_rest() {
    let (artifact, _materializer, session) = session();
    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");
    let owned = workspace_value(&artifact);
    let caller_values = artifact
        .resources()
        .iter()
        .filter(|resource| resource.value != owned)
        .map(|resource| resource.name.clone())
        .collect::<Vec<_>>();
    assert!(
        !caller_values.is_empty(),
        "Fix: the fixture must carry a caller-owned resource beside its workspace."
    );
    let caller = session
        .allocate_resident(4)
        .expect("the fixture materializer must allocate");

    let missing = session
        .resident_bindings_with_workspace(&workspace, [])
        .expect_err("a caller-owned value must not be defaulted");
    assert!(
        missing.to_string().contains("requires resident resource"),
        "the refusal must name the unbound entry resource; got `{missing}`"
    );

    let bound = session
        .resident_bindings_with_workspace(
            &workspace,
            caller_values
                .iter()
                .map(|name| (name.as_str(), &caller))
                .collect::<Vec<_>>(),
        )
        .expect("the workspace plus every caller-owned value must bind");

    assert_eq!(
        bound.resources().get(&owned),
        workspace
            .regions()
            .get(&owned)
            .map(|resource| BoundResource::Resident(resource.clone()))
            .as_ref(),
        "the workspace's own region must reach the binding set"
    );
    for name in &caller_values {
        let value = artifact
            .resources()
            .iter()
            .find(|resource| &resource.name == name)
            .expect("the caller value is a canonical resource")
            .value;
        assert_eq!(
            bound.resources().get(&value),
            Some(&BoundResource::Resident(caller.clone()))
        );
    }
}
