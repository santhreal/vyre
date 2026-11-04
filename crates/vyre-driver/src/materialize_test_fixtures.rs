//! Fixtures the materialize test modules share.
//!
//! Both test modules compile one graph with the same budget, wrap it in a
//! payload with the same format, profile and geometry, and read it back through
//! the same instance core. Each carried its own copy of that setup, so the two
//! drifted: a change to the fixture device, the search budget or the payload
//! shape reached whichever file the author had open. The setup lives here once,
//! and a test states only what it varies.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, DataType, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileObjective, CompileRequest,
    DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, TargetEntryPoint,
    TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};

use crate::materialize::{InstanceCore, NEUTRAL_MESSAGES};
use crate::{BackendError, DeviceIdentity};

/// Payload format the fixture artifacts declare.
pub(super) fn test_format() -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", 1).unwrap()
}

/// Target profile the fixture payloads declare.
pub(super) fn test_profile() -> TargetProfile {
    TargetProfile::new("test.target-binary", 1, [32, 1, 1], 32, 1024, 0).unwrap()
}

/// Device identity the fixture instances report.
pub(super) fn test_device() -> DeviceIdentity {
    DeviceIdentity {
        backend: "test",
        device: "test-device".into(),
        generation: 1,
    }
}

/// Instance core over `artifact` and `payload`, with one module slot per fusion.
pub(super) fn test_instance_core(
    artifact: &Artifact,
    payload: &TargetPayload,
) -> Result<InstanceCore, BackendError> {
    InstanceCore::new_with_module_slots(
        artifact,
        payload,
        test_device(),
        NEUTRAL_MESSAGES,
        vec![BTreeMap::new(); artifact.fusion().len()],
    )
}

/// A `u32` contract of 32 elements with the given access and lifetime.
pub(super) fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(32)],
        access,
        lifetime,
    }
}

/// Compile `graph` with the fixture search budget and facts.
pub(super) fn compile_graph(graph: ProgramGraph) -> Artifact {
    compile_graph_with_search(graph, 128)
}

/// Compile `graph` with `search` candidates, for a test that bounds the search.
pub(super) fn compile_graph_with_search(graph: ProgramGraph, search: u32) -> Artifact {
    compile_request(graph, test_facts(), search)
}

/// Compile `graph` against `facts`, for a test that states external state.
pub(super) fn compile_graph_with_facts(graph: ProgramGraph, facts: ExternalFacts) -> Artifact {
    compile_request(graph, facts, 128)
}

/// External facts naming no resident value.
fn test_facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0; 32]), BTreeMap::new())
}

fn compile_request(graph: ProgramGraph, facts: ExternalFacts, search: u32) -> Artifact {
    let request = CompileRequest::new(
        graph,
        facts,
        DeviceFacts::unknown(),
        SearchBudget::new(search, 1_000_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .unwrap();
    compile(&request).expect("compilation must succeed")
}

/// One binding of `resource` at `group` and `slot`.
pub(super) fn binding(
    resource: ArtifactValueId,
    group: u32,
    slot: u32,
    memory: TargetResourceMemory,
    access: TargetResourceAccess,
) -> TargetResourceBinding {
    TargetResourceBinding {
        resource,
        group,
        slot,
        memory,
        access,
    }
}

/// Bindings of `values` in group 0, one slot each in the order given.
///
/// Every fixture payload binds its resources in one group at consecutive
/// slots, so the group and the slot are derived here and a test states only
/// the value and its access.
pub(super) fn global_bindings(
    values: &[(ArtifactValueId, TargetResourceAccess)],
) -> Vec<TargetResourceBinding> {
    values
        .iter()
        .enumerate()
        .map(|(slot, &(resource, access))| {
            binding(
                resource,
                0,
                u32::try_from(slot).unwrap(),
                TargetResourceMemory::Global,
                access,
            )
        })
        .collect()
}

/// One entry point over the launch geometry the artifact recorded.
///
/// The geometry is read out of the record rather than stated by the caller. A
/// payload states what the device launches, and admission accepts only the
/// geometry the compiler selected, so a stated shape here would be a fixture
/// that can never be attached to the artifact it names.
pub(super) fn entry_point(
    artifact: &Artifact,
    name: &str,
    node: ArtifactNodeId,
    resource_bindings: Vec<TargetResourceBinding>,
) -> TargetEntryPoint {
    let selected = artifact
        .geometry()
        .iter()
        .find(|record| record.node == node)
        .expect("the fixture artifact records geometry for every node it carries");
    TargetEntryPoint {
        name: name.to_string(),
        node,
        workgroup_size: selected.workgroup_size,
        grid_size: selected.grid,
        dynamic_shared_bytes: selected.dynamic_shared_bytes,
        resource_bindings,
    }
}

/// The payload `artifact` and `entries` describe.
///
/// # Panics
///
/// Panics when the payload does not validate. A test that expects a refusal, or
/// states why the payload must be accepted, calls [`try_payload`].
pub(super) fn test_payload(artifact: &Artifact, entries: Vec<TargetEntryPoint>) -> TargetPayload {
    try_payload(artifact, entries).unwrap()
}

/// The payload `artifact` and `entries` describe, or the reason it is refused.
pub(super) fn try_payload(
    artifact: &Artifact,
    entries: Vec<TargetEntryPoint>,
) -> Result<TargetPayload, vyre_megakernel::CompileError> {
    TargetPayload::new(
        artifact,
        test_format(),
        test_profile(),
        entries,
        vec![1, 2, 3],
    )
}
/// The artifact `graph` compiles to, and a one-entry payload binding `values`.
///
/// Most materialize fixtures are a single-node graph launched by a single entry
/// point that binds every value in group 0. That shape is stated here so a test
/// states the graph and the bindings and nothing about the seam between them.
pub(super) fn single_entry(
    graph: ProgramGraph,
    values: &[(ArtifactValueId, TargetResourceAccess)],
) -> (Artifact, TargetPayload) {
    let artifact = compile_graph(graph);
    let payload = test_payload(
        &artifact,
        vec![entry_point(
            &artifact,
            "entry0",
            ArtifactNodeId(0),
            global_bindings(values),
        )],
    );
    (artifact, payload)
}
