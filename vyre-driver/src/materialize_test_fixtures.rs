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
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileRequest, DeviceFacts, Digest,
    ExternalFacts, SearchBudget, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
    TargetProfile, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
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
        1_000_000,
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

/// One entry point over the fixture launch geometry.
pub(super) fn entry_point(
    name: &str,
    node: ArtifactNodeId,
    resource_bindings: Vec<TargetResourceBinding>,
) -> TargetEntryPoint {
    entry_point_with_geometry(name, node, [32, 1, 1], [1, 1, 1], 0, resource_bindings)
}

/// One entry point over a stated launch geometry.
pub(super) fn entry_point_with_geometry(
    name: &str,
    node: ArtifactNodeId,
    workgroup_size: [u32; 3],
    grid_size: [u32; 3],
    dynamic_shared_bytes: u32,
    resource_bindings: Vec<TargetResourceBinding>,
) -> TargetEntryPoint {
    TargetEntryPoint {
        name: name.into(),
        node,
        workgroup_size,
        grid_size,
        dynamic_shared_bytes,
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
