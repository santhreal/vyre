//! The canonical compiler artifact fixtures, owned by the workspace.
//!
//! Every crate downstream of `vyre-megakernel` that tests admission, packaging,
//! or artifact reporting needs the same input: a one-node graph over a few
//! external values, compiled under fixed facts and a bounded search. Four crates
//! each carried their own copy of that setup, and `vyre-megakernel` and
//! `vyre-runtime` carried it byte-identically because their assertions compare
//! digests that only agree while the two copies agree.
//!
//! The copies also had to keep restating which buffer declaration a value
//! contract implies, so a fixture whose declared element count disagreed with its
//! contract shape was accepted by whichever crate wrote it and rejected by the
//! next one to copy it. `decl_for` states that mapping once.
//!
//! Shared the same way as `crate::preferred_dispatch_backend_contract`: each
//! consumer calls into this one owner from its own suite.

use std::collections::BTreeMap;

use crate::graph_shapes;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, GraphInput, GraphOutput,
    Node, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::allocation::DeviceSlot;
use vyre_megakernel::mesh::{CollectiveSupport, MeshAxis, MeshDevice, MeshFacts, MeshLink};
use vyre_megakernel::{
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileObjective, CompileRequest,
    DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, TargetEntryPoint,
    TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory, ValidatedCompileRequest,
};

/// A single-dimension external value contract.
pub fn contract(
    dtype: DataType,
    count: u32,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    crate::graph_values::typed_vector(count, dtype, access, lifetime)
}

/// Element count a fixture value's contract declares.
fn declared_count(name: &str, contract: &ValueContract) -> u32 {
    match contract.shape.as_slice() {
        [ShapeDim::Known(count)] => {
            u32::try_from(*count)
                .expect("fixture element count must fit u32. Fix: declare a count within u32 in the caller's ValueContract")
        }
        shape => panic!("fixture value `{name}` must declare one known dimension, got {shape:?}"),
    }
}

/// Buffer declaration a fixture value's contract implies.
///
/// The mapping is the contract rather than a convenience. A read-only value is a
/// read buffer, a write-only value is an output buffer the caller reads back, and
/// a read-write value is retained storage. Uniform and workgroup memory are not
/// graph externals, so those contracts are rejected here instead of silently
/// becoming global storage.
fn decl_for(name: &str, slot: u32, contract: &ValueContract) -> BufferDecl {
    let decl = match &contract.access {
        BufferAccess::ReadOnly => BufferDecl::read(name, slot, contract.dtype.clone()),
        BufferAccess::WriteOnly => BufferDecl::output(name, slot, contract.dtype.clone()),
        BufferAccess::ReadWrite => BufferDecl::read_write(name, slot, contract.dtype.clone()),
        access => {
            panic!("fixture value `{name}` declares {access:?}, which is not a graph external")
        }
    };
    decl.with_count(declared_count(name, contract))
}

/// One node that reads one caller-supplied value of `count` elements, wired.
///
/// [`graph_over`] leaves ports unwired, so its logical region has a domain of one
/// point and a placement has nothing to cut. A fixture that wants a domain states
/// it through the port contract, which is where the logical stage reads it.
pub fn wired_input_graph(count: u32) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let value = contract(
        DataType::U32,
        count,
        BufferAccess::ReadOnly,
        ValueLifetime::Invocation,
    );
    let input = graph
        .add_external_value("input", value.clone())
        .expect("fixture external value must be valid. Fix: correct the contract this fixture declares for `input`");
    graph
        .add_node(
            "entry",
            Program::wrapped(vec![decl_for("input", 0, &value)], [8, 1, 1], Vec::new()),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: value,
            }],
            Vec::new(),
        )
        .expect("fixture node must be valid. Fix: correct the buffer declaration or the graph input this fixture wires to `entry`");
    graph
}

/// One node that all-reduces a caller-visible value of `count` elements.
///
/// A placement records a transfer only where the program states an exchange, so
/// a fixture that wants routed communication states the collective itself. The
/// reduced value is read-write for the invocation, which keeps the region a
/// reduction rather than retained state.
pub fn collective_output_graph(count: u32) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let reduced = contract(
        DataType::U32,
        count,
        BufferAccess::ReadWrite,
        ValueLifetime::Invocation,
    );
    graph
        .add_node(
            "reduce",
            Program::wrapped(
                vec![decl_for("out", 0, &reduced)],
                [8, 1, 1],
                vec![Node::AllReduce {
                    buffer: "out".into(),
                    op: CollectiveOp::Sum,
                    group: CommGroup::WORLD,
                }],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: reduced,
                retained_successor_of: None,
            }],
        )
        .expect("fixture collective node must be valid. Fix: correct the reduced contract or the collective output this fixture declares");
    graph
}

/// One node that atomically updates a caller-visible value of `count` elements.
pub fn atomic_output_graph(count: u32) -> ProgramGraph {
    graph_shapes::atomic_output_graph(count, [8, 1, 1])
}

/// One node that reads the caller-supplied value it also writes.
pub fn in_place_input_graph(count: u32) -> ProgramGraph {
    graph_shapes::in_place_input_graph(count, [8, 1, 1])
}

/// Two nodes over one-element values, the second consuming the first.
///
/// One point per region is deliberate: no axis has a bound to cut, so the only
/// placement that uses more than one device is the one that runs consecutive
/// regions on consecutive devices.
pub fn chained_graph() -> ProgramGraph {
    graph_shapes::chained_graph(1, [1, 1, 1])
}

/// One-node graph over `values`, bound to Program slots in the order given.
pub fn graph_over(
    node: &str,
    workgroup_size: [u32; 3],
    values: &[(&str, ValueContract)],
) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let mut decls = Vec::with_capacity(values.len());
    for (slot, (name, contract)) in values.iter().enumerate() {
        let slot = u32::try_from(slot)
            .expect("fixture slot must fit u32. Fix: pass fewer values to graph_over");
        graph
            .add_external_value(*name, contract.clone())
            .expect("fixture resource must be valid. Fix: correct the contract the caller paired with this value name");
        decls.push(decl_for(name, slot, contract));
    }
    graph
        .add_node(
            node,
            Program::wrapped(decls, workgroup_size, Vec::new()),
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture node must be valid. Fix: correct the declarations graph_over builds from the caller's values");
    graph
}

/// One node over a read-only parameter block and a write-only result.
///
/// The ahead-of-time suites compile this graph to compare artifact identities,
/// so the two element counts, the two access modes and the workgroup size are
/// part of what those comparisons mean. Two copies of it drift into two graphs
/// that no longer produce the same identity, and the assertions keep passing
/// against different subjects.
pub fn params_and_out_graph() -> ProgramGraph {
    graph_over(
        "main",
        [64, 1, 1],
        &[
            (
                "params",
                contract(
                    DataType::U32,
                    256,
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            ),
            (
                "out",
                contract(
                    DataType::U32,
                    64,
                    BufferAccess::WriteOnly,
                    ValueLifetime::Output,
                ),
            ),
        ],
    )
}

/// Compile a fixture graph under the facts and bounded search every fixture uses.
///
/// `facts_seed` fills the external facts digest, which participates in the
/// request digest, so two fixtures that differ only in seed compile to different
/// artifact identities.
pub fn compile_graph(graph: ProgramGraph, facts_seed: u8) -> Artifact {
    compile_placed(graph, facts_seed, None)
}

/// Compile a fixture graph against one authenticated device mesh.
///
/// The placement is a schedule decision, so a fixture that wants more than one
/// device supplies the mesh and lets selection cut the graph.
pub fn compile_graph_on_mesh(graph: ProgramGraph, facts_seed: u8, mesh: MeshFacts) -> Artifact {
    compile_placed(graph, facts_seed, Some(mesh))
}

/// Compile a fixture graph against one mesh under an objective that ranks the
/// bytes one device holds first.
///
/// A placement that spreads whole regions leaves one submission as long as it
/// was, so a latency objective keeps the single-device placement. A caller that
/// states peak memory is the caller a spread placement is for.
pub fn compile_graph_on_mesh_for_memory(
    graph: ProgramGraph,
    facts_seed: u8,
    mesh: MeshFacts,
) -> Artifact {
    let request = placed_request_with(
        graph,
        facts_seed,
        Some(mesh),
        CompileObjective::minimize_latency()
            .with_primary(ObjectiveMetric::PeakMemory)
            .with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    );
    compile(&request)
        .expect("fixture request must compile. Fix: correct the graph or raise the artifact-bytes bound this fixture sets")
}

/// The validated request one fixture graph compiles from, placed on `mesh`.
///
/// A case that expects the compile to fail needs the request rather than the
/// artifact, so the request is built here once and both paths use it.
pub fn mesh_request(graph: ProgramGraph, mesh: MeshFacts) -> ValidatedCompileRequest {
    placed_request(graph, 0, Some(mesh))
}

fn compile_placed(graph: ProgramGraph, facts_seed: u8, mesh: Option<MeshFacts>) -> Artifact {
    let request = placed_request(graph, facts_seed, mesh);
    compile(&request)
        .expect("fixture request must compile. Fix: correct the graph or the placement this fixture compiles it under")
}

fn placed_request(
    graph: ProgramGraph,
    facts_seed: u8,
    mesh: Option<MeshFacts>,
) -> ValidatedCompileRequest {
    placed_request_with(
        graph,
        facts_seed,
        mesh,
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
}

fn placed_request_with(
    graph: ProgramGraph,
    facts_seed: u8,
    mesh: Option<MeshFacts>,
    objective: CompileObjective,
) -> ValidatedCompileRequest {
    let collectives = graph
        .nodes()
        .iter()
        .any(|node| node.program.stats().distributed_collectives());
    let native_f16 = graph.nodes().iter().any(|node| node.program.stats().f16());
    let constant_identities = graph
        .values()
        .iter()
        .filter(|value| value.contract.lifetime == ValueLifetime::Constant)
        .map(|value| (value.id, Digest([facts_seed.wrapping_add(1); 32])))
        .collect();
    let mut facts = ExternalFacts::new(Digest([facts_seed; 32]), BTreeMap::new());
    facts.constant_identities = constant_identities;
    let request = CompileRequest::new(
        graph,
        facts,
        device_facts_for(collectives, native_f16),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        objective,
    );
    let request = match mesh {
        Some(mesh) => request.with_mesh(mesh),
        None => request,
    };
    request
        .validate()
        .expect("fixture request must validate. Fix: correct the graph, the device facts, or the mesh this fixture places it on")
}

/// Facts for the device a fixture graph is compiled against.
///
/// A graph that states a distributed collective needs a device that carries one,
/// and a graph that stores binary16 needs a device with native f16, so the
/// fixture states each capability exactly where a program uses it instead of
/// granting it to every fixture compile.
fn device_facts_for(collectives: bool, native_f16: bool) -> DeviceFacts {
    if !collectives && !native_f16 {
        return DeviceFacts::unknown();
    }
    DeviceFacts::new(
        BackendCapabilities {
            supports_distributed_collectives: collectives,
            has_native_f16: native_f16,
            ..BackendCapabilities::NONE
        },
        0,
    )
}

/// One mesh axis of `extent` coordinates.
pub fn mesh_axis(extent: u32) -> MeshAxis {
    MeshAxis {
        name: "device".to_owned(),
        extent,
    }
}

/// One mesh device at one coordinate, in its own failure domain.
pub fn mesh_device(slot: u16, coordinate: u32, memory_capacity_bytes: u64) -> MeshDevice {
    MeshDevice {
        slot: DeviceSlot(slot),
        coordinate: vec![coordinate],
        memory_capacity_bytes,
        failure_domain: u32::from(slot),
    }
}

/// One directed mesh link at the bandwidth and latency every fixture prices.
pub fn mesh_link(from: u16, to: u16) -> MeshLink {
    MeshLink {
        from: DeviceSlot(from),
        to: DeviceSlot(to),
        bandwidth_bytes_per_ns: 64,
        latency_ns: 1_000,
    }
}

/// A two-device mesh of one axis, symmetric links, and every exchange kind.
///
/// Both capacities are large enough that no fixture is refused for capacity, so
/// a capacity case states its own smaller figure.
pub fn two_device_mesh() -> MeshFacts {
    MeshFacts::new(
        vec![mesh_axis(2)],
        vec![mesh_device(0, 0, 1 << 30), mesh_device(1, 1, 1 << 30)],
        vec![mesh_link(0, 1), mesh_link(1, 0)],
        CollectiveSupport::ALL,
    )
    .expect("fixture mesh must authenticate. Fix: correct the device and link records this two-device mesh declares")
}

/// The canonical read-only single-input artifact.
///
/// `vyre-megakernel` and `vyre-runtime` both assert against this exact artifact,
/// including its digest, so it is one function rather than two copies that have
/// to stay identical for those assertions to keep meaning what they say.
pub fn single_input_graph(workgroup_size: [u32; 3]) -> ProgramGraph {
    graph_over(
        "entry",
        workgroup_size,
        &[(
            "input",
            contract(
                DataType::U32,
                8,
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        )],
    )
}

/// The canonical single-input artifact, compiled under the neutral fact seed.
///
/// Every consumer that admits, packages or reports on an artifact starts from
/// this one, so their digest assertions compare the same bytes.
pub fn neutral_artifact(workgroup_size: [u32; 3]) -> Artifact {
    compile_graph(single_input_graph(workgroup_size), 0)
}

/// The artifact of one whole `Program`, wrapped as the single node of a graph.
///
/// The graph fixtures above build a node's `Program` out of value contracts,
/// which is what an admission or packaging case varies. A cache-identity case
/// varies the `Program` itself, so the program is the argument here and every
/// buffer it declares becomes an external value carrying that buffer's own
/// element type, count and access: a fixture's graph contract cannot disagree
/// with the program it wraps.
pub fn artifact_for_program(program: Program) -> Artifact {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        graph
            .add_external_value(
                buffer.name(),
                contract(
                    buffer.element(),
                    buffer.count(),
                    buffer.access(),
                    ValueLifetime::Invocation,
                ),
            )
            .expect("fixture resource must be valid. Fix: correct the buffer declaration the program carries for this value");
    }
    graph
        .add_node("main", program, Vec::new(), Vec::new())
        .expect(
            "fixture node must be valid. Fix: correct the program this fixture wraps as `main`",
        );
    compile_graph(graph, 0)
}

/// The smallest artifact: one retained word, stored once.
///
/// Every pipeline-cache case that needs an artifact and does not care which one
/// uses this, so those cases compare the same bytes.
pub fn tiny_artifact() -> Artifact {
    artifact_for_program(Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    ))
}

/// The single-binding entry point the target payload fixtures attach.
///
/// Geometry is read out of the artifact rather than restated here. The envelope
/// admits only the geometry the compiler selected, so a hand-written shape would
/// be a fixture that can never be attached to the artifact it names.
pub fn entry_point(artifact: &Artifact) -> TargetEntryPoint {
    entry_over(
        artifact,
        "entry",
        ArtifactNodeId(0),
        vec![TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 3,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadOnly,
        }],
    )
}

/// One entry point over `bindings`, launched at the geometry the artifact
/// recorded for `node`.
///
/// Both suites that build payload entries read the launch out of the artifact
/// and then restate the same six fields. Naming and binding are what the cases
/// vary, so those are arguments and the rest is here once.
pub fn entry_over(
    artifact: &Artifact,
    name: &str,
    node: ArtifactNodeId,
    resource_bindings: Vec<TargetResourceBinding>,
) -> TargetEntryPoint {
    let launch = artifact
        .geometry()
        .iter()
        .find(|record| record.node == node)
        .unwrap_or_else(|| panic!("fixture artifact records no geometry for node {}", node.0));
    TargetEntryPoint {
        name: name.to_string(),
        node,
        workgroup_size: launch.workgroup_size,
        grid_size: launch.grid,
        dynamic_shared_bytes: launch.dynamic_shared_bytes,
        resource_bindings,
    }
}

/// The payload format the fixture target compilers declare, at `version`.
///
/// Format identity and profile identity are compared against each other when a
/// payload is attached, so a suite that spells one of them differently from the
/// next tests a rejection it did not mean to state.
pub fn payload_format(version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new(FIXTURE_TARGET, version).expect("fixture format must be valid")
}

/// The target profile the fixture payloads declare, at `version`.
///
/// One workgroup of 64 invocations with 1024 bytes of shared memory and no
/// dynamic shared allocation: large enough for every fixture entry point and
/// small enough that no device fact rejects it.
pub fn target_profile(version: u16) -> TargetProfile {
    TargetProfile::new(FIXTURE_TARGET, u64::from(version), [64, 1, 1], 64, 1_024, 0)
        .expect("fixture profile must be valid")
}

/// The target the payload fixtures name.
const FIXTURE_TARGET: &str = "test.target-binary";
