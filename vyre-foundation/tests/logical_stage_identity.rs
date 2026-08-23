//! Logical-stage identity contracts.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, GraphInput, GraphOutput,
    Node, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::logical::{
    LogicalDependenceKind, LogicalExtent, LogicalProgramGraph, LogicalRegionKind,
    LOGICAL_ALGORITHM_VERSION,
};

fn map_program(workgroup: u32, addend: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::output("output", 1, DataType::U32).with_count(4),
        ],
        [workgroup, 1, 1],
        vec![Node::store(
            "output",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(addend)),
        )],
    )
}

fn semantic_wire(workgroup: u32, addend: u32) -> Vec<u8> {
    let graph = ProgramGraph::from_program("map", map_program(workgroup, addend))
        .expect("fixture graph must validate");
    LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate")
        .semantic_wire()
        .to_vec()
}

fn geometry_observing_wire(workgroup: u32, index: Expr, with_shared_storage: bool) -> Vec<u8> {
    let mut buffers = vec![BufferDecl::output("output", 0, DataType::U32).with_count(256)];
    if with_shared_storage {
        buffers.push(BufferDecl::workgroup("scratch", 256, DataType::U32));
    }
    let program = Program::wrapped(
        buffers,
        [workgroup, 1, 1],
        vec![Node::store("output", index, Expr::u32(1))],
    );
    let graph =
        ProgramGraph::from_program("geometry", program).expect("fixture graph must validate");
    LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate")
        .semantic_wire()
        .to_vec()
}

/// WHY: physical workgroup geometry is selected after logical semantics. If it
/// enters semantic identity, equivalent algorithms fragment semantic caches and
/// cannot share schedule search results.
#[test]
fn semantic_identity_excludes_workgroup_geometry() {
    assert_eq!(semantic_wire(32, 7), semantic_wire(256, 7));
}

/// WHY: normalizing workgroup geometry must not erase executable semantics.
/// Mutating the body while keeping graph topology and geometry fixed must change
/// the logical identity.
#[test]
fn semantic_identity_preserves_program_behavior() {
    assert_ne!(semantic_wire(32, 7), semantic_wire(32, 8));
}

/// WHY: a local invocation id changes meaning when workgroup width changes.
/// Its declared width is a semantic constraint, not a schedule-only choice.
#[test]
fn semantic_identity_preserves_observed_local_geometry() {
    assert_ne!(
        geometry_observing_wire(32, Expr::LocalId { axis: 0 }, false),
        geometry_observing_wire(256, Expr::LocalId { axis: 0 }, false)
    );
}

/// WHY: workgroup storage is sized against the declared cooperative width even
/// when the executable body does not read a local-id builtin.
#[test]
fn semantic_identity_preserves_workgroup_storage_geometry() {
    assert_ne!(
        geometry_observing_wire(32, Expr::u32(0), true),
        geometry_observing_wire(256, Expr::u32(0), true)
    );
}

/// WHY: the logical stage is a validated boundary, not an identity-only wrapper.
/// Bindings not declared by graph contracts are stale request state and fail
/// before schedule search.
#[test]
fn logical_stage_rejects_unowned_symbol_bindings() {
    let graph =
        ProgramGraph::from_program("map", map_program(32, 7)).expect("fixture graph must validate");
    let error = LogicalProgramGraph::validate(&graph, &BTreeMap::from([("stale".to_string(), 4)]))
        .expect_err("an unused symbolic binding must fail");
    assert!(error
        .to_string()
        .contains("unexpected symbolic extent `stale`"));
}

fn symbolic_two_stage_graph() -> ProgramGraph {
    let contract = ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    };
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", contract.clone())
        .expect("external value must validate");
    let stage = |input_name: &str, output_name: &str| {
        Program::wrapped(
            vec![
                BufferDecl::storage(input_name, 0, BufferAccess::ReadWrite, DataType::U32),
                BufferDecl::storage(output_name, 0, BufferAccess::ReadWrite, DataType::U32),
            ],
            [1, 1, 1],
            Vec::new(),
        )
    };
    let (_, first_outputs) = graph
        .add_node(
            "first",
            stage("input", "middle"),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "middle".into(),
                name: "middle".into(),
                contract: contract.clone(),
                retained_successor_of: None,
            }],
        )
        .expect("first stage must validate");
    graph
        .add_node(
            "second",
            stage("middle", "output"),
            vec![GraphInput {
                buffer: "middle".into(),
                value: first_outputs[0],
                contract: contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract,
                retained_successor_of: None,
            }],
        )
        .expect("second stage must validate");
    graph
}

/// WHY: schedule search can remap a library composition only when every region
/// contains a bounded domain, layout, alias set, effects, and explicit producer
/// dependencies derived from the typed graph contract.
#[test]
fn logical_regions_close_the_domain_contract() {
    let graph = symbolic_two_stage_graph();
    let logical =
        LogicalProgramGraph::validate(&graph, &BTreeMap::from([("items".to_string(), 8)]))
            .expect("symbolic graph must produce a closed logical domain");
    assert_eq!(LOGICAL_ALGORITHM_VERSION, 2);
    assert_eq!(logical.regions().len(), 2);

    let first = &logical.regions()[0];
    assert_eq!(first.kind, LogicalRegionKind::Parallel);
    assert_eq!(
        first.extents,
        [LogicalExtent::GraphValue {
            value: 1,
            axis: 0,
            symbol: "items".into(),
            bound: 8,
        }]
    );
    assert_eq!(first.index_map.axes, ["axis0"]);
    assert_eq!(first.index_map.row_major_strides, [1]);
    assert_eq!(first.layout.storage_order, [0]);
    assert_eq!(first.layout.strides, [1]);
    assert!(first.layout.contiguous);
    assert!(first.reduction_axes.is_empty());
    assert!(first.aliases.inputs_disjoint);
    assert!(first.aliases.outputs_disjoint);
    assert_eq!(first.aliases.in_place_values, [0]);
    assert_eq!(first.effects.reads, [0]);
    assert_eq!(first.effects.writes, [0, 1]);
    assert!(!first.effects.atomics);
    assert!(!first.effects.synchronizes);
    assert_eq!(first.max_points, 8);

    let second = &logical.regions()[1];
    assert_eq!(second.dependencies.len(), 1);
    assert_eq!(second.dependencies[0].predecessor.0, 0);
    assert_eq!(second.dependencies[0].values, [1]);
    assert_eq!(second.dependencies[0].kind, LogicalDependenceKind::Flow);
}

/// WHY: retained succession is the only intentional cross-submission alias.
/// The logical stage must authenticate both the prior/new value pair and the
/// in-place input effect before schedule search.
#[test]
fn logical_regions_record_retained_aliases() {
    let contract = ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(4)],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Retained,
    };
    let mut graph = ProgramGraph::new();
    let prior = graph
        .add_external_value("state.0", contract.clone())
        .expect("retained input must validate");
    graph
        .add_node(
            "update",
            Program::wrapped(
                vec![
                    BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(4),
                ],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![GraphInput {
                buffer: "state".into(),
                value: prior,
                contract: contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "state".into(),
                name: "state.1".into(),
                contract,
                retained_successor_of: Some(prior),
            }],
        )
        .expect("retained update must validate");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("retained logical domain must validate");
    let region = &logical.regions()[0];
    assert_eq!(region.kind, LogicalRegionKind::RetainedState);
    assert_eq!(region.aliases.retained_successors, [(1, 0)]);
    assert_eq!(region.aliases.in_place_values, [0]);
    assert_eq!(region.effects.writes, [0, 1]);
    assert!(region.effects.retained_state);
}

/// WHY: a dynamic graph-value extent without a compile-request binding is not
/// a schedulable domain. It must fail at the logical boundary, before search.
#[test]
fn logical_stage_rejects_unresolved_graph_value_extents() {
    let graph = symbolic_two_stage_graph();
    let error = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect_err("unresolved graph-value extent must fail");
    assert_eq!(
        error.to_string(),
        "logical graph is missing symbolic extent `items`"
    );
    let error = LogicalProgramGraph::validate(&graph, &BTreeMap::from([("items".to_string(), 0)]))
        .expect_err("zero graph-value extent must fail");
    assert!(error
        .to_string()
        .contains("unresolved extent at graph value"));
}

/// WHY: `ProgramGraph::from_program` preserves an unresolved runtime buffer as
/// a zero extent. Search must not interpret that marker as an empty workload.
#[test]
fn logical_stage_rejects_unresolved_runtime_buffer_extents() {
    let graph = ProgramGraph::from_program(
        "runtime",
        Program::wrapped(
            vec![BufferDecl::read("input", 0, DataType::U32)],
            [1, 1, 1],
            Vec::new(),
        ),
    )
    .expect("runtime-sized graph must remain representable before specialization");
    let error = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect_err("runtime-sized graph must specialize before schedule search");
    assert_eq!(
        error.to_string(),
        "logical region for graph node GraphNodeId(0) has unresolved extent at graph value GraphValueId(0) axis 0"
    );
}
fn logical_kind(program: Program) -> LogicalRegionKind {
    let graph = ProgramGraph::from_program("kind", program).expect("kind graph must validate");
    LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("kind domain must validate")
        .regions()[0]
        .kind
}

/// WHY: region structure is a semantic input to schedule search. Collapsing
/// ordered, reduction, or retained work into generic parallel work permits
/// illegal remapping.
#[test]
fn logical_stage_classifies_every_structured_region_kind() {
    assert_eq!(
        logical_kind(map_program(32, 1)),
        LogicalRegionKind::Parallel
    );
    assert_eq!(
        logical_kind(Program::wrapped(
            vec![BufferDecl::output("output", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::loop_for(
                "index",
                Expr::u32(0),
                Expr::u32(1),
                Vec::new(),
            )],
        )),
        LogicalRegionKind::Sequential
    );

    let mut reduction_graph = ProgramGraph::new();
    reduction_graph
        .add_node(
            "reduction",
            Program::wrapped(
                vec![BufferDecl::storage(
                    "output",
                    1,
                    BufferAccess::ReadWrite,
                    DataType::U32,
                )],
                [1, 1, 1],
                vec![Node::AllReduce {
                    buffer: "output".into(),
                    op: CollectiveOp::Sum,
                    group: CommGroup::WORLD,
                }],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(1)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .expect("reduction graph must validate");
    let reduction = LogicalProgramGraph::validate(&reduction_graph, &BTreeMap::new())
        .expect("reduction domain must validate");
    assert_eq!(reduction.regions()[0].kind, LogicalRegionKind::Reduction);
    assert_eq!(reduction.regions()[0].reduction_axes, [0]);
    assert!(reduction.regions()[0].effects.synchronizes);
    assert_eq!(
        logical_kind(Program::wrapped(
            vec![
                BufferDecl::storage("state", 1, BufferAccess::ReadWrite, DataType::U32,)
                    .with_count(1),
            ],
            [1, 1, 1],
            Vec::new(),
        )),
        LogicalRegionKind::RetainedState
    );
}
