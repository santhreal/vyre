//! Tests defending search budget bounds and dependency endpoint classification contracts.

use std::collections::BTreeMap;
use vyre_foundation::ir::ProgramGraph;
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, ArtifactNodeId, ArtifactValueId, CompileObjective, CompileRequest, DependencyEdge,
    DependencyEndpoint, DependencyKind, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;
use graph_fixtures::producer_consumer_pair;
use vyre_test_support::pass_programs::copy_program;

fn fixture_graph() -> ProgramGraph {
    producer_consumer_pair(
        copy_program("input", "intermediate"),
        copy_program("intermediate", "output"),
    )
}

fn fixture_facts() -> ExternalFacts {
    ExternalFacts::new(
        Digest([0x5A; 32]),
        BTreeMap::from([("items".to_string(), 16)]),
    )
}

fn fixture_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256).with_occupancy(0, 0)
}

#[test]
fn candidate_search_respects_max_candidates_budget() {
    for max_candidates in [1, 2, 8, 32] {
        let budget = SearchBudget::new(max_candidates, 1_000_000, 8, 0, 1_000_000_000);
        let request = CompileRequest::new(
            fixture_graph(),
            fixture_facts(),
            fixture_device(),
            budget,
            CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
        )
        .validate()
        .expect("compile request must validate");

        let artifact = compile(&request).expect("compilation must succeed within budget");
        let explored = artifact.selected_plan().candidates_explored;
        assert!(
            explored <= max_candidates,
            "explored candidates {explored} exceeded max_candidates budget {max_candidates}"
        );
    }
}

#[test]
fn dependency_endpoint_and_kind_classification() {
    let node_0 = DependencyEndpoint::Node(ArtifactNodeId(0));
    let node_1 = DependencyEndpoint::Node(ArtifactNodeId(1));
    let val_0 = DependencyEndpoint::Value(ArtifactValueId(0));

    let edge_data = DependencyEdge {
        from: node_0,
        to: node_1,
        kind: DependencyKind::Data,
        value: Some(ArtifactValueId(0)),
    };
    assert_eq!(edge_data.kind, DependencyKind::Data);
    assert_eq!(edge_data.from, node_0);
    assert_eq!(edge_data.to, node_1);
    assert_eq!(edge_data.value, Some(ArtifactValueId(0)));

    let edge_mat = DependencyEdge {
        from: node_0,
        to: val_0,
        kind: DependencyKind::Materialization,
        value: Some(ArtifactValueId(0)),
    };
    assert_eq!(edge_mat.kind, DependencyKind::Materialization);
    assert_eq!(edge_mat.from, node_0);
    assert_eq!(edge_mat.to, val_0);

    let edge_ret = DependencyEdge {
        from: node_0,
        to: node_1,
        kind: DependencyKind::Retained,
        value: Some(ArtifactValueId(1)),
    };
    assert_eq!(edge_ret.kind, DependencyKind::Retained);
}

/// WHY: Section 193.4 requires proving dependency endpoint classification across every
/// DependencyEndpoint and DependencyKind variant: Node-to-Node edges constrain executable
/// group stages, Node-to-Value materialization edges are sinks, and Value-to-Node retained
/// inputs with no producer are external sources.
#[test]
fn compiled_graph_contains_all_dependency_endpoint_and_kind_variants() {
    use vyre_foundation::ir::{
        BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ShapeDim,
        ValueContract, ValueLifetime,
    };

    let mut graph = ProgramGraph::new();
    let state_contract = ValueContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Retained,
    };
    let inv_contract = ValueContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadOnly,
        lifetime: ValueLifetime::Invocation,
    };

    let ext_input = graph
        .add_external_value("input_data", inv_contract.clone())
        .expect("external input must register");
    let ext_state = graph
        .add_external_value("initial_state", state_contract.clone())
        .expect("external retained state must register");

    let prog0 = Program::wrapped(
        vec![
            BufferDecl::storage("in_buf", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
            BufferDecl::storage("state_buf", 1, BufferAccess::ReadWrite, DataType::F32)
                .with_count(16),
            BufferDecl::storage("temp_out", 2, BufferAccess::ReadWrite, DataType::F32)
                .with_count(16),
            BufferDecl::storage("state_next", 3, BufferAccess::ReadWrite, DataType::F32)
                .with_count(16),
        ],
        [16, 1, 1],
        vec![
            Node::store("temp_out", Expr::u32(0), Expr::load("in_buf", Expr::u32(0))),
            Node::store(
                "state_next",
                Expr::u32(0),
                Expr::load("state_buf", Expr::u32(0)),
            ),
        ],
    );

    let (node0, out0) = graph
        .add_node(
            "stage0",
            prog0,
            vec![
                GraphInput {
                    buffer: "in_buf".into(),
                    value: ext_input,
                    contract: inv_contract,
                },
                GraphInput {
                    buffer: "state_buf".into(),
                    value: ext_state,
                    contract: state_contract.clone(),
                },
            ],
            vec![
                GraphOutput {
                    buffer: "temp_out".into(),
                    name: "data_temp".into(),
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Symbol("items".into())],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Invocation,
                    },
                    retained_successor_of: None,
                },
                GraphOutput {
                    buffer: "state_next".into(),
                    name: "state_stage0".into(),
                    contract: state_contract.clone(),
                    retained_successor_of: Some(ext_state),
                },
            ],
        )
        .expect("node0 must connect");
    let temp_val = out0[0];
    let state_val_0 = out0[1];

    let prog1 = Program::wrapped(
        vec![
            BufferDecl::storage("temp_in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
            BufferDecl::storage("state_in", 1, BufferAccess::ReadWrite, DataType::F32)
                .with_count(16),
            BufferDecl::output("final_out", 2, DataType::F32).with_count(16),
        ],
        [16, 1, 1],
        vec![Node::store(
            "final_out",
            Expr::u32(0),
            Expr::add(
                Expr::load("temp_in", Expr::u32(0)),
                Expr::load("state_in", Expr::u32(0)),
            ),
        )],
    );

    let (node1, out1) = graph
        .add_node(
            "stage1",
            prog1,
            vec![
                GraphInput {
                    buffer: "temp_in".into(),
                    value: temp_val,
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Symbol("items".into())],
                        access: BufferAccess::ReadOnly,
                        lifetime: ValueLifetime::Invocation,
                    },
                },
                GraphInput {
                    buffer: "state_in".into(),
                    value: state_val_0,
                    contract: state_contract.clone(),
                },
            ],
            vec![GraphOutput {
                buffer: "final_out".into(),
                name: "result".into(),
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Symbol("items".into())],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: Some(state_val_0),
            }],
        )
        .expect("node1 must connect");
    let final_val = out1[0];

    let req = CompileRequest::new(
        graph,
        fixture_facts(),
        fixture_device(),
        SearchBudget::new(32, 1_000_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&req).expect("compilation must succeed");
    let dependencies = artifact.dependencies();

    let has_value_to_node_retained = dependencies.iter().any(|edge| {
        edge.from == DependencyEndpoint::Value(ArtifactValueId(ext_state.0))
            && edge.to == DependencyEndpoint::Node(ArtifactNodeId(node0.0))
            && edge.kind == DependencyKind::Retained
    });
    assert!(
        has_value_to_node_retained,
        "must classify external retained input as Value-to-Node Retained dependency"
    );

    let has_node_to_node_retained = dependencies.iter().any(|edge| {
        edge.from == DependencyEndpoint::Node(ArtifactNodeId(node0.0))
            && edge.to == DependencyEndpoint::Node(ArtifactNodeId(node1.0))
            && edge.kind == DependencyKind::Retained
    });
    assert!(
        has_node_to_node_retained,
        "must classify inter-node retained transition as Node-to-Node Retained dependency"
    );

    let has_node_to_node_data = dependencies.iter().any(|edge| {
        edge.from == DependencyEndpoint::Node(ArtifactNodeId(node0.0))
            && edge.to == DependencyEndpoint::Node(ArtifactNodeId(node1.0))
            && edge.kind == DependencyKind::Data
    });
    assert!(
        has_node_to_node_data,
        "must classify intermediate dataflow as Node-to-Node Data dependency"
    );

    let has_node_to_value_materialization = dependencies.iter().any(|edge| {
        edge.from == DependencyEndpoint::Node(ArtifactNodeId(node1.0))
            && edge.to == DependencyEndpoint::Value(ArtifactValueId(final_val.0))
            && edge.kind == DependencyKind::Materialization
    });
    assert!(
        has_node_to_value_materialization,
        "must classify public output as Node-to-Value Materialization sink dependency"
    );

    assert!(
        !artifact.selected_plan().fusion.is_empty(),
        "stage derivation must complete without panicking on Value endpoints"
    );
    assert!(
        artifact
            .selected_plan()
            .fusion
            .iter()
            .all(|f| f.stage <= 100),
        "fusion records must record valid stage assignments"
    );
}
