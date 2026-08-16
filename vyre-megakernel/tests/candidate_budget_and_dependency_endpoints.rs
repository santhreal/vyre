//! Tests defending search budget bounds and dependency endpoint classification contracts.

use std::collections::BTreeMap;
use vyre_foundation::ir::ProgramGraph;
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, ArtifactNodeId, ArtifactValueId, CompileRequest, DependencyEdge, DependencyEndpoint,
    DependencyKind, DeviceFacts, Digest, ExternalFacts, SearchBudget,
};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;
use graph_fixtures::{copy_program, producer_consumer_pair};

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
            1_000_000,
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
