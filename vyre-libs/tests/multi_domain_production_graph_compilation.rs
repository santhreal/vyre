//! Three-domain whole-graph production compilation tests (BACKLOG row 48 and 51).
//!
//! Proves that representative complete graphs from at least three unrelated domains
//! (Dense Neural Pipeline, CSR Graph Traversal, and Streaming Parser Pipeline) compose
//! through domain-neutral whole-graph construction APIs and compile through the identical
//! production compilation pipeline (`CompileRequest` -> `vyre::compiler::compile` -> `Artifact`).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre_foundation::ir::{GraphValueId, ProgramGraph, ValueLifetime};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_libs::graph_compositions::{
    build_csr_graph_traversal_pipeline, build_dense_neural_pipeline,
    build_streaming_parser_pipeline,
};

fn facts_for(graph: &ProgramGraph, domain_byte: u8) -> ExternalFacts {
    let mut facts = ExternalFacts::new(Digest([domain_byte; 32]), BTreeMap::new());
    for (val_id, val) in graph.values().iter().enumerate() {
        if val.contract.lifetime == ValueLifetime::Constant {
            facts
                .constant_identities
                .insert(GraphValueId(val_id as u32), Digest([domain_byte; 32]));
        }
    }
    facts
}

#[test]
fn domain_1_dense_neural_pipeline_compiles_through_production_path() {
    let graph =
        build_dense_neural_pipeline(4, 16, 32, 8).expect("dense neural pipeline graph must build");

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("logical algorithm stage must validate");
    assert!(!logical.regions().is_empty());
    for region in logical.regions() {
        assert!(region.progress.guaranteed_termination);
        assert!(region.scratch.reusable);
    }

    let request = CompileRequest::new(
        graph.clone(),
        facts_for(&graph, 1),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("dense neural graph must compile to artifact");
    assert!(!artifact.nodes().is_empty());
    assert!(!artifact.abi().entries.is_empty());
    artifact.validate_abi().expect("abi must be valid");
    artifact
        .validate_geometry()
        .expect("geometry must be valid");
    let bytes = artifact
        .to_bytes()
        .expect("artifact must serialize to bytes");
    assert!(!bytes.is_empty());
}

#[test]
fn domain_2_csr_graph_traversal_compiles_through_production_path() {
    let graph = build_csr_graph_traversal_pipeline(16, 48, 2)
        .expect("csr graph traversal graph must build");

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("logical algorithm stage must validate");
    assert!(!logical.regions().is_empty());
    let has_stateful = logical.regions().iter().any(|r| r.kind.is_stateful());
    assert!(
        has_stateful,
        "CSR traversal must contain stateful/recurrent regions"
    );

    let request = CompileRequest::new(
        graph.clone(),
        facts_for(&graph, 2),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("csr graph must compile to artifact");
    assert!(!artifact.nodes().is_empty());
    assert!(!artifact.abi().entries.is_empty());
    artifact.validate_abi().expect("abi must be valid");
    artifact
        .validate_geometry()
        .expect("geometry must be valid");
    let bytes = artifact
        .to_bytes()
        .expect("artifact must serialize to bytes");
    assert!(!bytes.is_empty());
}

#[test]
fn domain_3_streaming_parser_pipeline_compiles_through_production_path() {
    let graph =
        build_streaming_parser_pipeline(32).expect("streaming parser pipeline graph must build");

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("logical algorithm stage must validate");
    assert!(!logical.regions().is_empty());

    let request = CompileRequest::new(
        graph.clone(),
        facts_for(&graph, 3),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("streaming parser graph must compile to artifact");
    assert!(!artifact.nodes().is_empty());
    assert!(!artifact.abi().entries.is_empty());
    artifact.validate_abi().expect("abi must be valid");
    artifact
        .validate_geometry()
        .expect("geometry must be valid");
    let bytes = artifact
        .to_bytes()
        .expect("artifact must serialize to bytes");
    assert!(!bytes.is_empty());
}

#[test]
fn three_unrelated_domains_share_identical_production_compilation_route() {
    let g1 = build_dense_neural_pipeline(2, 8, 16, 4).expect("domain 1 build");
    let g2 = build_csr_graph_traversal_pipeline(8, 16, 1).expect("domain 2 build");
    let g3 = build_streaming_parser_pipeline(16).expect("domain 3 build");
    let r1 = CompileRequest::new(
        g1.clone(),
        facts_for(&g1, 11),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("req 1");

    let r2 = CompileRequest::new(
        g2.clone(),
        facts_for(&g2, 12),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("req 2");

    let r3 = CompileRequest::new(
        g3.clone(),
        facts_for(&g3, 13),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("req 3");

    let a1 = compile(&r1).expect("compile domain 1");
    let a2 = compile(&r2).expect("compile domain 2");
    let a3 = compile(&r3).expect("compile domain 3");

    assert!(!a1.to_bytes().unwrap().is_empty());
    assert!(!a2.to_bytes().unwrap().is_empty());
    assert!(!a3.to_bytes().unwrap().is_empty());
}
