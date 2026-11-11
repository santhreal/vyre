//! Three-domain whole-graph production compilation tests.
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
#[cfg(feature = "visual")]
use vyre_libs::graph_compositions::{
    build_interactive_graphics_pipeline, InteractiveGraphicsPipelineParams,
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
#[cfg(feature = "visual")]
#[test]
fn domain_4_interactive_graphics_pipeline_compiles_through_production_path() {
    let params = InteractiveGraphicsPipelineParams {
        width: 16,
        height: 16,
        box_count: 8,
        segment_count: 4,
        stroke_radius: 1,
        stroke_color: 0xFF00_00FF,
        glyph_count: 2,
        atlas_w: 8,
        atlas_h: 8,
        clip_rect: (2, 2, 14, 14),
        patch_w: 4,
        patch_h: 4,
        patch_dest: (4, 4),
    };
    let graph = build_interactive_graphics_pipeline(params)
        .expect("interactive graphics pipeline graph must build");

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("logical algorithm stage must validate");
    assert!(!logical.regions().is_empty());

    let request = CompileRequest::new(
        graph.clone(),
        facts_for(&graph, 4),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("interactive graphics graph must compile to artifact");
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

#[cfg(feature = "visual")]
#[test]
fn four_unrelated_domains_share_identical_production_compilation_route() {
    three_unrelated_domains_share_identical_production_compilation_route();
    let g4 = build_interactive_graphics_pipeline(InteractiveGraphicsPipelineParams {
        width: 8,
        height: 8,
        box_count: 4,
        segment_count: 2,
        stroke_radius: 1,
        stroke_color: 0xFF00_00FF,
        glyph_count: 1,
        atlas_w: 4,
        atlas_h: 4,
        clip_rect: (0, 0, 8, 8),
        patch_w: 2,
        patch_h: 2,
        patch_dest: (0, 0),
    })
    .expect("domain 4 build");

    let r4 = CompileRequest::new(
        g4.clone(),
        facts_for(&g4, 14),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("req 4");

    let a4 = compile(&r4).expect("compile domain 4");
    assert!(!a4.to_bytes().unwrap().is_empty());
}
