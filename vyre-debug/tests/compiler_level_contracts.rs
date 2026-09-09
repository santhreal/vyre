//! Tests for 5-level compiler inspection, candidate reports, allocation reports, and structural diffs.

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_debug::{
    diff_allocations, diff_artifacts, diff_compiler_levels, diff_program_graphs, diff_programs,
    diff_search_certificates, diff_selected_plans, AllocationReport, CandidateReport,
    CompilerLevelView,
};

fn sample_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(4),
            BufferDecl::read_write("y", 1, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![Node::store(
            "y",
            Expr::gid_x(),
            Expr::add(Expr::load("x", Expr::gid_x()), Expr::u32(1)),
        )],
    )
}

fn sample_graph() -> ProgramGraph {
    let p = sample_program();
    ProgramGraph::from_program("main", p).expect("program graph")
}

#[test]
fn compiler_level_views_cover_all_levels() {
    let prog = sample_program();
    let graph = sample_graph();

    // Level 1: Semantic IR
    let lvl1_prog = CompilerLevelView::from_program("main", &prog);
    assert_eq!(lvl1_prog.level_number(), 1);
    assert_eq!(lvl1_prog.label(), "level_1_semantic_ir");
    assert!(lvl1_prog.render().contains("Level 1 (Semantic IR)"));

    let lvl1_graph = CompilerLevelView::from_program_graph(&graph);
    assert_eq!(lvl1_graph.level_number(), 1);

    // Level 2: Optimizer view
    let lvl2 = CompilerLevelView::Optimizer {
        pass_count: 3,
        proof_citations: vec!["algebraic_simplification".into()],
        inferred_facts: std::collections::BTreeMap::new(),
    };
    assert_eq!(lvl2.level_number(), 2);
    assert_eq!(lvl2.label(), "level_2_optimizer");

    // Level 4: Megakernel Artifact view
    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([1; 32]), std::collections::BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(4, 1_000, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("validated request");

    let artifact = compile(&req).expect("compiled artifact");
    let lvl4 = CompilerLevelView::from_artifact(&artifact);
    assert_eq!(lvl4.level_number(), 4);
    assert_eq!(lvl4.label(), "level_4_megakernel_plan");
    assert!(lvl4.render().contains("Level 4 (Megakernel Plan)"));

    // Structural diff between levels
    let diff_mismatch = diff_compiler_levels(&lvl1_prog, &lvl2);
    assert!(!diff_mismatch.is_identical);
    assert!(diff_mismatch.deltas[0].contains("level mismatch"));

    let diff_same = diff_compiler_levels(&lvl1_prog, &lvl1_prog);
    assert!(diff_same.is_identical);
    assert!(diff_same.deltas.is_empty());
}

#[test]
fn candidate_report_inspects_search_certificate_and_pruned_families() {
    let graph = sample_graph();
    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([2; 32]), std::collections::BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(8, 1_000, 2, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("validated request");

    let artifact = compile(&req).expect("compiled artifact");
    let cert = &artifact.selected_plan().certificate;
    let report = CandidateReport::from_certificate(cert);

    assert!(report.total_derived >= 1);
    let diff = diff_search_certificates(cert, cert);
    assert!(diff.is_identical);
    assert_eq!(diff.candidate_count_delta, 0);
}

#[test]
fn allocation_report_and_diff() {
    let graph = sample_graph();
    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([3; 32]), std::collections::BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(4, 1_000, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("validated request");

    let artifact = compile(&req).expect("compiled artifact");
    let alloc = AllocationReport::from_artifact(&artifact);

    assert!(alloc.total_global_bytes > 0);
    assert_eq!(alloc.resources.len(), 2);

    let diff = diff_allocations(&alloc, &alloc);
    assert!(diff.is_identical);
    assert_eq!(diff.global_bytes_delta, 0);
}

#[test]
fn program_and_graph_structural_diffs() {
    let p1 = sample_program();
    let p2 = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(4),
            BufferDecl::read_write("y", 1, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![
            Node::store(
                "y",
                Expr::gid_x(),
                Expr::add(Expr::load("x", Expr::gid_x()), Expr::u32(1)),
            ),
            Node::store("y", Expr::gid_x(), Expr::u32(42)),
        ],
    );

    let diff_p = diff_programs(&p1, &p2);
    assert!(!diff_p.is_identical);
    assert_eq!(diff_p.op_count_delta, 1);

    let g1 = sample_graph();
    let mut g2 = ProgramGraph::new();
    g2.add_node("node1", sample_program(), Vec::new(), Vec::new()).unwrap();
    g2.add_node("node2", sample_program(), Vec::new(), Vec::new()).unwrap();

    let diff_g = diff_program_graphs(&g1, &g2);
    assert!(!diff_g.is_identical);
    assert_eq!(diff_g.node_count_delta, 1);
}

#[test]
fn artifact_and_plan_structural_diffs() {
    let graph = sample_graph();
    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([4; 32]), std::collections::BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(4, 1_000, 1, 0, 1_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("validated request");

    let a1 = compile(&req).expect("artifact 1");
    let a2 = compile(&req).expect("artifact 2");

    let diff_a = diff_artifacts(&a1, &a2);
    assert!(diff_a.is_identical);
    assert!(diff_a.digest_matches);

    let diff_plan = diff_selected_plans(a1.selected_plan(), a2.selected_plan());
    assert!(diff_plan.is_identical);
}
