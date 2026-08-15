//! Thesis workload coverage contracts for the meta-harness.

use std::collections::BTreeSet;

/// WHY: the release thesis is that vyre proves whole-program compiler and
/// security workloads, not element-wise throughput. Each id below is one such
/// workload, so losing one silently would let the suite regress to primitives
/// while still reporting a pass.
///
/// `scan.literal_set.irregular_hotloop.4m` is deliberately absent. The
/// literal-set engine it measured left the workspace with the scan product, so
/// nothing here can execute it; `scan.ac.irregular_literals.4m` keeps irregular
/// literal scanning covered through the matching kernels that stayed.
///
/// The three `frontend.rust.*` lexer and loop workloads are absent for the same
/// reason: the crate they lexed moved to `software/frontend-rust`, so no case in
/// this workspace can execute them. Parsing stays a proven class through the C
/// front end that remains in `vyre-libs`, which is what the two `c_lexer` and
/// `c_ast_traversal` ids below measure.
#[test]
fn benchmark_registry_contains_program_level_thesis_workloads() {
    let registry = vyre_bench::registry::collect_all();
    let ids = registry
        .iter()
        .map(|case| case.id().0)
        .collect::<BTreeSet<_>>();

    for required_id in [
        "parser.c_lexer.small_state_transition.4k",
        "release.c_ast_traversal.1m",
        "dataflow.ifds.skewed.closure.1m",
        "dataflow.ifds.skewed.step.1m",
        "scan.ac.irregular_literals.4m",
        "foundation.dfa_match.256k",
        "primitives.graph.csr_skewed_frontier.1m",
        "primitives.graph.frontier_step.1m",
        "runtime.megakernel.truth.1024",
    ] {
        assert!(
            ids.contains(required_id),
            "benchmark registry is missing thesis workload {required_id}"
        );
    }
}

#[test]
fn release_suite_cannot_regress_to_elementwise_only_evidence() {
    let registry = vyre_bench::registry::collect_all();
    let release_cases = registry
        .iter()
        .filter(|case| case.active_in_suite(&vyre_bench::api::suite::SuiteKind::Release))
        .collect::<Vec<_>>();

    let mut evidence_classes = BTreeSet::new();
    for case in release_cases {
        let metadata = case.metadata();
        let id = metadata.id.0;
        let tags = metadata.tags;
        // Keyed on the workload's own tag, not on a crate-shaped id prefix: the
        // `frontend.rust.*` and `frontend.c.*` ids this used to name left with
        // their crates, and an id prefix that matches nothing detects no class
        // while reading as though it does.
        if tags.iter().any(|tag| tag == "parser" || tag == "ast") {
            evidence_classes.insert("parsing");
        }
        if tags.iter().any(|tag| tag == "graph" || tag == "frontier") {
            evidence_classes.insert("graph_traversal");
        }
        if tags.iter().any(|tag| tag == "dfa" || tag == "pattern") || id.contains("dfa_match") {
            evidence_classes.insert("pattern_matching");
        }
        if id.starts_with("runtime.megakernel.") {
            evidence_classes.insert("megakernel");
        }
        if id.starts_with("runtime.nvme_gpu_ingest.")
            || tags
                .iter()
                .any(|tag| tag == "io_uring" || tag == "gpu-ingest" || tag == "zero-copy")
        {
            evidence_classes.insert("zero_copy_ingest");
        }
        if id.contains("egraph") || id.contains("optimizer") {
            evidence_classes.insert("optimizer");
        }
    }

    for required_class in [
        "parsing",
        "graph_traversal",
        "pattern_matching",
        "megakernel",
        "zero_copy_ingest",
        "optimizer",
    ] {
        assert!(
            evidence_classes.contains(required_class),
            "release benchmark suite is missing {required_class} evidence"
        );
    }
}
