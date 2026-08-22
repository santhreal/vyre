#[test]
fn release_matrix_does_not_attach_condition_eval_to_specialized_workloads() {
    let registry = vyre_bench::registry::collect_all();
    let matrix = vyre_bench::release_matrix::build_release_matrix(&registry);
    for family_id in [
        "metadata-conditions",
        "offset-count-aggregation",
        "entropy-window",
    ] {
        let family = matrix
            .families
            .iter()
            .find(|family| family.id == family_id)
            .unwrap_or_else(|| panic!("Fix: release matrix missing family `{family_id}`."));
        assert!(
            !family
                .matched_cases
                .iter()
                .any(|case| case == "conditions.yara_like.eval.1m"),
            "Fix: workload `{family_id}` must not inherit the generic condition-eval release case."
        );
        assert!(
            !family
                .cpu_sota_100x_cases
                .iter()
                .any(|case| case == "conditions.yara_like.eval.1m"),
            "Fix: workload `{family_id}` must not count generic condition-eval as its CPU-SOTA 100x proof case."
        );
    }
}

/// WHY: this workload used to also match `frontend.c.parser.linux_driver_pipeline`,
/// a whole-pipeline parse whose throughput says nothing about AST traversal. That
/// case left the registry with the C frontend crate, which made the two negative
/// assertions here unfailable: they named one dead id. Pinning the matched set
/// exactly fails on any broad case attaching to this family, not just that one.
#[test]
fn release_matrix_attaches_only_the_canonical_case_to_the_ast_motif_workload() {
    let registry = vyre_bench::registry::collect_all();
    let matrix = vyre_bench::release_matrix::build_release_matrix(&registry);
    let family = matrix
        .families
        .iter()
        .find(|family| family.id == "ast-motif-traversal")
        .expect("Fix: release matrix missing AST motif traversal family.");

    assert_eq!(
        family.matched_cases,
        ["release.ast_motif_traversal.1m"],
        "Fix: the AST motif traversal workload measures one canonical release case. Any other \
         case attached here reports some other program's throughput as AST traversal."
    );
    assert_eq!(
        family.cpu_sota_100x_cases,
        ["release.ast_motif_traversal.1m"],
        "Fix: the CPU-SOTA 100x proof for this workload must come from the canonical release \
         case, not from broader pipeline evidence."
    );
    assert_eq!(
        family.max_cpu_sota_min_speedup_x,
        Some(100.0),
        "Fix: AST motif traversal workload max CPU-SOTA speedup must come from the canonical release case."
    );
}
