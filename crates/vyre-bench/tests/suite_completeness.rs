//! Suite completeness test.
use vyre_bench::api::suite::SuiteKind;

#[test]
fn test_suite_completeness() {
    let registry = vyre_bench::registry::collect_all();

    // SuiteKind::Smoke
    let smoke_cases: Vec<_> = registry
        .iter()
        .filter(|c| c.active_in_suite(&SuiteKind::Smoke))
        .collect();
    // Verify adversarial case is NOT in Smoke
    assert!(!smoke_cases
        .iter()
        .any(|c| c.id().0.starts_with("adversarial.")));
    // Verify foundation is in Smoke
    assert!(smoke_cases
        .iter()
        .any(|c| c.id().0.starts_with("foundation.")));

    // SuiteKind::Adversarial
    let adv_cases: Vec<_> = registry
        .iter()
        .filter(|c| c.active_in_suite(&SuiteKind::Adversarial))
        .collect();
    assert!(adv_cases
        .iter()
        .any(|c| c.id().0.starts_with("adversarial.")));

    // SuiteKind::Release
    let release_cases: Vec<_> = registry
        .iter()
        .filter(|c| c.active_in_suite(&SuiteKind::Release))
        .collect();
    assert!(release_cases
        .iter()
        .any(|c| c.id().0.starts_with("adversarial.")));
    assert!(release_cases
        .iter()
        .any(|c| c.id().0.starts_with("foundation.")));

    // Verify JSON Report Schema top-level wall_ns (B-5)
    let mut case = vyre_bench::report::fixture::case("test", &[]);
    case.status = "passed".to_string();
    case.wall_ns = Some(100.0);
    let dummy_report = vyre_bench::report::fixture::schema("smoke", vec![case]);
    let json_str = vyre_bench::report::json::generate_json_report(&dummy_report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(
        parsed["blockers"].as_array().is_some_and(Vec::is_empty),
        "Top-level JSON must carry an explicit blockers array"
    );
    let cases = parsed["cases"].as_array().unwrap();
    let case_obj = cases[0].as_object().unwrap();
    assert!(
        case_obj.contains_key("wall_ns"),
        "Top-level JSON must contain wall_ns"
    );
    assert!(case_obj.contains_key("id"));
    assert!(case_obj.contains_key("status"));
}

/// WHY: `--suite megakernel` is strictly opt-in (unbound empty-suite cases must
/// never leak into it) and must cover all six megakernel workload classes:
/// graph, parsing, pattern scanning, dataflow, math/simulation, and inference.
#[test]
fn megakernel_suite_is_opt_in_and_covers_all_six_workload_classes() {
    use std::collections::BTreeSet;

    let registry = vyre_bench::registry::collect_all();
    let megakernel_ids: BTreeSet<String> = registry
        .iter()
        .filter(|case| case.active_in_suite(&SuiteKind::Megakernel))
        .map(|case| case.id().0)
        .collect();

    let expected_by_class: &[(&str, &[&str])] = &[
        (
            "graph",
            &[
                "primitives.graph.csr_skewed_frontier.1m",
                "callgraph.reachability.step.262k",
                "release.megakernel_queue.1m",
            ],
        ),
        ("parsing", &["release.ast_motif_traversal.1m"]),
        (
            "pattern_scanning",
            &["scan.ac.irregular_count.4m", "foundation.dfa_match.256k"],
        ),
        (
            "dataflow",
            &[
                "release.alias_reaching_def.1m",
                "release.ifds_witness.1m",
                "dataflow.ifds.skewed.step.1m",
            ],
        ),
        (
            "math_simulation",
            &[
                "foundation.matmul.256",
                "foundation.stencil3.u32.1m",
                "foundation.reduce.sum.crossover",
            ],
        ),
        (
            "inference",
            &[
                "nn.linear_4bit_affine_grouped.1m",
                "foundation.attention.64",
            ],
        ),
    ];

    for (class_name, required_ids) in expected_by_class {
        for required in *required_ids {
            assert!(
                megakernel_ids.contains(*required),
                "megakernel class `{class_name}` missing required case `{required}` (enrolled: {megakernel_ids:?})"
            );
        }
    }

    assert!(
        !megakernel_ids.contains("foundation.gather.u32.1m"),
        "un-enrolled micro cases with empty suite slices must not leak into SuiteKind::Megakernel"
    );
    assert!(
        !megakernel_ids
            .iter()
            .any(|id| id.starts_with("adversarial.")),
        "adversarial cases must not enroll in SuiteKind::Megakernel"
    );
}
