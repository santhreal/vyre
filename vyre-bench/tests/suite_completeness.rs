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
