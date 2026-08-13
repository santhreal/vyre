//! Determinism gate test.
#![allow(missing_docs, clippy::field_reassign_with_default, unsafe_code)]

use vyre_bench::api::suite::SuiteKind;
use vyre_bench::runner::{execute_suite, RunConfig};

#[test]
fn test_determinism_gate() {
    let mut config = RunConfig::default();
    config.measured_samples = Some(30);
    config.determinism_runs = 3;
    config.case_ids = vec!["synthetic.flaky".to_string()];

    let registry = vyre_bench::registry::collect_all();

    let report = execute_suite(&registry, &SuiteKind::custom("flaky_test"), &config);
    assert_eq!(report.cases.len(), 1);
    let case = &report.cases[0];

    assert_eq!(
        case.status, "unstable",
        "Flaky case should be marked unstable"
    );
    let stats = case.metrics.get("wall_ns").expect("Missing wall_ns");
    assert!(
        stats.determinism_cv.is_some(),
        "determinism_cv should be populated"
    );
    assert!(
        stats.determinism_cv.unwrap() > 0.05,
        "CV should be high for flaky case"
    );
}

/// A repeatable synthetic workload gets a computed finite determinism CV.
#[test]
fn test_repeatable_workload_determinism() {
    unsafe {
        std::env::set_var("VYRE_ALLOW_FEW_SAMPLES", "1");
    }
    let mut config = RunConfig::default();
    config.measured_samples = Some(5);
    config.determinism_runs = 3;
    config.case_ids = vec!["synthetic.flaky".to_string()];

    let registry = vyre_bench::registry::collect_all();
    let report = execute_suite(&registry, &SuiteKind::custom("flaky_test"), &config);
    assert_eq!(report.cases.len(), 1);
    let stats = report.cases[0]
        .metrics
        .get("wall_ns")
        .expect("Missing wall_ns");
    let cv = stats
        .determinism_cv
        .expect("Fix: the runner must report a determinism CV whenever determinism_runs > 1");
    assert!(
        cv.is_finite() && cv >= 0.0,
        "Fix: determinism CV must be a finite non-negative number, got {cv}"
    );
}
