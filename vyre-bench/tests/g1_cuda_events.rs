//! CUDA events test.
#![allow(missing_docs)]
#![allow(clippy::field_reassign_with_default)]
use vyre_bench::api::suite::SuiteKind;
use vyre_bench::runner::{execute_suite, RunConfig};

#[test]
fn test_cuda_events_populated() {
    let mut config = RunConfig::default();
    config.measured_samples = Some(30);
    config.backend_id = Some("cuda".to_string());
    config.case_ids = vec!["foundation.elementwise.add.1m".to_string()];
    let registry = vyre_bench::registry::collect_all();

    let report = execute_suite(&registry, &SuiteKind::Smoke, &config);
    assert_eq!(
        report.cases.len(),
        1,
        "CUDA event test must execute exactly one case; empty reports indicate broken case selection or backend acquisition"
    );

    let case = &report.cases[0];
    assert_eq!(
        case.status, "pass",
        "CUDA event benchmark failed before timing assertions: {:?}",
        case.correctness
    );
    let metrics = &case.metrics;

    let dispatch = metrics
        .get("dispatch_ns")
        .expect("CUDA device timestamp metric missing");
    assert!(dispatch.p50 > 0, "CUDA device time should be > 0");

    for optional_host_phase in ["kernel_queue_submit_ns", "device_sync_ns"] {
        if let Some(metric) = metrics.get(optional_host_phase) {
            assert!(
                metric.p50 > 0,
                "reported CUDA host phase `{optional_host_phase}` must be positive"
            );
        }
    }
}
