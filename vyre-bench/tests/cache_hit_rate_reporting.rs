//! Cache hit rate is reported only when a run observes a cache contract.
#![allow(missing_docs)]
#![allow(clippy::field_reassign_with_default)]
use vyre_bench::api::suite::SuiteKind;
use vyre_bench::runner::{execute_suite, RunConfig};

#[test]
fn report_cache_rate_is_absent_without_an_observed_cache_contract() {
    let mut config = RunConfig::default();
    config.measured_samples = Some(30);
    config.backend_id = Some("cuda".to_string());
    config.case_ids = vec!["foundation.elementwise.add.1m".to_string()];
    let registry = vyre_bench::registry::collect_all();

    let report = execute_suite(&registry, &SuiteKind::Smoke, &config);
    assert_eq!(report.cases.len(), 1);
    assert_eq!(
        report.summary.cache_hit_rate, None,
        "the report must not manufacture a cache-hit ratio without counted lookups and hits"
    );
}
