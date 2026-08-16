//! `audit` pattern analysis contracts.

use vyre_lower::pattern_audit::PatternAudit;
use vyre_emit_naga::patterns::*;
use vyre_lower::descriptor_builder::descriptor;

#[test]
fn empty_kernel_yields_zero_candidates() {
    let desc = descriptor("empty").build();
    let report = audit(&desc);
    assert_eq!(report.kernel_id, "empty");
    assert_eq!(report.finding_count(), 0);
    assert!(!report.has_any());
}

#[test]
fn merge_aggregates_findings() {
    let mut acc = NagaAuditReport::zero();
    let desc = descriptor("k").dispatch(64, 1, 1).build();
    let r1 = audit(&desc);
    let r2 = audit(&desc);
    acc.merge(r1);
    acc.merge(r2);
    // No findings on empty kernels  -  sums to 0.
    assert_eq!(acc.finding_count(), 0);
}

#[test]
fn format_short_and_is_clean_on_empty() {
    let desc = descriptor("k").build();
    let r = audit(&desc);
    assert!(r.is_clean());
    let s = r.format_short();
    assert!(s.contains("k (naga)"));
    assert!(s.contains("0 candidates"));
}

#[test]
fn nonempty_kernel_audit_doesnt_panic() {
    let report = audit(&super::single_store_desc("k"));
    assert_eq!(report.kernel_id, "k");
    // 3-op, 1-binding kernel sits below every naga pattern threshold
    // (vec_pack needs Load/Store fusion groups, prewarm needs
    // ops >= 50 or bindings >= 4).
    // The contract this test enforces is "audit returns cleanly on
    // a real kernel without panicking", not a non-zero candidate count.
    assert_eq!(report.finding_count(), 0);
}
