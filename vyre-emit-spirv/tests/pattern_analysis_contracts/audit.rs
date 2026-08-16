//! `audit` pattern analysis contracts.

use vyre_lower::pattern_audit::PatternAudit;
use vyre_emit_spirv::patterns::*;
use vyre_lower::descriptor_builder::{body, descriptor, op};
use vyre_lower::KernelOpKind;

#[test]
fn empty_kernel_yields_no_findings() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let report = audit(&desc);
    assert_eq!(report.kernel_id, "empty");
    assert_eq!(report.finding_count(), 0);
    assert!(!report.has_any());
}

#[test]
fn oversized_workgroup_shows_in_audit() {
    let desc = descriptor("huge").dispatch(2048, 1, 1).build();
    let report = audit(&desc);
    assert!(report.has_any());
    assert!(!report.workgroup_validation.violations.is_empty());
}

#[test]
fn spirv_audit_merge_aggregates() {
    let mut acc = SpirvAuditReport::zero();
    let desc = descriptor("k")
        .dispatch(64, 1, 1)
        .body(body().op(op(KernelOpKind::SubgroupBallot, [0], 0)))
        .build();
    acc.merge(audit(&desc));
    // After merging in a kernel that uses SubgroupBallot, the
    // ballot capability bit should be set on the aggregate.
    assert!(acc.subgroup.capabilities.ballot);
}

#[test]
fn format_short_and_is_clean_on_empty() {
    let desc = descriptor("k").dispatch(64, 1, 1).build();
    let r = audit(&desc);
    assert!(r.is_clean());
    let s = r.format_short();
    assert!(s.contains("k (spirv)"));
    assert!(s.contains("0 findings"));
}

#[test]
fn subgroup_op_promotes_capability_in_audit() {
    let desc = descriptor("sg")
        .dispatch(64, 1, 1)
        .body(body().op(op(KernelOpKind::SubgroupBallot, [0], 0)))
        .build();
    let report = audit(&desc);
    assert!(report.has_any());
    assert!(report.subgroup.capabilities.ballot);
}
