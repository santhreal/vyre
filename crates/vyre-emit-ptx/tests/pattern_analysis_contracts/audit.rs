//! `audit` pattern analysis contracts.

use vyre_emit_ptx::patterns::*;
use vyre_emit_ptx::ComputeCapability;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, global_rw, lit, op};
use vyre_lower::pattern_audit::PatternAudit;
use vyre_lower::{KernelOpKind, LiteralValue};

#[test]
fn empty_kernel_yields_zero_candidates() {
    let desc = descriptor("empty").build();
    let report = audit(&desc, ComputeCapability::SM_70);
    assert_eq!(report.kernel_id, "empty");
    assert_eq!(report.finding_count(), 0);
    assert!(!report.has_any());
}

#[test]
fn vec_load_chain_shows_up_in_audit() {
    let desc = descriptor("vload_chain")
        .slot(global_rw(0, DataType::U32, "buf"))
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::LoadGlobal, [0, 0], 2),
                    op(
                        KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                        [0, 1],
                        3,
                    ),
                    op(KernelOpKind::LoadGlobal, [0, 3], 4),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let report = audit(&desc, ComputeCapability::SM_70);
    assert!(report.has_any());
    assert_eq!(report.vec_load.candidates.len(), 1);
    assert_eq!(report.finding_count(), 1);
}

#[test]
fn ptx_audit_merge_aggregates_candidates() {
    let mut acc = PtxAuditReport::zero();
    // Merge two empty reports  -  both have no findings, so aggregate
    // stays empty.
    let desc = descriptor("k").dispatch(64, 1, 1).build();
    acc.merge(audit(&desc, ComputeCapability::SM_70));
    acc.merge(audit(&desc, ComputeCapability::SM_70));
    assert_eq!(acc.finding_count(), 0);
}

#[test]
fn format_short_and_is_clean_on_empty() {
    let desc = descriptor("k").build();
    let r = audit(&desc, ComputeCapability::SM_80);
    assert!(r.is_clean());
    let s = r.format_short();
    assert!(s.contains("k (ptx sm_8_0)"));
    assert!(s.contains("0 candidates"));
}

#[test]
fn audit_carries_target_through() {
    let desc = descriptor("k").build();
    let r80 = audit(&desc, ComputeCapability::SM_80);
    let r90 = audit(&desc, ComputeCapability::SM_90);
    assert_eq!(r80.target, ComputeCapability::SM_80);
    assert_eq!(r90.target, ComputeCapability::SM_90);
}
