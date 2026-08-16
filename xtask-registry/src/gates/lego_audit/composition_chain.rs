//! Check 6: a non-leaf operation exposes at least one registered child region.
//!
//! Composition that leaves no `source_region` behind is invisible: the audit
//! cannot tell an operation that reuses a primitive from one that copied its
//! body.

use super::*;

/// Enforce only the canonical primitive adoption and exception contract.
/// Check 6: composition-chain coverage  -  every non-leaf op should have
/// at least one child Region with a `source_region` pointing at
/// another registered op. Ops that explicitly declare leaf status in the
/// canonical operation contract are exempt.
pub(super) fn check_6_composition_chain_coverage(report: &mut Report, ops: &[OpInfo]) -> usize {
    let mut flagged = 0usize;
    report.note("[6/10] Composition-chain coverage (non-leaf ops must have ≥ 1 child Region with source_region)".to_string());
    for op in ops {
        // Tier 2 intrinsics and Tier 2.5 primitives are leaves unless
        // their own bodies choose to compose deeper primitives.
        if matches!(op.tier, Tier::T2 | Tier::T2_5) || !under_composition_rules(op) {
            continue;
        }
        if op.children.is_empty() {
            report.find(violation(format!("  ⚠ {} has no registered child Regions  -  either mark it a leaf primitive or wrap inlined sub-bodies via vyre_foundation::composition::wrap_child_region(<child_op_id>, ...).",
                op.id)));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note(
            "  ✓ every non-leaf op names at least one child op in its Region chain".to_string(),
        );
    }
    flagged
}
