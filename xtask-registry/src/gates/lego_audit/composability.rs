//! Check 8: no operation is an island.
//!
//! A non-leaf operation is either composed by a caller or composes a child.
//! One that does neither is reachable from nothing and reaches nothing, which is
//! how a large operation ends up with no path to the surface.

use super::*;

pub(super) fn check_8_composability(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[8/10] Composability (every non-leaf op must be composed by ≥ 1 caller OR compose ≥ 1 child op)".to_string());
    let mut callers: HashMap<String, usize> = HashMap::new();
    for op in ops {
        for child in &op.children {
            *callers.entry(child.clone()).or_insert(0) += 1;
        }
    }
    let mut flagged = 0usize;
    for op in ops {
        if matches!(op.tier, Tier::T2 | Tier::T2_5) || !under_composition_rules(op) {
            continue;
        }
        let upstream = callers.get(&op.id).copied().unwrap_or(0);
        let downstream = op.children.len();
        if upstream == 0 && downstream == 0 {
            report.find(violation(format!("  ⚠ {} is an island: {} upstream caller(s), {} child op(s), {} total nodes. Fix: either wire it as a child of a caller, or wrap its body via vyre_foundation::composition::wrap_child_region(<existing_primitive>, ...).",
                op.id,
                upstream,
                downstream,
                op.own_nodes + op.composed_nodes)));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note("  ✓ no island ops".to_string());
    }
    flagged
}
