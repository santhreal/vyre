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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::lego_audit::test_ops::op;

    #[test]
    fn island_operations_are_detected_and_connected_ops_pass() {
        let mut island = op("vyre-libs::math::isolated_op", Tier::T3, &[]);
        island.own_nodes = 50;
        island.composed_nodes = 0;

        let mut caller = op(
            "vyre-libs::math::caller_op",
            Tier::T3,
            &["vyre-libs::math::callee_op"],
        );
        caller.own_nodes = 30;
        caller.composed_nodes = 20;

        let mut callee = op("vyre-libs::math::callee_op", Tier::T3, &[]);
        callee.own_nodes = 30;
        callee.composed_nodes = 0;

        let mut report = Report::clean();
        let flagged = check_8_composability(&mut report, &[island, caller, callee]);
        assert_eq!(
            flagged, 1,
            "only the isolated op with no callers and no children should be flagged"
        );
        assert!(report.findings[0]
            .message
            .contains("vyre-libs::math::isolated_op is an island"));
    }
}
