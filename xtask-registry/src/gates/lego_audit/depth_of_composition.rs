//! Check 2: a Tier 3 operation composes registered children or declares itself a leaf.
//!
//! A Tier 3 operation that inlines everything it needs is a wall of IR that no
//! other operation can reuse. Either a quarter of its nodes come from registered
//! children, or it is an irreducible pure-IR leaf and says so.

use super::*;

/// Check 2: per-op composition depth  -  for Tier 3 ops, composed_nodes
/// should dominate own_nodes.
pub(super) fn check_2_depth_of_composition(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[2/10] Depth-of-composition (Tier 3 ops compose ≥25% registered child nodes or declare a pure-IR leaf)".to_string());
    let shallow = shallow_tier3_ops(ops);
    for op in &shallow {
        report.find(violation(format!("  ✗ {} Tier 3 op has own={} composed={} and {} child op(s)  -  registered child composition is below 25%. Wrap sub-bodies in vyre_foundation::composition::wrap_child_region(<primitive_id>, ...), or explicitly classify an irreducible pure-IR leaf.",
            op.id, op.own_nodes, op.composed_nodes, op.children.len())));
    }
    if shallow.is_empty() {
        report.note(
            "  ✓ Tier 3 ops meet registered-child depth or declare reviewed pure-IR leaves"
                .to_string(),
        );
    }
    shallow.len()
}

/// Every Tier 3 operation whose registered-child composition is under a quarter
/// of the nodes it owns.
///
/// The rule is separated from the reporting for the same reason checks 1 and 10
/// separate theirs: the question a reader asks of this check is which operations
/// it selects, and that is one expression rather than a loop carrying a counter.
pub(super) fn shallow_tier3_ops(ops: &[OpInfo]) -> Vec<&OpInfo> {
    ops.iter()
        .filter(|op| op.tier == Tier::T3 && under_composition_rules(op))
        .filter(|op| {
            let total = op.own_nodes + op.composed_nodes;
            op.children.is_empty() || op.composed_nodes.saturating_mul(4) < total
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::lego_audit::test_ops::op;

    /// This boundary test locks the minimum material composition ratio at exactly 25%.
    #[test]
    fn quarter_composed_tier3_operation_passes_depth_gate() {
        let mut composed = op(
            "vyre-libs::nn::reviewed_orchestrator",
            Tier::T3,
            &["vyre-libs::nn::child"],
        );
        composed.own_nodes = 75;
        composed.composed_nodes = 25;
        assert_eq!(
            check_2_depth_of_composition(&mut Report::clean(), &[composed]),
            0
        );
    }

    /// This negative twin prevents a nominal child edge from hiding an almost entirely inlined Tier-3 implementation.
    #[test]
    fn below_quarter_composed_tier3_operation_fails_depth_gate() {
        let mut inlined = op(
            "vyre-libs::nn::inlined_orchestrator",
            Tier::T3,
            &["vyre-libs::nn::child"],
        );
        inlined.own_nodes = 76;
        inlined.composed_nodes = 24;
        assert_eq!(
            check_2_depth_of_composition(&mut Report::clean(), &[inlined]),
            1
        );
    }

}
