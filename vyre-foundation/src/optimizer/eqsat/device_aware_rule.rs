//! Gating a rewrite rule on a caller-owned device-fact predicate.

use super::{DeviceAwareRule, EClassId, EGraph, ENodeLang, Rule};

impl<L: ENodeLang, F: Fn() -> bool> DeviceAwareRule<L, F> {
    /// Wrap `inner` so it only fires when `predicate()` returns true.
    pub fn new(inner: Box<dyn Rule<L>>, predicate: F) -> Self {
        Self { inner, predicate }
    }
}

impl<L: ENodeLang, F: Fn() -> bool> Rule<L> for DeviceAwareRule<L, F> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn matches(&self, egraph: &EGraph<L>) -> Vec<(EClassId, EClassId)> {
        if (self.predicate)() {
            self.inner.matches(egraph)
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::arith_fixture::{Arith, PairConstSelfRule, UnionEqualConstsRule};
    use super::super::{DeviceAwareRule, EGraph, Rule};

    #[test]
    fn device_aware_rule_predicate_true_forwards_matches() {
        // First half: with no Consts, even the always-on inner rule
        // produces no matches. The forwarder must propagate that.
        let egraph: EGraph<Arith> = EGraph::new();
        let inner: Box<dyn Rule<Arith>> = Box::new(PairConstSelfRule);
        let rule = DeviceAwareRule::new(inner, || true);
        assert!(
            rule.matches(&egraph).is_empty(),
            "empty egraph must yield empty matches even with predicate true"
        );

        // Second half: add a Const and confirm the predicate-true
        // forwarder surfaces the inner rule's hits.
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _a = egraph.add(Arith::Const(7));
        let inner: Box<dyn Rule<Arith>> = Box::new(PairConstSelfRule);
        let rule = DeviceAwareRule::new(inner, || true);
        assert!(
            !rule.matches(&egraph).is_empty(),
            "predicate true must forward the inner rule's matches"
        );
    }

    #[test]
    fn device_aware_rule_predicate_false_returns_empty() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(7));
        let _ = egraph.add(Arith::Const(7)); // hashcons collapses, but rule loop still scans
        let inner: Box<dyn Rule<Arith>> = Box::new(UnionEqualConstsRule);
        let rule = DeviceAwareRule::new(inner, || false);
        let matches = rule.matches(&egraph);
        assert!(
            matches.is_empty(),
            "predicate false must short-circuit to empty"
        );
    }

    #[test]
    fn device_aware_rule_forwards_inner_name() {
        let inner: Box<dyn Rule<Arith>> = Box::new(UnionEqualConstsRule);
        let rule = DeviceAwareRule::new(inner, || true);
        assert_eq!(rule.name(), "union_equal_consts");
    }
}
