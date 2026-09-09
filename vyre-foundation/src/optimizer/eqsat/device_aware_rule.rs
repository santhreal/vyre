//! Typed hardware capability rules and contract validation.
//!
//! Closure-based rule gating (`Fn() -> bool`) is eliminated in favor of
//! [`HardwarePropertyRule`](super::HardwarePropertyRule), which carries typed
//! [`RuleFactIdentity`](super::RuleFactIdentity), [`ProofTerm`](super::ProofTerm),
//! and deterministic [`RuleCacheKey`](super::RuleCacheKey).

#[cfg(test)]
mod tests {
    use super::super::arith_fixture::{Arith, PairConstSelfRule, UnionEqualConstsRule};
    use super::super::{EGraph, HardwarePropertyRule, Rule, RuleFactIdentity, TargetFact};

    #[test]
    fn hardware_property_rule_satisfied_forwards_matches() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _a = egraph.add(Arith::Const(7));
        let inner: Box<dyn Rule<Arith>> = Box::new(PairConstSelfRule);
        let rule = HardwarePropertyRule::new(
            inner,
            vec![TargetFact::TensorCoreAvailable],
            vec![TargetFact::TensorCoreAvailable, TargetFact::SubgroupSize(32)],
        );
        assert!(
            !rule.matches(&egraph).is_empty(),
            "satisfied hardware rule must forward the inner rule's matches"
        );
    }

    #[test]
    fn hardware_property_rule_unsatisfied_returns_empty() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(7));
        let inner: Box<dyn Rule<Arith>> = Box::new(UnionEqualConstsRule);
        let rule = HardwarePropertyRule::new(
            inner,
            vec![TargetFact::TensorCoreAvailable],
            vec![TargetFact::SubgroupSize(32)],
        );
        let matches = rule.matches(&egraph);
        assert!(
            matches.is_empty(),
            "unsatisfied hardware rule must short-circuit to empty"
        );
    }

    #[test]
    fn hardware_property_rule_exposes_typed_identity_proof_and_cache_key() {
        let inner: Box<dyn Rule<Arith>> = Box::new(UnionEqualConstsRule);
        let rule = HardwarePropertyRule::new(
            inner,
            vec![TargetFact::SubgroupSize(32)],
            vec![TargetFact::SubgroupSize(32)],
        );
        assert_eq!(rule.name(), "union_equal_consts");
        assert_eq!(rule.witness(), UnionEqualConstsRule.witness());

        match rule.fact_identity() {
            RuleFactIdentity::HardwareProperty { required } => {
                assert_eq!(required, vec![TargetFact::SubgroupSize(32)]);
            }
            other => panic!("expected HardwareProperty identity, got {other:?}"),
        }

        let proof = rule.proof_term();
        assert_eq!(proof.rule_name, "union_equal_consts");
        assert!(!proof.justification.is_empty());

        let key = rule.cache_key();
        assert_ne!(key.0, [0u8; 32]);
    }
}
