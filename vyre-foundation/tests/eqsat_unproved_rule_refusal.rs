//! Equality saturation refuses a rule that records no proof.
//!
//! Saturation is how the compiler derives alternative programs on its own, so a
//! rewrite whose witness records opacity instead of proof must not be able to
//! reach it. The class this closes is "an unproved rewrite entered candidate
//! search": the check sits at `try_saturate_named`, the single choke point every
//! raw-rule, named-family, and per-family entry point funnels through, so a new
//! caller inherits the refusal instead of needing its own copy.
//!
//! Not covered here: whether an individual `Structural` argument is true. That
//! is a review obligation on the argument text, not a property a test can read.

use vyre_foundation::optimizer::eqsat::{
    saturate, saturate_per_family, saturate_with_report, try_saturate, try_saturate_named,
    try_saturate_per_family, try_saturate_with_report, EChildren, EClassId, EGraph, EGraphError,
    ENodeLang, Family, Rule,
};
use vyre_foundation::optimizer::rewrite_contract::RewriteWitness;

/// Two-variant toy language: a literal and a binary sum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Toy {
    Const(u32),
    Add(EClassId, EClassId),
}

impl ENodeLang for Toy {
    fn children(&self) -> EChildren {
        match self {
            Self::Const(_) => EChildren::new(),
            Self::Add(a, b) => EChildren::from_slice(&[*a, *b]),
        }
    }

    fn with_children(&self, children: &[EClassId]) -> Self {
        match self {
            Self::Const(value) => Self::Const(*value),
            Self::Add(..) => Self::Add(children[0], children[1]),
        }
    }
}

/// Reflexive rule with a stated structural argument.
struct ProvedRule;

impl Rule<Toy> for ProvedRule {
    fn name(&self) -> &'static str {
        "toy_reflexive"
    }

    fn witness(&self) -> RewriteWitness {
        RewriteWitness::Structural("a class is equal to itself")
    }

    fn matches(&self, egraph: &EGraph<Toy>) -> Vec<(EClassId, EClassId)> {
        egraph
            .iter_nodes()
            .filter(|(_, node)| matches!(node, Toy::Const(_)))
            .map(|(id, _)| (id, id))
            .collect()
    }
}

/// Same matches, no recorded proof.
struct UnprovedRule;

impl Rule<Toy> for UnprovedRule {
    fn name(&self) -> &'static str {
        "toy_unproved"
    }

    fn witness(&self) -> RewriteWitness {
        RewriteWitness::Opaque("no equality argument is recorded for this fixture")
    }

    fn matches(&self, egraph: &EGraph<Toy>) -> Vec<(EClassId, EClassId)> {
        egraph
            .iter_nodes()
            .filter(|(_, node)| matches!(node, Toy::Const(_)))
            .map(|(id, _)| (id, id))
            .collect()
    }
}

struct UnprovedFamily;

impl Family<Toy> for UnprovedFamily {
    fn name(&self) -> &'static str {
        "toy_unproved_family"
    }

    fn rules(&self) -> Vec<Box<dyn Rule<Toy>>> {
        vec![Box::new(UnprovedRule)]
    }
}

fn seeded_graph() -> EGraph<Toy> {
    let mut egraph = EGraph::new();
    let left = egraph.add(Toy::Const(2));
    let right = egraph.add(Toy::Const(3));
    egraph.add(Toy::Add(left, right));
    egraph
}

fn expect_refusal(result: Result<impl core::fmt::Debug, EGraphError>) {
    match result {
        Err(EGraphError::UnprovedRule { rule }) => {
            assert_eq!(
                rule, "toy_unproved",
                "the refusal must name the offending rule"
            );
        }
        other => panic!("saturation must refuse an opaque-witness rule, got {other:?}"),
    }
}

#[test]
fn a_proved_rule_saturates() {
    let mut egraph = seeded_graph();
    let rules: Vec<Box<dyn Rule<Toy>>> = vec![Box::new(ProvedRule)];
    let report = try_saturate_named(&mut egraph, "toy", &rules, 4)
        .expect("a rule with a recorded proof must be admitted");
    assert_eq!(report.rewrite_family, "toy");
    assert_eq!(report.rule_count, 1);
}

#[test]
fn an_unproved_rule_is_refused_by_every_saturation_entry_point() {
    let rules: Vec<Box<dyn Rule<Toy>>> = vec![Box::new(UnprovedRule)];

    let mut egraph = seeded_graph();
    expect_refusal(try_saturate_named(&mut egraph, "toy", &rules, 4));

    let mut egraph = seeded_graph();
    expect_refusal(try_saturate(&mut egraph, &rules, 4));

    let mut egraph = seeded_graph();
    expect_refusal(try_saturate_with_report(&mut egraph, &rules, 4));

    let mut egraph = seeded_graph();
    let families: [&dyn Family<Toy>; 1] = [&UnprovedFamily];
    expect_refusal(try_saturate_per_family(&mut egraph, &families, |_| 4));
}

#[test]
fn a_mixed_rule_set_is_refused_before_any_union_is_applied() {
    let mut egraph = seeded_graph();
    let before = egraph.class_count();
    let rules: Vec<Box<dyn Rule<Toy>>> = vec![Box::new(ProvedRule), Box::new(UnprovedRule)];
    expect_refusal(try_saturate_named(&mut egraph, "toy", &rules, 4));
    assert_eq!(
        egraph.class_count(),
        before,
        "a refused rule set must leave the egraph untouched"
    );
}

/// The refusal precedes the empty-rule-set and zero-budget early returns, so a
/// caller cannot launder an unproved rule through a degenerate budget.
#[test]
fn a_zero_budget_does_not_admit_an_unproved_rule() {
    let mut egraph = seeded_graph();
    let rules: Vec<Box<dyn Rule<Toy>>> = vec![Box::new(UnprovedRule)];
    expect_refusal(try_saturate_named(&mut egraph, "toy", &rules, 0));
}

/// The infallible compatibility wrappers report the refusal and apply nothing.
#[test]
fn the_infallible_wrappers_apply_nothing_for_an_unproved_rule() {
    let mut egraph = seeded_graph();
    let before = egraph.class_count();
    let rules: Vec<Box<dyn Rule<Toy>>> = vec![Box::new(UnprovedRule)];

    assert_eq!(
        saturate(&mut egraph, &rules, 4),
        0,
        "a refused run consumes no iterations"
    );
    let report = saturate_with_report(&mut egraph, &rules, 4);
    assert_eq!(report.iters_used, 0);
    assert_eq!(report.applied_equivalences, 0);

    let families: [&dyn Family<Toy>; 1] = [&UnprovedFamily];
    assert!(
        saturate_per_family(&mut egraph, &families, |_| 4).is_empty(),
        "a refused family run reports no per-family telemetry"
    );
    assert_eq!(egraph.class_count(), before);
}

/// The refusal message names the rule and the corrective action, because a
/// contributor reading it has to know which of a family's rules to record.
#[test]
fn the_refusal_states_the_rule_and_the_fix() {
    let rendered = EGraphError::UnprovedRule {
        rule: "toy_unproved",
    }
    .to_string();
    assert!(
        rendered.contains("toy_unproved"),
        "message must name the rule: {rendered}"
    );
    assert!(
        rendered.contains("Fix:"),
        "message must state the corrective action: {rendered}"
    );
}
