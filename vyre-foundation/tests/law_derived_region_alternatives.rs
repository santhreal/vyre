//! Alternatives come from declared laws, not from an anticipated recipe.
//!
//! WHY this suite exists: every rewrite in this tree used to be a hand-written
//! sequence for a shape somebody had in mind. Two consequences followed. A
//! reformulation nobody anticipated was unreachable however many laws the
//! operator declared, and a law registered for an operator changed nothing
//! until a pass was written to read it. `optimizer::law_saturation` derives the
//! rewrite set from `algebraic_law_registry` instead, so this suite asserts the
//! derivation, not a list of rewrites:
//!
//! - a declared law produces a rewrite, and withdrawing the law withdraws it;
//! - two laws compose into an alternative no rewrite names;
//! - the rounding law id derives no reassociation, which is the numerical
//!   contract carried by the law vocabulary rather than by a special case;
//! - every declared `AlgebraicLaw` has a recorded derivation or a recorded
//!   reason it derives nothing, and a new law turns this suite red.
//!
//! What this does NOT catch: whether a registered law is true of the operator
//! it is registered for. `verify-rewrite-proofs` discharges that question
//! against an SMT solver.

use std::collections::BTreeSet;

use vyre::ir::{BinOp, Expr};
use vyre_foundation::optimizer::law_saturation::{
    derive_alternatives, derived_rewrites, law_derivation, DerivedRewriteKind, LawDerivation,
    LawSaturationBudget, LawSaturationStop,
};
use vyre_spec::{AlgebraicLaw, MonotonicDirection};

/// Fewest `AlgebraicLaw` variants a working source enumeration can find.
const MIN_LAW_VARIANTS: usize = 20;

fn bin(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr::BinOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn load(buffer: &str) -> Expr {
    Expr::load(buffer, Expr::u32(0))
}

/// `(a + b) + c` over three distinct loads.
fn left_nested_sum() -> Expr {
    bin(BinOp::Add, bin(BinOp::Add, load("a"), load("b")), load("c"))
}

/// The law named by a declared variant name, with the smallest payload that
/// names the variant.
///
/// A hand-written mapping is what the closure test holds to the source: the
/// variant set is parsed from `vyre-spec` at run time, and a name with no row
/// here fails.
fn law_named(name: &str) -> Option<AlgebraicLaw> {
    fn check(_op: fn(&[u8]) -> Vec<u8>, _args: &[u32]) -> bool {
        true
    }
    Some(match name {
        "Commutative" => AlgebraicLaw::Commutative,
        "Associative" => AlgebraicLaw::Associative,
        "Identity" => AlgebraicLaw::Identity { element: 0 },
        "LeftIdentity" => AlgebraicLaw::LeftIdentity { element: 0 },
        "RightIdentity" => AlgebraicLaw::RightIdentity { element: 0 },
        "SelfInverse" => AlgebraicLaw::SelfInverse { result: 0 },
        "Idempotent" => AlgebraicLaw::Idempotent,
        "Absorbing" => AlgebraicLaw::Absorbing { element: 0 },
        "LeftAbsorbing" => AlgebraicLaw::LeftAbsorbing { element: 0 },
        "RightAbsorbing" => AlgebraicLaw::RightAbsorbing { element: 0 },
        "Involution" => AlgebraicLaw::Involution,
        "DeMorgan" => AlgebraicLaw::DeMorgan {
            inner_op: "and",
            dual_op: "or",
        },
        "Monotone" => AlgebraicLaw::Monotone,
        "Monotonic" => AlgebraicLaw::Monotonic {
            direction: MonotonicDirection::NonDecreasing,
        },
        "Bounded" => AlgebraicLaw::Bounded { lo: 0, hi: 1 },
        "Complement" => AlgebraicLaw::Complement {
            complement_op: "not",
            universe: u32::MAX,
        },
        "DistributiveOver" => AlgebraicLaw::DistributiveOver { over_op: "add" },
        "LatticeAbsorption" => AlgebraicLaw::LatticeAbsorption { dual_op: "min" },
        "InverseOf" => AlgebraicLaw::InverseOf { op: "add" },
        "Trichotomy" => AlgebraicLaw::Trichotomy {
            less_op: "lt",
            equal_op: "eq",
            greater_op: "gt",
        },
        "ZeroProduct" => AlgebraicLaw::ZeroProduct { holds: true },
        "Custom" => AlgebraicLaw::Custom {
            name: "custom",
            description: "custom predicate",
            arity: 1,
            check,
        },
        "CategoricalIdentity" => AlgebraicLaw::CategoricalIdentity,
        "CategoricalAssociative" => AlgebraicLaw::CategoricalAssociative,
        _ => return None,
    })
}

/// The `AlgebraicLaw` variant names `vyre-spec` declares, read from source.
fn declared_law_variants() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-spec")
        .join("src")
        .join("algebraic_law.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} to derive the law set: {err}"));
    let body = vyre_test_support::braced_body(&source, "pub enum AlgebraicLaw {")
        .unwrap_or_else(|| panic!("Fix: {path:?} no longer declares `pub enum AlgebraicLaw`"));
    vyre_test_support::top_level_variant_names(body)
}

/// Every declared law states what it derives, or why it derives nothing.
#[test]
fn every_declared_law_has_a_recorded_derivation() {
    let declared = declared_law_variants();
    assert!(
        declared.len() >= MIN_LAW_VARIANTS,
        "Fix: the AlgebraicLaw source enumeration found only {} variants, below the floor of \
         {MIN_LAW_VARIANTS}; the scan is broken, not the enum",
        declared.len()
    );

    for name in &declared {
        let law = law_named(name).unwrap_or_else(|| {
            panic!(
                "Fix: AlgebraicLaw::{name} has no row in law_named; add one so its derivation is \
                 judged"
            )
        });
        assert_ne!(
            law_derivation(&law),
            LawDerivation::Unrecorded,
            "Fix: AlgebraicLaw::{name} has no recorded derivation; state the region rewrites it \
             authorizes in law_saturation::law_rule::law_derivation, or record why it authorizes \
             none"
        );
    }
}

/// An exact combine derives the rewrites its registered laws state.
#[test]
fn an_exact_combine_derives_its_registered_rewrites() {
    let rewrites = derived_rewrites(true);
    let add_kinds: BTreeSet<String> = rewrites
        .iter()
        .filter(|rewrite| rewrite.op == BinOp::Add)
        .map(|rewrite| format!("{:?}", rewrite.kind))
        .collect();
    assert!(
        add_kinds.iter().any(|kind| kind.contains("Commute")),
        "Fix: the exact add combine registers Commutative but no commute rewrite was derived: \
         {add_kinds:?}"
    );
    assert!(
        add_kinds.iter().any(|kind| kind.contains("Reassociate")),
        "Fix: the exact add combine registers Associative but no reassociation was derived: \
         {add_kinds:?}"
    );
    assert!(
        add_kinds.iter().any(|kind| kind.contains("Identity")),
        "Fix: the exact add combine registers an identity element but no identity rewrite was \
         derived: {add_kinds:?}"
    );
}

/// Every derived rewrite carries evidence candidate search accepts.
///
/// Saturation refuses a rewrite whose witness records opacity, so a derivation
/// that produced one would abort the run rather than contribute to it.
#[test]
fn every_derived_rewrite_is_admissible_in_candidate_search() {
    for rewrite in derived_rewrites(true) {
        assert!(
            rewrite.witness.admits_candidate_search(),
            "Fix: derived rewrite {} carries a witness that records opacity",
            rewrite.name
        );
    }
}

/// A rounding element type derives no reassociation.
#[test]
fn a_rounding_element_type_derives_no_reassociation() {
    let rounding: Vec<&'static str> = derived_rewrites(false)
        .into_iter()
        .filter(|rewrite| {
            rewrite.op == BinOp::Add
                && matches!(
                    rewrite.kind,
                    DerivedRewriteKind::ReassociateLeft | DerivedRewriteKind::ReassociateRight
                )
        })
        .map(|rewrite| rewrite.name)
        .collect();
    assert!(
        rounding.is_empty(),
        "Fix: the rounding add law id declares no associativity, but reassociation was still \
         derived: {rounding:?}"
    );

    let commutes = derived_rewrites(false)
        .into_iter()
        .any(|rewrite| rewrite.op == BinOp::Add && rewrite.kind == DerivedRewriteKind::Commute);
    assert!(
        commutes,
        "Fix: the rounding add law id declares commutativity, so the commute rewrite must still \
         be derived"
    );
}

/// The derivation reaches a regrouping of an exact sum.
#[test]
fn an_exact_sum_reassociates() {
    let derived = derive_alternatives(&left_nested_sum(), true, LawSaturationBudget::default())
        .expect("the mirror and the expansion loop must not fail on a three-operand sum");
    assert!(
        derived.report.applied_equivalences > 0,
        "Fix: no equality was derived for an associative and commutative sum: {:?}",
        derived.report
    );

    let mut mirror = derived.mirror;
    let right_nested = bin(BinOp::Add, load("a"), bin(BinOp::Add, load("b"), load("c")));
    assert!(
        mirror
            .holds_equivalent(&right_nested)
            .expect("membership must not fail"),
        "Fix: the declared associativity law did not equate the two groupings of the same sum"
    );
}

/// A rounding sum's two groupings stay apart.
///
/// This is the same expression and the same mechanism as
/// [`an_exact_sum_reassociates`], with only the element exactness changed, so a
/// derivation that ignored the numerical contract would be visible here and
/// nowhere else.
#[test]
fn a_rounding_sum_does_not_reassociate() {
    let derived = derive_alternatives(&left_nested_sum(), false, LawSaturationBudget::default())
        .expect("the mirror and the expansion loop must not fail on a three-operand sum");
    let mut mirror = derived.mirror;
    let right_nested = bin(BinOp::Add, load("a"), bin(BinOp::Add, load("b"), load("c")));
    assert!(
        !mirror
            .holds_equivalent(&right_nested)
            .expect("membership must not fail"),
        "Fix: a rounding element type reassociated; two orders of the same addends round \
         differently, so the derivation must not equate them"
    );
}

/// Two laws compose into an alternative no rewrite names.
///
/// `(a + b) + c` and `c + (b + a)` are equal by associativity and
/// commutativity together. No derived rewrite states that pair: each states one
/// law over one operator, and the composition is what the bounded expansion
/// finds.
#[test]
fn two_laws_compose_into_an_alternative_no_rewrite_names() {
    let derived = derive_alternatives(&left_nested_sum(), true, LawSaturationBudget::default())
        .expect("the mirror and the expansion loop must not fail on a three-operand sum");
    let mut mirror = derived.mirror;
    let reversed = bin(BinOp::Add, load("c"), bin(BinOp::Add, load("b"), load("a")));
    assert!(
        mirror
            .holds_equivalent(&reversed)
            .expect("membership must not fail"),
        "Fix: composing the declared associativity and commutativity laws must reach \
         c + (b + a); the bounded expansion stopped at {:?}",
        derived.report
    );
}

/// The declared identity element folds without a rule naming the shape.
///
/// The extracted representative's operand order is not asserted: commutativity
/// makes `a + b` and `b + a` the same class at the same cost, so the fold is
/// stated as the absence of the identity literal plus equivalence to the
/// folded term.
#[test]
fn a_declared_identity_element_folds() {
    let with_zero = bin(
        BinOp::Add,
        bin(BinOp::Add, load("a"), Expr::u32(0)),
        load("b"),
    );
    let derived = derive_alternatives(&with_zero, true, LawSaturationBudget::default())
        .expect("the mirror and the expansion loop must not fail");
    let best = derived
        .best
        .expect("extraction must stay inside its depth budget on a four-node term");
    assert!(
        !mentions_literal_zero(&best),
        "Fix: the registered identity element of the exact add combine must let extraction drop \
         `+ 0`; the smallest derived term was {best:?}"
    );

    let mut mirror = derived.mirror;
    let folded = bin(BinOp::Add, load("a"), load("b"));
    assert!(
        mirror
            .holds_equivalent(&folded)
            .expect("membership must not fail"),
        "Fix: dropping the identity element must equate the term with a + b"
    );
}

/// Whether any subexpression is the literal `0`.
fn mentions_literal_zero(expr: &Expr) -> bool {
    match expr {
        Expr::LitU32(0) => true,
        Expr::BinOp { left, right, .. } => {
            mentions_literal_zero(left) || mentions_literal_zero(right)
        }
        _ => false,
    }
}

/// A zero budget derives nothing and says so.
#[test]
fn a_zero_iteration_budget_derives_nothing() {
    let budget = LawSaturationBudget {
        iterations: 0,
        ..LawSaturationBudget::default()
    };
    let derived = derive_alternatives(&left_nested_sum(), true, budget)
        .expect("a zero budget is a stop reason, not an error");
    assert_eq!(derived.report.stop_reason, LawSaturationStop::ZeroBudget);
    assert_eq!(derived.report.applied_equivalences, 0);
    assert_eq!(
        derived.report.class_count_before, derived.report.class_count_after,
        "Fix: a zero-budget run grew the graph"
    );
}

/// A class budget stops the expansion and says so.
///
/// The stop reason is asserted as well as the bound: a run that stopped
/// because it converged and a run that stopped because it reached its class
/// budget hold different alternatives, and a caller that cannot tell them
/// apart cannot decide whether raising the bound is worth a second run.
#[test]
fn a_class_budget_stops_the_expansion() {
    let budget = LawSaturationBudget {
        classes: 8,
        ..LawSaturationBudget::default()
    };
    let derived = derive_alternatives(&left_nested_sum(), true, budget)
        .expect("a reached bound is a stop reason, not an error");
    assert_eq!(
        derived.report.stop_reason,
        LawSaturationStop::ExpansionBudget,
        "Fix: the run reached its class budget and reported {:?}",
        derived.report
    );
    assert!(
        derived.report.class_count_after <= 8 + 2,
        "Fix: the expansion passed its class budget: {:?}",
        derived.report
    );
}
