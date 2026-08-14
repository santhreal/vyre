//! Liquid-type / shape-predicate SMT-style evaluator (P-PRIM-15).
//!
//! Evaluates refinement predicates over a concrete `u32` count. The grammar
//! IS `vyre_foundation::ir::ShapePredicate`, re-exported here as
//! `ShapeFormula`, so validation and primitive evaluation cannot drift. The
//! evaluator is what lives at the primitive layer; the predicate type comes
//! from the foundation, which is why the `types` feature depends on it.
//!
//! Supported predicates cover bounds, exact cardinality, divisibility,
//! modular equality, Boolean composition, and affine ranges over the
//! count. That gives optimizer passes a single primitive predicate
//! substrate instead of pass-local proof encodings.

use vyre_foundation::ir::ShapePredicate;

/// Refinement predicate over a `u32` count.
///
/// The primitive layer uses the canonical foundation predicate directly so
/// validation and primitive evaluation cannot drift.
pub type ShapeFormula = ShapePredicate;

/// Evaluate a [`ShapeFormula`] against a `count`. Free function form
/// for callers that prefer `evaluate(formula, count)` over the
/// method-call style.
#[must_use]
pub fn evaluate(formula: &ShapeFormula, count: u32) -> bool {
    formula.evaluate(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_holds_at_or_above() {
        let f = ShapeFormula::AtLeast(8);
        assert!(!f.evaluate(7));
        assert!(f.evaluate(8));
        assert!(f.evaluate(100));
    }

    #[test]
    fn at_most_holds_at_or_below() {
        let f = ShapeFormula::AtMost(8);
        assert!(f.evaluate(0));
        assert!(f.evaluate(8));
        assert!(!f.evaluate(9));
    }

    #[test]
    fn exactly_only_at_match() {
        let f = ShapeFormula::Exactly(64);
        assert!(!f.evaluate(63));
        assert!(f.evaluate(64));
        assert!(!f.evaluate(65));
    }

    #[test]
    fn multiple_of_zero_never_holds() {
        let f = ShapeFormula::MultipleOf(0);
        for c in [0u32, 1, 7, 8, u32::MAX] {
            assert!(!f.evaluate(c), "MultipleOf(0) must never hold (c = {c})");
        }
    }

    #[test]
    fn multiple_of_alignment() {
        let f = ShapeFormula::MultipleOf(4);
        for &(c, expect) in &[(0u32, true), (3, false), (4, true), (7, false), (16, true)] {
            assert_eq!(f.evaluate(c), expect, "c = {c}");
        }
    }

    #[test]
    fn conjunction_requires_both() {
        let f = ShapeFormula::And(
            Box::new(ShapeFormula::AtLeast(8)),
            Box::new(ShapeFormula::MultipleOf(4)),
        );
        // Need >= 8 AND % 4 == 0.
        assert!(!f.evaluate(0));
        assert!(!f.evaluate(7));
        assert!(!f.evaluate(8 + 1)); // 9 not multiple of 4
        assert!(f.evaluate(8));
        assert!(f.evaluate(12));
    }

    #[test]
    fn disjunction_accepts_either_side() {
        let f = ShapeFormula::Or(
            Box::new(ShapeFormula::Exactly(4)),
            Box::new(ShapeFormula::Exactly(8)),
        );
        assert!(f.evaluate(4));
        assert!(f.evaluate(8));
        assert!(!f.evaluate(6));
    }

    #[test]
    fn negation_inverts_predicate() {
        let f = ShapeFormula::Not(Box::new(ShapeFormula::AtMost(8)));
        assert!(!f.evaluate(8));
        assert!(f.evaluate(9));
    }

    #[test]
    fn modular_equality_requires_canonical_remainder() {
        let f = ShapeFormula::ModEquals {
            modulus: 8,
            remainder: 3,
        };
        assert!(f.evaluate(11));
        assert!(!f.evaluate(12));
        assert!(!ShapeFormula::ModEquals {
            modulus: 8,
            remainder: 8,
        }
        .evaluate(8));
        assert!(!ShapeFormula::ModEquals {
            modulus: 0,
            remainder: 0,
        }
        .evaluate(0));
    }

    #[test]
    fn affine_range_uses_wide_arithmetic() {
        let f = ShapeFormula::AffineRange {
            scale: 2,
            offset: -4,
            min: 12,
            max: 20,
        };
        assert!(!f.evaluate(7));
        assert!(f.evaluate(8));
        assert!(f.evaluate(12));
        assert!(!f.evaluate(13));
        assert!(!ShapeFormula::AffineRange {
            scale: i64::MAX,
            offset: i64::MAX,
            min: i64::MIN,
            max: i64::MAX,
        }
        .evaluate(u32::MAX));
    }

    #[test]
    fn proves_non_empty_at_least() {
        assert!(ShapeFormula::AtLeast(1).proves_non_empty());
        assert!(ShapeFormula::AtLeast(64).proves_non_empty());
        assert!(!ShapeFormula::AtLeast(0).proves_non_empty());
    }

    #[test]
    fn proves_non_empty_exactly() {
        assert!(ShapeFormula::Exactly(1).proves_non_empty());
        assert!(!ShapeFormula::Exactly(0).proves_non_empty());
    }

    #[test]
    fn proves_non_empty_through_conjunction() {
        // (count >= 8) && (count % 4 == 0)  -  first disjunct proves
        // non-empty.
        let f = ShapeFormula::And(
            Box::new(ShapeFormula::AtLeast(8)),
            Box::new(ShapeFormula::MultipleOf(4)),
        );
        assert!(f.proves_non_empty());
    }

    #[test]
    fn proves_non_empty_no_lower_bound() {
        // (count <= 256) && (count % 4 == 0)  -  neither bounds count > 0.
        let f = ShapeFormula::And(
            Box::new(ShapeFormula::AtMost(256)),
            Box::new(ShapeFormula::MultipleOf(4)),
        );
        assert!(!f.proves_non_empty());
    }

    #[test]
    fn proves_non_empty_for_modular_and_boolean_forms() {
        assert!(ShapeFormula::ModEquals {
            modulus: 4,
            remainder: 1,
        }
        .proves_non_empty());
        assert!(!ShapeFormula::ModEquals {
            modulus: 4,
            remainder: 0,
        }
        .proves_non_empty());
        assert!(ShapeFormula::Or(
            Box::new(ShapeFormula::AtLeast(1)),
            Box::new(ShapeFormula::Exactly(7)),
        )
        .proves_non_empty());
        assert!(!ShapeFormula::Or(
            Box::new(ShapeFormula::AtLeast(0)),
            Box::new(ShapeFormula::Exactly(7)),
        )
        .proves_non_empty());
    }

    /// Closure-bar: free-function form must agree with method form.
    #[test]
    fn free_function_matches_method() {
        let f = ShapeFormula::And(
            Box::new(ShapeFormula::AtLeast(8)),
            Box::new(ShapeFormula::AtMost(64)),
        );
        for c in [0u32, 7, 8, 16, 64, 65, 100] {
            assert_eq!(f.evaluate(c), evaluate(&f, c), "drift on c = {c}");
        }
    }

    /// Adversarial: u32::MAX boundary on AtLeast / AtMost must not
    /// overflow.
    #[test]
    fn u32_max_boundary() {
        assert!(ShapeFormula::AtLeast(u32::MAX).evaluate(u32::MAX));
        assert!(!ShapeFormula::AtLeast(u32::MAX).evaluate(u32::MAX - 1));
        assert!(ShapeFormula::AtMost(u32::MAX).evaluate(u32::MAX));
        assert!(ShapeFormula::AtMost(u32::MAX).evaluate(0));
    }

    /// describe() round-trip: rendering must contain the operand.
    #[test]
    fn describe_renders_operand() {
        assert!(ShapeFormula::AtLeast(42).describe().contains("42"));
        assert!(ShapeFormula::AtMost(7).describe().contains("7"));
        assert!(ShapeFormula::Exactly(64).describe().contains("64"));
        assert!(ShapeFormula::MultipleOf(4).describe().contains("4"));
    }
}
