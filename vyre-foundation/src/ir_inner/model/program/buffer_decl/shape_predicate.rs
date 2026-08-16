//! The refinement predicate a buffer binding attaches to its element count.

/// Refinement predicate over a buffer's element count (P-1.0-V3.1).
///
/// Represents a small grammar of constraints a `BufferDecl` author
/// can attach. The validator (P-1.0-V3.2) checks each predicate
/// against the program's static count and the optimizer (P-1.0-V3.3)
/// uses verified predicates to prove loop-bound and alignment
/// invariants for vectorization.
///
/// `None` (the default) is "unconstrained"; existing programs keep
/// their current behavior.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShapePredicate {
    /// `count >= n`. Holds when the runtime element count is at
    /// least `n`. Used to prove non-empty workgroup buffers and
    /// minimum vectorization tile sizes.
    AtLeast(u32),
    /// `count <= n`. Holds when the count never exceeds `n`. Used
    /// to bound dispatch sizes and prevent oversized allocations.
    AtMost(u32),
    /// `count == n`. The strongest constraint; the count is fixed.
    Exactly(u32),
    /// `count % n == 0`. Used for alignment proofs (e.g. SIMD lanes).
    MultipleOf(u32),
    /// `count % modulus == remainder`. Invalid modular forms evaluate
    /// false, so static validation catches impossible declarations.
    ModEquals {
        /// Divisor used by the modular equality.
        modulus: u32,
        /// Required remainder. Must be less than `modulus` to match.
        remainder: u32,
    },
    /// `min <= count * scale + offset <= max`, evaluated with wide
    /// arithmetic for frontend-derived affine constraints.
    AffineRange {
        /// Multiplicative coefficient applied to `count`.
        scale: i64,
        /// Constant term added after scaling.
        offset: i64,
        /// Inclusive lower bound for the affine expression.
        min: i64,
        /// Inclusive upper bound for the affine expression.
        max: i64,
    },
    /// Conjunction of two predicates (`p1 && p2`). Both must hold.
    And(Box<ShapePredicate>, Box<ShapePredicate>),
    /// Disjunction of two predicates (`p1 || p2`). Either may hold.
    Or(Box<ShapePredicate>, Box<ShapePredicate>),
    /// Negation of a predicate.
    Not(Box<ShapePredicate>),
}

impl ShapePredicate {
    /// Evaluate the predicate against a concrete `count`. Returns
    /// `true` when the predicate holds. P-1.0-V3.2 uses this from
    /// the `validate()` pass; P-1.0-V3.3 calls it from optimizer
    /// passes that need a yes/no proof.
    #[must_use]
    pub fn holds(&self, count: u32) -> bool {
        match self {
            Self::AtLeast(n) => count >= *n,
            Self::AtMost(n) => count <= *n,
            Self::Exactly(n) => count == *n,
            Self::MultipleOf(n) => *n != 0 && count % *n == 0,
            Self::ModEquals { modulus, remainder } => {
                *modulus != 0 && *remainder < *modulus && count % *modulus == *remainder
            }
            Self::AffineRange {
                scale,
                offset,
                min,
                max,
            } => {
                let value = i128::from(count) * i128::from(*scale) + i128::from(*offset);
                value >= i128::from(*min) && value <= i128::from(*max)
            }
            Self::And(a, b) => a.holds(count) && b.holds(count),
            Self::Or(a, b) => a.holds(count) || b.holds(count),
            Self::Not(inner) => !inner.holds(count),
        }
    }

    /// Evaluate the predicate against a concrete count.
    #[must_use]
    pub fn evaluate(&self, count: u32) -> bool {
        self.holds(count)
    }

    /// Whether this predicate proves that the count cannot be zero.
    #[must_use]
    pub fn proves_non_empty(&self) -> bool {
        match self {
            Self::AtLeast(n) | Self::Exactly(n) => *n > 0,
            Self::ModEquals { modulus, remainder } => {
                *modulus != 0 && *remainder < *modulus && *remainder > 0
            }
            Self::AffineRange {
                offset, min, max, ..
            } => {
                let zero_value = i128::from(*offset);
                zero_value < i128::from(*min) || zero_value > i128::from(*max)
            }
            Self::And(left, right) => left.proves_non_empty() || right.proves_non_empty(),
            Self::Or(left, right) => left.proves_non_empty() && right.proves_non_empty(),
            _ => false,
        }
    }

    /// Human-readable form for error messages.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::AtLeast(n) => format!("count >= {n}"),
            Self::AtMost(n) => format!("count <= {n}"),
            Self::Exactly(n) => format!("count == {n}"),
            Self::MultipleOf(n) => format!("count % {n} == 0"),
            Self::ModEquals { modulus, remainder } => format!("count % {modulus} == {remainder}"),
            Self::AffineRange {
                scale,
                offset,
                min,
                max,
            } => {
                format!("{min} <= count * {scale} + {offset} <= {max}")
            }
            Self::And(a, b) => format!("({}) && ({})", a.describe(), b.describe()),
            Self::Or(a, b) => format!("({}) || ({})", a.describe(), b.describe()),
            Self::Not(inner) => format!("!({})", inner.describe()),
        }
    }
}

#[cfg(test)]
mod shape_predicate_tests {
    use super::*;
    use crate::ir_inner::model::program::BufferDecl;
    use crate::ir_inner::model::op_signature::DataType;

    #[test]
    fn at_least_holds_when_count_meets_minimum() {
        let p = ShapePredicate::AtLeast(64);
        assert!(p.holds(64));
        assert!(p.holds(128));
        assert!(!p.holds(32));
    }

    #[test]
    fn at_most_holds_when_count_within_bound() {
        let p = ShapePredicate::AtMost(64);
        assert!(p.holds(0));
        assert!(p.holds(64));
        assert!(!p.holds(65));
    }

    #[test]
    fn exactly_holds_only_for_match() {
        let p = ShapePredicate::Exactly(7);
        assert!(p.holds(7));
        assert!(!p.holds(6));
        assert!(!p.holds(8));
    }

    #[test]
    fn multiple_of_holds_for_aligned_count() {
        let p = ShapePredicate::MultipleOf(64);
        assert!(p.holds(0));
        assert!(p.holds(64));
        assert!(p.holds(128));
        assert!(!p.holds(63));
        assert!(!p.holds(65));
    }

    #[test]
    fn multiple_of_zero_never_holds() {
        let p = ShapePredicate::MultipleOf(0);
        assert!(!p.holds(0));
        assert!(!p.holds(64));
    }

    #[test]
    fn and_combines_two_predicates() {
        // count >= 64 && count % 32 == 0
        let p = ShapePredicate::And(
            Box::new(ShapePredicate::AtLeast(64)),
            Box::new(ShapePredicate::MultipleOf(32)),
        );
        assert!(p.holds(64));
        assert!(p.holds(96));
        assert!(!p.holds(32)); // satisfies MultipleOf but not AtLeast
        assert!(!p.holds(80)); // satisfies AtLeast but not MultipleOf
    }

    #[test]
    fn or_accepts_either_predicate() {
        let p = ShapePredicate::Or(
            Box::new(ShapePredicate::Exactly(8)),
            Box::new(ShapePredicate::Exactly(16)),
        );
        assert!(p.holds(8));
        assert!(p.holds(16));
        assert!(!p.holds(12));
    }

    #[test]
    fn not_inverts_predicate() {
        let p = ShapePredicate::Not(Box::new(ShapePredicate::AtMost(64)));
        assert!(!p.holds(64));
        assert!(p.holds(65));
    }

    #[test]
    fn mod_equals_requires_valid_modular_form() {
        assert!(ShapePredicate::ModEquals {
            modulus: 16,
            remainder: 4,
        }
        .holds(20));
        assert!(!ShapePredicate::ModEquals {
            modulus: 16,
            remainder: 4,
        }
        .holds(21));
        assert!(!ShapePredicate::ModEquals {
            modulus: 0,
            remainder: 0,
        }
        .holds(0));
        assert!(!ShapePredicate::ModEquals {
            modulus: 4,
            remainder: 4,
        }
        .holds(4));
    }

    #[test]
    fn affine_range_uses_wide_arithmetic() {
        let p = ShapePredicate::AffineRange {
            scale: 4,
            offset: -8,
            min: 24,
            max: 40,
        };
        assert!(!p.holds(7));
        assert!(p.holds(8));
        assert!(p.holds(12));
        assert!(!p.holds(13));
        assert!(!ShapePredicate::AffineRange {
            scale: i64::MAX,
            offset: i64::MAX,
            min: i64::MIN,
            max: i64::MAX,
        }
        .holds(u32::MAX));
    }

    #[test]
    fn buffer_decl_default_shape_predicate_is_none() {
        let buf = BufferDecl::read("a", 0, DataType::U32);
        assert_eq!(buf.shape_predicate(), None);
    }

    #[test]
    fn with_shape_predicate_round_trip() {
        let buf = BufferDecl::read("a", 0, DataType::U32)
            .with_shape_predicate(ShapePredicate::MultipleOf(32));
        assert_eq!(buf.shape_predicate(), Some(&ShapePredicate::MultipleOf(32)));
    }

    #[test]
    fn describe_renders_human_readable() {
        assert_eq!(
            ShapePredicate::And(
                Box::new(ShapePredicate::AtLeast(64)),
                Box::new(ShapePredicate::MultipleOf(32)),
            )
            .describe(),
            "(count >= 64) && (count % 32 == 0)"
        );
    }
}
