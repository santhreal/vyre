//! Algebraic structure and semiring operators for tensor contractions.

use crate::element_zero;
use std::sync::Arc;
use vyre_foundation::ir::{DataType, Expr};
use vyre_spec::Semiring;

/// Algebraic structure for the contraction inner product: `acc = ⊕ (lhs ⊗ rhs)`.
#[derive(Clone)]
pub enum ContractionSemiring {
    /// Standard arithmetic: `⊗ = *`, `⊕ = +`, identity = 0.
    Standard,
    /// Canonical semirings from [`Semiring`]:
    /// `Real`, `MinPlus`, `MaxPlus`, `BoolOr`, `BoolAnd`, `MaxTimes`, `Lineage`, `Gf2`.
    Closed(Semiring),
    /// Unsigned 16.16 fixed-point arithmetic (`fixed_mul_16_16`, `+`, identity = 0).
    Fixed16_16,
    /// Custom combine and accumulate expressions over `DataType::U32`.
    Custom {
        /// Additive identity value for initializing accumulator.
        identity: u32,
        /// Scalar combine operation.
        combine: Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>,
        /// Scalar accumulate operation.
        accumulate: Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>,
    },
}

impl core::fmt::Debug for ContractionSemiring {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::Closed(s) => write!(f, "Closed({s:?})"),
            Self::Fixed16_16 => write!(f, "Fixed16_16"),
            Self::Custom { identity, .. } => f
                .debug_struct("Custom")
                .field("identity", identity)
                .finish(),
        }
    }
}

impl ContractionSemiring {
    /// Additive identity used to initialize accumulators.
    ///
    /// The standard structure's identity is the element type's own zero, so an
    /// f16 accumulator is not seeded with a `u32` literal. An element type with
    /// no scalar zero falls back to `u32` zero, which the IR validator rejects
    /// at the store that writes it rather than accepting a silent mistype.
    #[must_use]
    pub fn identity_expr(&self, dtype: &DataType) -> Expr {
        match self {
            Self::Standard => element_zero(dtype).unwrap_or_else(|| Expr::u32(0)),
            Self::Closed(s) => match dtype {
                DataType::F32 => match s {
                    Semiring::MinPlus | Semiring::BoolAnd => Expr::f32(f32::INFINITY),
                    _ => Expr::f32(0.0),
                },
                _ => Expr::u32(s.identity()),
            },
            Self::Fixed16_16 => Expr::u32(0),
            Self::Custom { identity, .. } => Expr::u32(*identity),
        }
    }

    /// Scalar combine operation: `lhs ⊗ rhs`.
    #[must_use]
    pub fn combine_expr(&self, a: Expr, b: Expr) -> Expr {
        match self {
            Self::Standard => Expr::mul(a, b),
            Self::Closed(s) => semiring_combine_expr(*s, a, b),
            Self::Fixed16_16 => fixed_mul_16_16_signed_expr(a, b),
            Self::Custom { combine, .. } => combine(a, b),
        }
    }

    /// Accumulator update: `acc ⊕ value`.
    #[must_use]
    pub fn accumulate_expr(&self, acc: Expr, val: Expr) -> Expr {
        match self {
            Self::Standard => Expr::add(acc, val),
            Self::Closed(s) => semiring_accumulate_expr(*s, acc, val),
            Self::Fixed16_16 => Expr::add(acc, val),
            Self::Custom { accumulate, .. } => accumulate(acc, val),
        }
    }
}

/// Combine expression for canonical semirings.
#[must_use]
pub fn semiring_combine_expr(semiring: Semiring, a: Expr, b: Expr) -> Expr {
    match semiring {
        Semiring::Real | Semiring::MaxTimes => Expr::mul(a, b),
        Semiring::MinPlus => {
            let max_const = Expr::u32(u32::MAX);
            let either_inf = Expr::or(
                Expr::eq(a.clone(), max_const.clone()),
                Expr::eq(b.clone(), max_const.clone()),
            );
            Expr::select(either_inf, max_const, Expr::add(a, b))
        }
        Semiring::MaxPlus => Expr::add(a, b),
        Semiring::BoolOr | Semiring::Gf2 => Expr::bitand(a, b),
        Semiring::BoolAnd => Expr::bitor(a, b),
        Semiring::Lineage => {
            let either_zero = Expr::or(
                Expr::eq(a.clone(), Expr::u32(0)),
                Expr::eq(b.clone(), Expr::u32(0)),
            );
            Expr::select(either_zero, Expr::u32(0), Expr::bitor(a, b))
        }
    }
}

/// Accumulate expression for canonical semirings.
#[must_use]
pub fn semiring_accumulate_expr(semiring: Semiring, acc: Expr, val: Expr) -> Expr {
    match semiring {
        Semiring::Real => Expr::add(acc, val),
        Semiring::MinPlus => Expr::min(acc, val),
        Semiring::MaxPlus | Semiring::MaxTimes => Expr::max(acc, val),
        Semiring::BoolOr | Semiring::Lineage => Expr::bitor(acc, val),
        Semiring::BoolAnd => Expr::bitand(acc, val),
        Semiring::Gf2 => Expr::bitxor(acc, val),
    }
}

/// Signed 16.16 fixed-point multiplication over [`Expr`].
#[must_use]
pub fn fixed_mul_16_16_signed_expr(left: Expr, right: Expr) -> Expr {
    let low = Expr::mul(left.clone(), right.clone());
    let unsigned_high = Expr::mulhi(left.clone(), right.clone());
    let left_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(left.clone(), Expr::u32(31)));
    let right_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(right.clone(), Expr::u32(31)));
    let correction_left = Expr::bitand(left_sign_mask, right);
    let correction_right = Expr::bitand(right_sign_mask, left);
    let signed_high = Expr::sub(Expr::sub(unsigned_high, correction_left), correction_right);
    Expr::bitor(
        Expr::shr(low, Expr::u32(16)),
        Expr::shl(signed_high, Expr::u32(16)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `Semiring` must pair the identity its accumulator is seeded with
    /// against the accumulate it feeds, or a contraction starts from a value
    /// its own reduction can never absorb.
    ///
    /// The match has no catch-all, so adding a semiring fails to compile until
    /// its identity and accumulate are decided together.
    #[test]
    fn each_semiring_identity_is_absorbed_by_its_own_accumulate() {
        for semiring in [
            Semiring::Real,
            Semiring::MinPlus,
            Semiring::MaxPlus,
            Semiring::BoolOr,
            Semiring::BoolAnd,
            Semiring::MaxTimes,
            Semiring::Lineage,
            Semiring::Gf2,
        ] {
            let expected = match semiring {
                Semiring::Real => Expr::add(Expr::u32(7), Expr::u32(3)),
                Semiring::MinPlus => Expr::min(Expr::u32(7), Expr::u32(3)),
                Semiring::MaxPlus | Semiring::MaxTimes => Expr::max(Expr::u32(7), Expr::u32(3)),
                Semiring::BoolOr | Semiring::Lineage => Expr::bitor(Expr::u32(7), Expr::u32(3)),
                Semiring::BoolAnd => Expr::bitand(Expr::u32(7), Expr::u32(3)),
                Semiring::Gf2 => Expr::bitxor(Expr::u32(7), Expr::u32(3)),
            };
            assert_eq!(
                semiring_accumulate_expr(semiring, Expr::u32(7), Expr::u32(3)),
                expected,
                "{semiring:?} accumulate must be the reduction its identity seeds"
            );
            assert_eq!(
                ContractionSemiring::Closed(semiring).identity_expr(&DataType::U32),
                Expr::u32(semiring.identity()),
                "{semiring:?} must seed the accumulator with the semiring's own identity"
            );
        }
    }

    /// A float `MinPlus` or `BoolAnd` accumulator seeds from positive infinity,
    /// because zero is smaller than every value a min reduction could find and
    /// would pin the result at zero.
    #[test]
    fn a_float_min_accumulator_seeds_from_infinity_not_zero() {
        for semiring in [Semiring::MinPlus, Semiring::BoolAnd] {
            assert_eq!(
                ContractionSemiring::Closed(semiring).identity_expr(&DataType::F32),
                Expr::f32(f32::INFINITY),
                "{semiring:?} over F32 must seed from infinity"
            );
        }
        for semiring in [Semiring::Real, Semiring::MaxPlus, Semiring::MaxTimes] {
            assert_eq!(
                ContractionSemiring::Closed(semiring).identity_expr(&DataType::F32),
                Expr::f32(0.0),
                "{semiring:?} over F32 must seed from zero"
            );
        }
    }

    /// `Standard` is multiply-accumulate over the accumulator's own type.
    #[test]
    fn the_standard_semiring_seeds_zero_in_the_accumulator_type() {
        assert_eq!(
            ContractionSemiring::Standard.identity_expr(&DataType::F32),
            Expr::f32(0.0)
        );
        assert_eq!(
            ContractionSemiring::Standard.identity_expr(&DataType::F64),
            Expr::f64(0.0)
        );
        assert_eq!(
            ContractionSemiring::Standard.identity_expr(&DataType::U32),
            Expr::u32(0)
        );
        assert_eq!(
            ContractionSemiring::Standard.combine_expr(Expr::u32(6), Expr::u32(7)),
            Expr::mul(Expr::u32(6), Expr::u32(7))
        );
        assert_eq!(
            ContractionSemiring::Standard.accumulate_expr(Expr::u32(6), Expr::u32(7)),
            Expr::add(Expr::u32(6), Expr::u32(7))
        );
    }

    /// A custom semiring routes through the caller's closures and seeds from
    /// the identity the caller declared, not from zero.
    #[test]
    fn a_custom_semiring_uses_the_closures_and_identity_it_was_given() {
        let semiring = ContractionSemiring::Custom {
            identity: 5,
            combine: Arc::new(|a, b| Expr::bitxor(a, b)),
            accumulate: Arc::new(|acc, val| Expr::max(acc, val)),
        };
        assert_eq!(semiring.identity_expr(&DataType::U32), Expr::u32(5));
        assert_eq!(
            semiring.combine_expr(Expr::u32(1), Expr::u32(2)),
            Expr::bitxor(Expr::u32(1), Expr::u32(2))
        );
        assert_eq!(
            semiring.accumulate_expr(Expr::u32(1), Expr::u32(2)),
            Expr::max(Expr::u32(1), Expr::u32(2))
        );
    }

    /// `MinPlus` over integers encodes unreachability as `u32::MAX` and must
    /// keep it absorbing: adding to an unreachable edge stays unreachable
    /// rather than wrapping to a short path.
    #[test]
    fn integer_min_plus_combine_keeps_unreachable_absorbing() {
        let combined = semiring_combine_expr(Semiring::MinPlus, Expr::u32(4), Expr::u32(9));
        let max_const = Expr::u32(u32::MAX);
        let expected = Expr::select(
            Expr::or(
                Expr::eq(Expr::u32(4), max_const.clone()),
                Expr::eq(Expr::u32(9), max_const.clone()),
            ),
            max_const,
            Expr::add(Expr::u32(4), Expr::u32(9)),
        );
        assert_eq!(
            combined, expected,
            "an unreachable operand must select u32::MAX instead of wrapping through add"
        );
    }

    /// Fixed 16.16 multiplication must sign-correct the high half. Dropping
    /// either correction term silently makes every negative operand wrong.
    #[test]
    fn fixed_point_multiply_corrects_both_operand_signs() {
        let left = Expr::u32(0x0001_8000);
        let right = Expr::u32(0xFFFF_0000);
        let low = Expr::mul(left.clone(), right.clone());
        let unsigned_high = Expr::mulhi(left.clone(), right.clone());
        let left_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(left.clone(), Expr::u32(31)));
        let right_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(right.clone(), Expr::u32(31)));
        let expected = Expr::bitor(
            Expr::shr(low, Expr::u32(16)),
            Expr::shl(
                Expr::sub(
                    Expr::sub(unsigned_high, Expr::bitand(left_sign_mask, right.clone())),
                    Expr::bitand(right_sign_mask, left.clone()),
                ),
                Expr::u32(16),
            ),
        );
        assert_eq!(fixed_mul_16_16_signed_expr(left, right), expected);
    }
}
