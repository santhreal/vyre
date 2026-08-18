//! Algebraic structure and semiring operators for tensor contractions.

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
    #[must_use]
    pub fn identity_expr(&self, dtype: &DataType) -> Expr {
        match self {
            Self::Standard => match dtype {
                DataType::F32 => Expr::f32(0.0),
                DataType::F64 => Expr::f64(0.0),
                _ => Expr::u32(0),
            },
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
