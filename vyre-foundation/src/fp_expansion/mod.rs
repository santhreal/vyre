//! Strict-IEEE expansion of the f32 transcendentals a backend would otherwise
//! lower to an approximate native instruction.
//!
//! A backend is permitted to answer `sin` with an approximation ROM accurate to
//! a few ulps, and every shipped one does, which is why
//! [`BACKEND_TRANSCENDENTAL_ULP_BUDGET`](crate::fp_parity::BACKEND_TRANSCENDENTAL_ULP_BUDGET)
//! is two orders of magnitude wider than the elementary budget beside it. A
//! caller that needs one program to produce identical bits on a device and in
//! the reference interpreter cannot use that instruction at all.
//!
//! [`expand_strict_transcendentals`](crate::fp_expansion::expand_strict_transcendentals)
//! rewrites those operations into f32 add, subtract, multiply, minimum, maximum
//! and comparison, integer operations on the exponent and mantissa fields,
//! selects, and bit-preserving casts between
//! f32 and u32. Every one of those is correctly rounded and specified to the
//! bit, so under
//! [`FloatLoweringMode::StrictIeee`](crate::fp_parity::FloatLoweringMode::StrictIeee),
//! which blocks multiply-add contraction, the device and the reference evaluate
//! the same arithmetic in the same order and agree bit for bit by construction.
//!
//! Bit identity with `libm` is a different and unreachable target: `libm`'s f32
//! transcendentals are not correctly rounded, and reproducing them needs the
//! f64 arithmetic they are computed in. Accuracy is instead bounded against
//! `libm` by
//! [`REFERENCE_TRANSCENDENTAL_ULP_BUDGET`](crate::fp_parity::REFERENCE_TRANSCENDENTAL_ULP_BUDGET).
//!
//! Two departures from the exact real function are stated in the contract
//! rather than hidden:
//!
//! - A subnormal argument flushes to a zero of its own sign, and a result below
//!   the smallest positive normal f32 flushes to `+0.0`. A subnormal cannot be
//!   bit-identical across a device that flushes it and a reference that does
//!   not, and [`canonical_f32`](crate::fp_parity::canonical_f32) already
//!   applies exactly this flush to every f32 the parity contract compares.
//! - `sin` and `cos` are defined for `|x| <= 401` and produce one quiet NaN
//!   outside it. An f32-only argument reduction cannot reach further, because
//!   401 is where the exact product of the reduction quotient with the leading
//!   limb of `pi/2` runs out of mantissa bits. The alternative, a saturated
//!   reduction, answers a larger argument with the value at the boundary, which
//!   is a wrong number rather than a refused one.

mod constants;
mod expansions;

/// The boundaries the contract above states. A coefficient is not one of them:
/// the polynomials are an implementation of the ulp bound, not a promise about
/// which polynomial produces it.
pub use constants::{
    EXP_OVERFLOW_ABOVE_BITS, EXP_UNDERFLOW_BELOW_BITS, MIN_NORMAL_BITS, NAN_RESULT_BITS,
    SIN_COS_DOMAIN_BITS, SIN_COS_SMALL_ARGUMENT_BITS,
};

use crate::fp_parity::{approximable_unary_ops, is_approximable_unary_op};
use crate::ir::{Expr, Program, UnOp};
use crate::optimizer::rewrite::rewrite_operand;
use crate::transform::rewrite_walk::{self, NodeRewrite};

/// An approximable f32 operation with no strict expansion.
///
/// Returned instead of leaving the operation approximate, because a strict
/// request that silently keeps a native transcendental produces exactly the
/// backend-dependent bits the mode exists to eliminate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictExpansionError {
    operation: String,
}

impl StrictExpansionError {
    /// The IR variant name of the operation with no expansion.
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.operation
    }
}

impl core::fmt::Display for StrictExpansionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "strict IEEE float lowering has no f32 expansion for `{}`. \
             Fix: dispatch this program under FloatLoweringMode::Contracted, or add an \
             expansion for `{}` to vyre_foundation::fp_expansion",
            self.operation, self.operation
        )
    }
}

impl std::error::Error for StrictExpansionError {}

/// Whether [`expand_strict_transcendentals`] replaces `op`.
///
/// The complement over [`approximable_unary_ops`] is
/// [`strict_unexpandable_unary_ops`], so the two answers are derived from one
/// classifier and an operator added to the approximability policy cannot land
/// in neither list.
#[must_use]
pub fn is_strict_expandable(op: &UnOp) -> bool {
    matches!(
        op,
        UnOp::Sin | UnOp::Cos | UnOp::Sqrt | UnOp::Exp | UnOp::Log
    )
}

/// Every approximable unary operator with no strict expansion, in wire tag
/// order.
#[must_use]
pub fn strict_unexpandable_unary_ops() -> Vec<UnOp> {
    approximable_unary_ops()
        .into_iter()
        .filter(|op| !is_strict_expandable(op))
        .collect()
}

/// The strict f32 expansion of `op` applied to `operand`, or `None` when `op`
/// has none.
///
/// The operand expression is duplicated across the expansion rather than bound
/// once: an `Expr` is a tree, and common-subexpression elimination in the
/// optimizer is what collapses the copies before emission.
#[must_use]
pub fn strict_expansion(op: &UnOp, operand: &Expr) -> Option<Expr> {
    match op {
        UnOp::Sin => Some(expansions::sine(operand)),
        UnOp::Cos => Some(expansions::cosine(operand)),
        UnOp::Sqrt => Some(expansions::square_root(operand)),
        UnOp::Exp => Some(expansions::exponential(operand)),
        UnOp::Log => Some(expansions::logarithm(operand)),
        _ => None,
    }
}

/// `program` with every approximable f32 operation replaced by its strict
/// expansion, or `None` when it contains none.
///
/// # Errors
///
/// Returns [`StrictExpansionError`] when the program reaches an approximable
/// operation with no expansion.
pub fn expand_strict_transcendentals(
    program: &Program,
) -> Result<Option<Program>, StrictExpansionError> {
    let mut policy = StrictExpansion { rejected: None };
    let entry = rewrite_walk::rewrite_body(program.entry(), &mut policy);
    if let Some(op) = policy.rejected {
        return Err(StrictExpansionError {
            operation: format!("{op:?}"),
        });
    }
    Ok(entry.map(|entry| program.with_rewritten_entry(entry)))
}

/// The rewrite policy: expand at every operand position of every node.
struct StrictExpansion {
    /// The first approximable operator with no expansion, if one was reached.
    rejected: Option<UnOp>,
}

impl NodeRewrite for StrictExpansion {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        let rejected = &mut self.rejected;
        rewrite_operand(expr, &mut |candidate| match candidate {
            Expr::UnOp { op, operand } if is_approximable_unary_op(op) => {
                let expanded = strict_expansion(op, operand);
                if expanded.is_none() && rejected.is_none() {
                    *rejected = Some(op.clone());
                }
                expanded
            }
            _ => None,
        })
    }
}
