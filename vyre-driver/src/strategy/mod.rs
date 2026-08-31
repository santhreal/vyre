//! Backend-specific lowering strategies.
//!
//! # Two-Layer Optimization Architecture
//!
//! Vyre separates optimizations into two layers with clear separation of
//! concerns:
//!
//! ## Layer 1  -  IR-Level Passes (`vyre-foundation/src/optimizer/passes/`)
//!
//! Pure mathematical rewrites that transform `Expr → Expr` in the IR.
//! Backend-agnostic  -  every backend benefits equally.
//!
//! | Pass | Example | Lives In |
//! |------|---------|----------|
//! | Strength reduce | `x / 7` → `mulhi(x, M) >> s` | `strength_reduce/` |
//! | Const fold | `3 + 4` → `7` | `const_fold/` |
//! | Shift-add decomp | `x * 5` → `(x<<2) + x` | `strength_reduce/` |
//! | FMA synthesis | `a*b + c` → `fma(a,b,c)` | `strength_reduce/` |
//! | Exact division | `(x*6)/3` → `x * inv(3)` | `strength_reduce/` |
//! | Lemire remainder | `x % 7` → `lowbits(x*M)*7>>32` | `strength_reduce/` |
//!
//! ## Layer 2  -  Backend Lowering Strategies (this module)
//!
//! Target-dependent emission decisions. These don't change WHAT the program
//! computes  -  they change HOW it's emitted for a specific chip/API.
//!
//! | Strategy | Backend | Effect |
//! |----------|---------|--------|
//! | primary-binary native multiply-high | backend | `MulHigh` → 1 instruction |
//! | secondary-text native multiply-high | backend | `MulHigh` → 1 instruction |
//! | 16-bit half-word decomp | target-text fallback | `MulHigh` → 14 ALU ops |
//! | Dual-issue FP32/INT32 | capable device | Division via FP pipeline |
//! | Matrix-core batching | capable device | Batched int8 multiply |
//!
//! # Adding a New Strategy
//!
//! 1. Implement [`crate::strategy::LoweringStrategy`] in your backend crate
//! 2. Register it via `inventory::submit!`
//! 3. The lowering pipeline auto-selects the highest-priority applicable
//!    strategy based on [`vyre_foundation::validate::BackendCapabilities`]
//!
//! # Vyre Law Zero
//!
//! > Runtime performance is sacred. No avoidable runtime overhead, ever.
//!
//! Layer 1 runs at compile time  -  zero cost.
//! Layer 2 runs at kernel compile time (once for the megakernel)  -  amortized to zero.
//! At GPU runtime, only the optimal native instructions execute.

use vyre_foundation::ir::{BinOp, Expr};
use vyre_foundation::optimizer::passes::algebraic::precision_hint::{
    PrecisionHint, TranscendentalOp,
};
use vyre_foundation::validate::BackendCapabilities;

/// A lowered expression ready for backend emission.
///
/// This is the output of a [`LoweringStrategy`]. It can be either a
/// rewritten Vyre `Expr` or a backend-specific opaque instruction
/// sequence (represented as a tagged enum for extensibility).
#[derive(Debug, Clone)]
pub enum LoweredExpr {
    /// Rewritten as a Vyre IR expression (most strategies do this).
    Expr(Expr),
    /// The strategy handled emission directly  -  the lowering pipeline
    /// should not process this expression further.
    Emitted,
}

/// A backend-specific lowering strategy.
///
/// Strategies are the extensibility point for target-dependent
/// optimizations. Each strategy declares:
/// - **what** it can optimize (via [`can_apply`](LoweringStrategy::can_apply))
/// - **how well** (via [`priority`](LoweringStrategy::priority))
/// - **the transformation** (via [`lower`](LoweringStrategy::lower))
///
/// The lowering pipeline selects the highest-priority applicable
/// strategy for each expression.
pub trait LoweringStrategy: Send + Sync + std::fmt::Debug {
    /// Human-readable name for diagnostics and telemetry.
    fn name(&self) -> &str;

    /// Check whether this strategy applies given the target capabilities
    /// and the expression being lowered.
    fn can_apply(&self, caps: &BackendCapabilities, op: &BinOp) -> bool;

    /// Priority for strategy selection. Higher = preferred.
    ///
    /// Guidelines:
    /// - 100: native hardware instruction (OpUMulExtended, mul.hi.u32)
    /// - 50: multi-instruction but optimal (dual-issue trick)
    /// - 10: portable decomposition (16-bit arithmetic expansion)
    fn priority(&self) -> u32;

    /// Lower the given expression using this strategy.
    ///
    /// `left` and `right` are the operands of the binary operation.
    /// The strategy may return a rewritten `Expr` or signal that it
    /// handled emission directly.
    fn lower(&self, op: &BinOp, left: &Expr, right: &Expr) -> LoweredExpr;
}

/// Select the best available strategy for the given operation.
///
/// Returns `None` if no registered strategy applies, in which case
/// the lowering pipeline should use its default emission path.
pub fn select_strategy<'a>(
    strategies: &'a [Box<dyn LoweringStrategy>],
    caps: &BackendCapabilities,
    op: &BinOp,
) -> Option<&'a dyn LoweringStrategy> {
    strategies
        .iter()
        .filter(|s| s.can_apply(caps, op))
        .max_by_key(|s| s.priority())
        .map(|s| s.as_ref())
}

/// Concrete lower/emit plan selected from a foundation precision hint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionLoweringPlan {
    /// Keep the default f32/device-transcendental lowering.
    DefaultF32,
    /// Emit this site through native f16 ALU and widen the result to f32.
    NativeF16 {
        /// Maximum absolute source operand carried from the foundation hint.
        max_abs_operand: f32,
    },
    /// Emit a bounded polynomial for the transcendental instead of a native
    /// device call.
    PolynomialTranscendental {
        /// Target operation.
        op: TranscendentalOp,
        /// Maximum absolute argument bound from the foundation hint.
        argument_bound: f32,
        /// Required backend-side polynomial degree.
        degree: u8,
    },
}

/// Select a backend-neutral lower/emit plan for a precision hint.
///
/// Foundation owns candidate discovery. This function owns the shared
/// capability gate every emitter uses before choosing the faster code shape.
#[must_use]
pub fn select_precision_lowering(
    caps: &BackendCapabilities,
    hint: &PrecisionHint,
) -> PrecisionLoweringPlan {
    match hint {
        PrecisionHint::F16Eligible { max_abs_operand } if caps.has_native_f16 => {
            PrecisionLoweringPlan::NativeF16 {
                max_abs_operand: *max_abs_operand,
            }
        }
        PrecisionHint::TranscendentalPolynomial { op, argument_bound }
            if caps.has_transcendental_polynomial_emit =>
        {
            PrecisionLoweringPlan::PolynomialTranscendental {
                op: *op,
                argument_bound: *argument_bound,
                degree: polynomial_degree_for(*op, *argument_bound),
            }
        }
        _ => PrecisionLoweringPlan::DefaultF32,
    }
}

fn polynomial_degree_for(op: TranscendentalOp, argument_bound: f32) -> u8 {
    match op {
        TranscendentalOp::Sin => {
            if argument_bound <= 0.25 {
                3
            } else {
                5
            }
        }
        TranscendentalOp::Cos => {
            if argument_bound <= 0.25 {
                4
            } else {
                6
            }
        }
        TranscendentalOp::Exp | TranscendentalOp::Ln => 5,
    }
}
