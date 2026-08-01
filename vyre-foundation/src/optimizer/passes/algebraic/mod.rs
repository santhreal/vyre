//! Algebraic / peephole catalog (Phase 4A).
//!
//! Cost-monotone-down rewrites derived from algebraic identities:
//! constant folding, strength reduction, spec-driven rule firing,
//! canonical-form normalization, and atomic-shape normalization.

/// Collapse identity-op Relaxed atomic RMW to plain `Expr::Load`
/// (ROADMAP A36  -  narrow atomic minimization that needs no alias
/// proof).
pub mod atomic_minimize;
/// Canonical-form rewrite (audit P0 #32  -  registered ProgramPass).
pub mod canonicalize;
/// Canonicalization transform engine  -  pure IR-to-IR fn body used by the
/// `CanonicalizePass` ProgramPass.
pub mod canonicalize_engine;
/// Compile-time constant folding.
pub mod const_fold;
/// Hardware quirk normalization (atomic ordering / scope edge cases).
pub mod normalize_atomics;
/// Mixed-precision + transcendental fast-path emit hints
/// (ROADMAP G1 + G5 foundation half).
pub mod precision_hint;
/// Algebraic rewrites derived from operation specifications.
/// Multiplication strength reduction.
pub mod strength_reduce;

use crate::ir::Program;
use crate::optimizer::PassAnalysis;

pub(crate) fn expression_bearing_analysis(program: &Program) -> PassAnalysis {
    if program
        .stats()
        .has_any_node_kind(crate::ir::stats::NODE_KIND_EXPRESSION_BEARING_MASK)
    {
        PassAnalysis::RUN
    } else {
        PassAnalysis::SKIP
    }
}
