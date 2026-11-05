//! Common-subexpression elimination  -  engine + ProgramPass registration colocated.
//!
//! Audit cleanup A6 (2026-04-30): hoisted from `transform/optimize/cse/`
//! so the engine lives next to its ProgramPass registration. The old
//! `transform/optimize/cse.rs` re-export shim is gone; downstream callers
//! use `crate::optimizer::passes::fusion_cse::cse::engine::cse(program)`
//! for the raw entry point or run the `CsePass` through the scheduler.
//!
//! ## Layout
//!
//! - `engine`  -  the `cse(program)` and `cse_into(program, &mut ctx)`
//!   entry points; thin wrappers over `CseCtx::nodes`.
//! - `cse_ctx.rs` + `impl_csectx.rs`  -  the per-pass scratchpad context
//!   (flat hashmap + scoped undo log + epoch invalidation; forks and
//!   side-effect clears cost O(1) unless actual bindings change).
//! - `expr_key.rs` + `impl_exprkey.rs`  -  structural-equality key for
//!   pure expressions; canonicalizes commutative operands.
//! - `expr_has_effect.rs`  -  conservative side-effect predicate (the
//!   safety gate that prevents merging effectful nodes).
//! - `type_key.rs` + `impl_typekey_from.rs`  -  compact `Copy` key for
//!   expression result types.
//! - `program_pass.rs`  -  the registered `CsePass` (ProgramPass impl) that
//!   hooks the engine into the scheduler's fixpoint loop.

/// Per-pass table of previously seen expression keys.
pub use cse_ctx::CseCtx;
pub(crate) use cse_ctx::{ScopeFrame, ScopedBinding};
/// Classify whether an expression is unsafe to merge.
pub(crate) use expr_has_effect::expr_has_effect;
/// Compact key for expression result types.
pub(crate) use type_key::TypeKey;

/// Per-pass context tracking seen expressions and their first binding name.
pub(crate) mod cse_ctx;
/// Raw `cse(program)` and `cse_into(program, &mut ctx)` entry points.
pub(crate) mod engine;
/// Conservative predicate: does this expression have observable side effects?
pub mod expr_has_effect;
pub(crate) mod expr_key;
pub(crate) mod impl_csectx;
pub(crate) mod impl_exprkey;
pub(crate) mod impl_typekey_from;
/// Registered `CsePass` (ProgramPass impl) for the engine.
pub(crate) mod program_pass;
pub(crate) mod type_key;

pub use engine::{cse, cse_into};
pub use program_pass::CsePass;

/// CSE test suites  -  adversarial cases for literal aliasing and non-literal
/// subexpression merging.
#[cfg(test)]
#[path = "../../../../../tests/internal/optimizer/passes/fusion_cse/cse/mod.rs"]
mod tests;
