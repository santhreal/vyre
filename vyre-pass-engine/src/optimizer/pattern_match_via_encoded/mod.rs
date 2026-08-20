//! Local-pattern rewrite engine as a dispatched compute kernel.
//!
//! V1 ships a hardcoded bank of algebraic-identity rewrites:
//!
//! - `Add 0 ?x   →   ?x`
//! - `Add ?x 0   →   ?x`
//! - `Mul 1 ?x   →   ?x`
//! - `Mul ?x 1   →   ?x`
//! - `Mul 0 ?x   →   0u32`
//! - `Mul ?x 0   →   0u32`
//!
//! Each rule fires per-Expr in a single GPU dispatch (no scope walk,
//! no structural-hash needed for this set). Output is a `rewrite_action`
//! buffer encoding the per-Expr decision; the decoder applies it.
//!
//! This is the architectural prototype for the universal pattern-match
//! engine: V2 takes the pattern bank as input buffers (kind/op/literal-
//! value templates per pattern) and runs the same kernel shape over
//! arbitrary rewrite rules sourced from a TOML database.
//! All the hardcoding below is a fixed instance of that more general
//! kernel.
//!
//! No host-reference escape in production. `ProgramDispatcher` injects the
//! backend; the same kernel runs unchanged on every backend.

mod bin_op_cse_rules;
mod bin_op_rules;
mod decode;
mod driver;
mod program;
/// Per-Expr rewrite-action discriminants written by the kernel.
pub mod rewrite_action;
mod rule_shapes;

pub use driver::{gpu_algebraic_identities, PatternMatchError};
pub use program::{build_pattern_match_program, build_pattern_match_program_with_cse};
