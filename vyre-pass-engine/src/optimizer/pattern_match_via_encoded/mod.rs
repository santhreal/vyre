//! Local-pattern rewrite engine executed as a semantic analysis graph.
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
//! Each rule fires per expression in one analysis Program. The compiler selects
//! its physical schedule. The decoder applies the returned `rewrite_action`
//! buffer.
//!
//! This is the architectural prototype for the universal pattern-match
//! engine: V2 takes the pattern bank as input buffers (kind/op/literal-
//! value templates per pattern) and runs the same kernel shape over
//! arbitrary rewrite rules sourced from a TOML database.
//! All the hardcoding below is a fixed instance of that more general
//! kernel.
//!
//! The same schedule-free kernel is supplied to every semantic executor.

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
