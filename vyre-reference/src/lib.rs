#![forbid(unsafe_code)]
#![allow(
    clippy::only_used_in_recursion,
    clippy::comparison_chain,
    clippy::ptr_arg
)]
//! Pure Rust reference interpreter for vyre IR programs.
//!
//! This module is the executable specification for IR semantics. It is
//! intentionally slow and direct: every current IR expression and node variant
//! has a named evaluator function.

/// Dual-reference trait and registry types.
pub mod dual;
/// Canonical dual implementations and reference evaluators.
pub mod dual_impls;
mod dual_registry;
pub use dual_registry::{dual_op_ids, resolve_dual, DualReferenceFacet};
mod error;
pub use error::ReferenceError;
mod reference_facet;
pub use reference_facet::{reference_facets, reference_fn, ReferenceFacet};
/// Runtime value representation for interpreter inputs and outputs.
pub mod value;

/// Atomic operation reference implementations.
pub mod atomics;
/// CPU operation traits used by concrete reference implementations.
pub mod cpu_op;
/// Canonical operation/reference-facet dispatch entry point.
///
/// Resolves semantic identity through `vyre-foundation` and invokes the
/// separate reference-owned facet.
pub mod dialect_dispatch;
/// Canonical reference execution tree.
pub mod execution;
/// Flat byte adapter used by [`crate::cpu_op::CpuOp`].
pub mod flat_cpu;
/// IEEE 754 strict floating-point utilities.
pub mod ieee754;
/// Subgroup simulator for lane-collective Cat-C ops.
pub mod subgroup;
/// Workgroup simulation: invocation IDs, shared memory.
pub mod workgroup;

mod float16;
mod oob;
mod ops;

/// A tally of out-of-bounds accesses the interpreter silently absorbed during a
/// tracked run, surfaces the masking that hides GPU/CPU parity hazards. See
/// [`reference_eval_oob_report`].
pub use oob::OobReport;

/// Test-only entry point that runs the hashmap interpreter over a Program.
#[cfg(test)]
pub use execution::eval_hashmap_reference;
/// Count arithmetic IR ops the reference interpreter executes in a scope (roofline /
/// complexity analysis) (a backend-agnostic dynamic operation count).
pub use execution::op_count::count_ops;
/// The interpreter's ABI: [`is_reference_input`] selects the buffers a caller must
/// supply a `Value` for, [`is_reference_output`] selects the buffers `reference_eval`
/// returns, and [`output_index`] locates a named output by that predicate, so test
/// harnesses derive both orderings from the interpreter instead of re-deriving (and
/// drifting from) them.
pub use execution::{is_reference_input, is_reference_output, output_index};
/// Execute a vyre Program on the pure Rust reference interpreter.
pub use execution::{
    reference_eval, reference_eval_lane_reversed, reference_eval_oob_report,
    reference_eval_with_dispatch, reference_eval_with_dispatch_oob_report,
    reference_eval_with_grid, run_arena_reference, run_arena_reference_with_dispatch,
    run_storage_graph,
};
