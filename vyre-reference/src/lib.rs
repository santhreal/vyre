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
/// Independent sequential mathematical witnesses for composite operations.
pub mod composition_witness;
/// Runtime value representation for interpreter inputs and outputs.
pub mod value;
/// Re-exported versioned numeric semantics authority.
pub use vyre_spec::{
    dequantize_grouped_f32, f32_to_f8e4m3, f32_to_f8e5m2, f32_to_fp4, f32_to_nf4,
    f8e4m3_decode_table, f8e4m3_to_f32, f8e5m2_decode_table, f8e5m2_to_f32, fp4_to_f32, i32_to_i4,
    i4_to_i32, nf4_to_f32, numeric_semantics_for, InfinityBehavior, NanBehavior, NumericFormat,
    NumericSemantics, OverflowBehavior, RoundingMode, SaturationBehavior, SignedZeroBehavior,
    SubnormalBehavior, FP4_DECODE_TABLE, I4_DECODE_TABLE, NF4_QUANTILE_TABLE,
    NUMERIC_SEMANTICS_SCHEMA_VERSION,
};
/// Source-derived inventory and migration audit for host execution oracles.
pub mod host_oracle_migration;

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
pub(crate) mod execution;
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

pub use execution::{expr, node, op_count, sequential};
/// The interpreter's ABI: [`is_reference_input`] selects the buffers a caller must
/// supply a `Value` for, [`is_reference_output`] selects the buffers `reference_eval`
/// returns, and [`output_index`] locates a named output by that predicate, so test
/// harnesses derive both orderings from the interpreter instead of re-deriving (and
/// drifting from) them.
pub use execution::{is_reference_input, is_reference_output, output_index};
/// Execute a vyre Program on the pure Rust reference interpreter.
pub use execution::{
    reference_eval, reference_eval_lane_reversed, reference_eval_lane_rotated,
    reference_eval_oob_report, reference_eval_with_dispatch,
    reference_eval_with_dispatch_oob_report, reference_eval_with_grid, run_arena_reference,
    run_arena_reference_with_dispatch, run_storage_graph,
};
