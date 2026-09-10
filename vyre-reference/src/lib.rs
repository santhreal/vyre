//! Pure Rust reference interpreter for vyre IR programs.
//!
//! This module is the executable specification for IR semantics. It is
//! intentionally slow and direct: every current IR expression and node variant
//! has a named evaluator function.
//!
//! # What the oracle borrows from `vyre-foundation`
//!
//! A differential oracle that reuses the transform it is checking cannot fail
//! on a defect in that transform, so the interpreter runs the program as
//! submitted. It calls no optimizer pass, no schedule selection, no lowering
//! and no emitter, and it rewrites no node on the way in. A whole-grid fence
//! is interpreted where it stands: a lane that reaches
//! `MemoryOrdering::GridSync` suspends, the dispatch driver runs every other
//! workgroup to its own fence, and only then does any lane resume. The
//! interpreter used to cut the program into segments with
//! `vyre_foundation::transform::grid_sync_split`, which is the transform a
//! backend without a cooperative launch runs, so the oracle and the backend
//! shared one answer to where a fence divides a program.
//!
//! Four dependencies remain, and none of them is a program rewrite.
//!
//! - The `ir` types themselves. An oracle for a program has to read the same
//!   `Program`, `Node`, `Expr` and `BufferDecl` the compiler reads. What it
//!   does not share is what they mean: every semantic arm is evaluated here.
//! - `Program::reconcile_runnable_top_level`, applied by
//!   `execution::program_for_interpreter` when the submitted program is not
//!   top-level `Region`-wrapped. It re-applies the wrapper `Program::wrapped`
//!   builds and moves no other node, so it changes the shape the interpreter
//!   admits and not the semantics it evaluates.
//! - `validate::validate_with_options`, which decides admission. The oracle
//!   refuses exactly the programs the validator refuses, on purpose: a program
//!   the compiler rejects has no expected output for a backend to be wrong
//!   against. A validator that admits an illegal program therefore reaches
//!   both sides, and that is the one shared judgement left on this path.
//! - The operation registry behind `Expr::Call`, through
//!   `operation::OperationRegistry` and `cpu_op::CpuFn`. The bytes come from
//!   the registered CPU reference of the callee, which is a hand-written
//!   scalar implementation rather than a lowering of the call, so the call ABI
//!   is shared and the arithmetic is not.
//!
//! One canonical evaluator owns program execution, and
//! [`ReferenceRequest`] is the only way to reach it. A request states the
//! logical program, the exact resource ABI, the workload envelope, the
//! numerical contract, the schedule-exploration policy and a mandatory work,
//! memory and recursion budget; submitting it resolves the logical iteration
//! domain from the program's own declared extents, arms that budget and
//! interprets the program through `execution::hashmap`. The crate carried
//! four more routes into the same semantics and eleven untyped free functions
//! in front of them, most of which armed no budget at all. Each was a second
//! answer to what a node means, and a differential oracle with two answers
//! cannot say which one a backend must match, so they are gone rather than
//! kept in agreement.

mod error;
pub use error::{ReferenceError, ReferenceErrorClass, ReferenceErrorKind, StepCeilingExceeded};
/// Typed, versioned reference execution requests, contracts, and certificates.
mod request;
pub use request::{
    DeterministicSchedulePolicy, DiagnosticPermissiveReport, ExactResourceAbi, ReferenceBudget,
    ReferenceCertificate, ReferenceRequest, StrictExecutionResult, WorkloadEnvelope,
    REFERENCE_ORACLE_VERSION, REFERENCE_REQUEST_SCHEMA_VERSION,
};
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

/// Atomic operation reference implementations.
pub mod atomics;
/// Canonical reference execution tree.
pub(crate) mod execution;
/// IEEE 754 strict floating-point utilities.
pub mod ieee754;
/// Subgroup simulator for lane-collective Cat-C ops.
pub mod subgroup;
/// Workgroup simulation: invocation ids and workgroup memory limits.
pub mod workgroup;

/// Bounded legal interleaving and race freedom verification oracle.
mod interleaving;
pub use interleaving::{
    explore_bounded_interleavings, verify_closed_type_coverage_in_oracle, InterleavingConfig,
    InterleavingReport, MemoryAccessKind, MemoryAccessRecord, RaceExplorationReport, RaceFinding,
    ShadowMemory,
};

/// Step orders one [`ReferenceRequest::explore_races`] call runs at most.
///
/// The exploration is bounded so the oracle keeps a termination contract. A
/// caller reads the exact width of a given exploration from
/// [`ReferenceRequest::declared_race_exploration_orders`], which never exceeds
/// this ceiling.
pub const MAX_RACE_EXPLORATION_ORDERS: usize = execution::MAX_RACE_EXPLORATION_ORDERS;

mod float16;
mod oob;
mod ops;

/// Evaluate one expression for one lane through the canonical evaluator.
pub use execution::single_expr::{reference_eval_expr, ReferenceMemory};
/// The interpreter's ABI: [`is_reference_input`] selects the buffers a caller must
/// supply a `Value` for, [`is_reference_output`] selects the buffers a request
/// returns, [`output_index`] locates a named output by that predicate, and
/// [`reference_inputs`] projects a declaration-order buffer list onto the input
/// ABI, and [`reference_input_values`] does the same for borrowed bytes and
/// names the buffer it is short of, so test harnesses and backends derive every
/// ordering from the interpreter instead of re-deriving (and drifting from)
/// them.
pub use execution::{
    is_reference_input, is_reference_output, output_index, reference_input_values,
    reference_inputs, ReferenceInputMismatch,
};
pub use execution::{op_count, step_budget};
/// Typed bytes backing one declared IR buffer, as the evaluator holds them.
pub use oob::Buffer;
/// A tally of out-of-bounds accesses a diagnostic permissive run absorbed,
/// which surfaces the masking that hides GPU/CPU parity hazards. See
/// [`ReferenceRequest::execute_permissive`].
pub use oob::OobReport;
