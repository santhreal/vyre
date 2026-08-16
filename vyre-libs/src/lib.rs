//! # vyre-libs: every composition in the workspace
//!
//! Each public function returns a [`vyre_foundation::ir::Program`] built from
//! existing IR. No shader source, no backend, no dispatch. Consumer dialects
//! and compiler-internal domains are equal residents.
//!
//! ```ignore
//! use vyre_libs::math::dot;
//! let program = dot("x", "y", "result", 128)?;
//! ```
//!
//! Product dialects include `math`, `nn`, `scan`, `hash`, `decode`, `parsing`,
//! `security`, `visual`, `logical`, and `rule`. `hash` replaced `crypto`.
//! `scan` replaced `matching`. Compiler-internal domains (`device`,
//! `solvers`, `encoding`, `analysis`, `scheduling`, `reasoning`,
//! `graph-dispatch`, `telemetry`) are compositions too. They are feature-gated
//! and are not in `full` because they submit no `OperationRegistration`.
//!
//! The sole Category B exception is the `math::atomic` family, which needs
//! the backend `Expr::Atomic` emitter arm.
//!
//! A domain may move to a dedicated crate only through a clean public cutover
//! that migrates every caller and removes the old path.
//!
//! Every public composition wraps its body in a
//! [`vyre_foundation::ir::Node::Region`] with a stable generator name. The
//! optimizer treats regions as atomic until an explicit inline pass unrolls
//! them.
//!
//! Defaults enable a math / linear / matching / decode core. `crypto` and
//! `matching-regex` are opt-in. Turn defaults off with
//! `default-features = false` and enable the dialect you need.

// Semantic catalog entries are immutable values over static identifiers and
// function pointers, so the standard auto-traits provide Send + Sync without
// unsafe code.
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::double_must_use,
    clippy::items_after_test_module,
    clippy::assertions_on_constants,
    clippy::overly_complex_bool_expr,
    clippy::filter_map_bool_then
)]
// P3.3 nested-dialect reshape: each sub-dialect's single op file
// shares the sub-dialect's module name (e.g. `math/broadcast/broadcast.rs`).
// That's the intended shape for community packs that add second/
// third ops to the same sub-dialect later; the lint would fight
// the architectural decision.
#![allow(clippy::module_inception)]

/// Region builder  -  the shared helper every composition routes through.

/// Domain-neutral byte-range ordering predicates.
pub mod range_ordering;

/// `TensorRef`  -  typed buffer-argument wrapper used by every Cat-A
/// composition for dtype + shape + name-uniqueness validation.
pub(crate) mod tensor_ref;

pub use tensor_ref::{check_dtype, check_shape, check_unique_names, TensorRef, TensorRefError};

/// Shared builder helpers every Cat-A composition reuses.
pub(crate) mod builder;
mod builder_catalog;

pub use builder::{check_same_shape, checked_element_count};
pub use builder::{check_tensors, BuildOptions};

pub mod buffer_names;

/// `ProgramDescriptor`  -  introspection surface for Cat-A Programs.
pub(crate) mod descriptor;

/// Host-side byte marshalling for `ProgramDispatcher` calls.
pub mod dispatch_buffers;

/// Host-side capacity reservation for dispatch staging buffers. Crate-root
/// plumbing, not a dialect: every dialect that stages a dispatch reserves
/// through this one owner.
#[cfg(feature = "device")]
pub(crate) mod scratch;

/// Host-side compiled-`Program` cache keyed by dispatch shape. Crate-root
/// plumbing for the same reason as `scratch`.
#[cfg(feature = "device")]
pub(crate) mod dispatch_program_cache;

pub use descriptor::{BufferDescriptor, ProgramDescriptor};

/// Derived view over canonical library operation registrations.
pub mod operation_catalog;

/// Per-module call counters for the composition surface.
#[cfg(feature = "telemetry")]
pub mod telemetry;

/// Device-boundary contracts: probe, memory ownership, resident graph layout.
#[cfg(feature = "device")]
pub mod device;

/// Scheduling, fusion, batching, and dispatch-strategy compositions.
#[cfg(feature = "scheduling")]
pub mod scheduling;

/// Static-analysis, fixpoint, diagnostics, and verification compositions.
#[cfg(feature = "analysis")]
pub mod analysis;

/// Logic, causal reasoning, categorical rewrites, and knowledge compilation.
#[cfg(feature = "reasoning")]
pub mod reasoning;

/// Bitset, provenance, matroid and fingerprint encoding compositions.
#[cfg(feature = "encoding")]
pub mod encoding;

/// Numerical solver, autotuning, and spectral compositions.
#[cfg(feature = "solvers")]
pub mod solvers;

/// Math dialect  -  linear algebra, scans, broadcasting.
#[cfg(any(
    feature = "math-linalg",
    feature = "math-scan",
    feature = "math-broadcast",
    feature = "math-algebra",
    feature = "math-succinct"
))]
pub mod math;

/// Logical dialect  -  element-wise boolean composition.
#[cfg(feature = "logical")]
pub mod logical;

/// Neural-network dialect  -  activation, normalization, attention, linear.
#[cfg(any(
    feature = "nn-activation",
    feature = "nn-linear",
    feature = "nn-norm",
    feature = "nn-attention"
))]
pub mod nn;

/// Pattern-scanning dialect: neutral substring, DFA, NFA, and regex
/// program builders plus immutable compilation artifacts.
#[cfg(any(
    feature = "matching-substring",
    feature = "matching-dfa",
    feature = "matching-nfa"
))]
pub mod scan;

/// Decode / decompression compositions  -  base64, hex, DEFLATE (stored),
/// more coming. Pairs with `vyre-libs::matching::dfa` in the fused
/// decode→scan pipeline (Innovation I.1).
#[cfg(feature = "decode")]
pub mod decode;

/// Hash / checksum dialect  -  FNV-1a-32, FNV-1a-64, CRC-32, Adler-32,
/// BLAKE3 compression. Consolidated from the former `vyre-libs::crypto`
/// module per Migration 3. Every op lives here as a pure Cat-A
/// composition over existing IR primitives (no dedicated target builder emitter
/// arm required, per the intrinsic-vs-library rule).
#[cfg(feature = "hash")]
pub mod hash;

/// Text-processing compositions for the GPU C parser pipeline
/// (Phase L1+): byte classification, UTF-8 validation, line index.
pub mod text;

/// Representation sub-dialect: bit-packing and unpacking.
pub mod representation;

/// GPU parser infrastructure (Phase L3+): bracket matching, DFA
/// lexer driver, LR(1) table walker. Grammar tables are generated
/// host-side by `downstream analyzer-grammar-gen` and loaded as ReadOnly buffers.
pub mod parsing;

/// Packed AST walks (`ast_walk_*` catalog ops).
pub mod graph;

/// Security / taint compositions for static program analysis.
/// Every op registers via `inventory::submit!` and lives under a
/// stable op id. The implementations compose graph and dataflow
/// primitives so downstream analyzers lower to one production GPU-facing
/// surface.
#[cfg(feature = "security")]
pub mod security;

/// GPU-accelerated visual effects  -  blur, shadow, filter chain,
/// gradient, compositing, and glass material. Tier 3 compositions
/// over `math::conv1d` (Tier 2.5) and bare IR expressions. The
/// Molten web engine's visual effect substrate.
#[cfg(feature = "visual")]
pub mod visual;

#[cfg(any(
    feature = "math-linalg",
    feature = "math-scan",
    feature = "math-broadcast"
))]
pub(crate) use builder::elementwise::{f32_elementwise_mul, F32MulRhs};
#[cfg(feature = "nn-linear-4bit")]
pub(crate) use math::linalg::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, MatrixShape,
};

// vyre-libs::hardware removed (audit 2026-04-21 BLOCKER-1/6).
// An intrinsic needs its own emitter arm and its own reference-interpreter
// arm, so every one of them lives in `vyre-primitives::hardware`. The atomic,
// clamp, lzcnt and tzcnt compositions live in `vyre-libs::math::*` and reach
// `Expr::Atomic`, `Expr::min`, `Expr::max` and `Expr::popcount` directly.
//
// vyre-libs::crypto removed (audit 2026-04-21 BLOCKER-3). Deprecated
// shim deleted in favor of the canonical path at `vyre-libs::hash`.
//
// vyre-libs::composite removed (audit 2026-04-21 BLOCKER-3). The three
// hash ops that lived there (adler32, crc32, fnv1a64) are canonical at
// `vyre-libs::hash::*`.

/// Rule-engine dialect  -  typed conditions, formulas, and program builder used
/// by detection rule compilers.
#[cfg(feature = "rule")]
pub mod rule;

/// Vector-widened string interning. CHD perfect hash
/// over Tier-B label families  -  60k+ function-name strings reduce
/// to one subgroup-shuffle + one DRAM load on the GPU.
#[cfg(feature = "intern")]
pub mod intern;

/// Operation contract presets used by catalog entries.
pub mod contracts;
/// Type-signature constants shared across op definitions.
pub(crate) mod signatures;
/// Re-exports every type-signature constant at the crate root for convenient access.
pub use signatures::{
    BOOL_OUTPUTS, BYTES_TO_BYTES_INPUTS, BYTES_TO_BYTES_OUTPUTS, BYTES_TO_U32_OUTPUTS,
    F32_F32_F32_INPUTS, F32_F32_INPUTS, F32_INPUTS, F32_OUTPUTS, I32_OUTPUTS, U32_INPUTS,
    U32_OUTPUTS, U32_U32_INPUTS,
};
/// Owner-local byte fixtures for semantic operation registrations and tests.
pub(crate) mod fixture_bytes;
/// Pre-sweep shader snapshot migration entries, collected via inventory.
/// `pub(crate)` because the registry is an internal pre-sweep tool  -
/// downstream dialects do not submit through this path.
pub(crate) mod test_migration;

/// Program composition helpers for parity suites, in-tree and downstream.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_parity_oracles;
