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

/// The declared seam one Tier 3 dialect crosses to compose another.
pub mod prelude;

/// Shared builder helpers every Cat-A composition reuses.
pub(crate) mod builder;

/// Shared plumbing every composition needs and no dialect owns: what a buffer
/// argument is, what a built `Program` declares, what a registration carries,
/// and what the host does to launch it.
pub(crate) mod plumbing;

#[cfg(feature = "graph")]
pub use builder::csr;
pub use builder::elementwise;
pub use builder::range_ordering;
pub use builder::state_machine;
pub use builder::stencil;
#[cfg(feature = "graph")]
pub use builder::CsrTraversalComposer;
pub use builder::ElementwiseComposer;
pub use builder::TableStateMachineComposer;
pub use builder::{check_same_shape, checked_element_count};
pub use builder::{check_tensors, BuildOptions};
pub use builder::{Grid2DComposer, Grid2DShape};
pub use plumbing::host::dispatch_buffers;
pub use plumbing::operand::buffer_names;
pub use plumbing::operand::tensor_ref::{
    check_dtype, check_shape, check_unique_names, TensorRef, TensorRefError,
};
pub use plumbing::program::descriptor::{BufferDescriptor, ProgramDescriptor};
pub use plumbing::registration::{contracts, operation_catalog};

/// Per-module call counters for the composition surface.
#[cfg(feature = "telemetry")]
pub use plumbing::host::telemetry;

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

/// Math dialect: linear algebra, scans, broadcasting, plus the reusable math
/// kernels. `math-dialect` gates the dialect surface; `math-kernels` gates the
/// kernels. `graph`, `geom` and `opt` reach `math::fixed` and
/// `math::fixed_u32_matmul` without either.
#[cfg(any(
    feature = "math-dialect",
    feature = "math-kernels",
    feature = "graph",
    feature = "geom",
    feature = "opt"
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

/// Language-model decode layer  -  paged key-value cache addressing and token
/// sampling, composed from the neural-net and math dialects.
#[cfg(feature = "llm")]
pub mod llm;

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
#[cfg(feature = "text")]
pub mod text;

/// Representation sub-dialect: bit-packing and unpacking.
#[cfg(feature = "representation")]
pub mod representation;

/// GPU parser infrastructure (Phase L3+): bracket matching, DFA
/// lexer driver, LR(1) table walker. Grammar tables are generated
/// host-side by `downstream analyzer-grammar-gen` and loaded as ReadOnly buffers.
// `parsing-kernels` and `go-parser` are the two roots. `parsing` names both
// language pipelines and `python-parser` names `parsing-kernels`, so a build
// that asks for either of those already sets one of the two.
#[cfg(any(
    feature = "parsing",
    feature = "parsing-kernels",
    feature = "go-parser",
    feature = "python-parser"
))]
pub mod parsing;

/// Packed AST walks (`ast_walk_*` catalog ops).
#[cfg(feature = "graph")]
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

/// Bitset kernels: `and`/`or`/`not`/`xor`/`popcount`/`any`/`contains` over
/// packed u32 bitsets. The NodeSet and ValueSet representation every graph
/// kernel consumes.
#[cfg(feature = "bitset")]
pub mod bitset;

/// Reduction kernels: `count`/`min`/`max`/`sum` over bitsets and fixed-width
/// ValueSets. Backs source-query dialect aggregates.
#[cfg(feature = "reduce")]
pub mod reduce;

/// Virtual filesystem DMA compositions: the `#include` hash resolver that
/// turns asset identifiers into asynchronous block loads. Built from
/// `Node::AsyncLoad` and `Node::AsyncWait`, so it composes existing IR and
/// carries no hardware contract of its own.
#[cfg(feature = "vfs")]
pub mod vfs;

/// Label to NodeSet resolver: turn a TagFamily bitmask into a NodeSet bitset.
#[cfg(feature = "label")]
pub mod label;

/// Frozen predicate kernels: the engine predicates (call_to, return_value_of,
/// arg_of, size_argument_of, edge, in_function, in_file, in_package,
/// literal_of, node_kind) that source-query stdlib rules compose into every
/// higher-level query.
#[cfg(feature = "predicate")]
pub mod predicate;

/// Deterministic fixpoint kernel: ping-pong with a convergence flag. Composes
/// `csr_forward_traverse` and bitset OR into the transitive-closure driver every
/// stdlib taint rule needs.
#[cfg(feature = "fixpoint")]
pub mod fixpoint;

/// Geometric and Clifford-algebra kernels. Multivector products for equivariant
/// networks, physics simulation, robotics, 3D vision.
#[cfg(feature = "geom")]
pub mod geom;

/// Optimization kernels: homotopy continuation, sum-of-squares, matroid
/// intersection.
#[cfg(feature = "opt")]
pub mod opt;

/// Topological-data-analysis kernels: Vietoris-Rips filtration and simplicial
/// complex operations.
#[cfg(feature = "topology")]
pub mod topology;

/// Pattern-matching kernels. Distinct from `scan`, which is the neutral
/// program-builder dialect over them.
#[cfg(feature = "matching-kernels")]
pub mod matching;

/// NFA kernels: subgroup-cooperative simulator.
#[cfg(feature = "nfa")]
pub mod nfa;

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

/// The type-signature constants an op declaration reads. The module holding
/// them is private, so these re-exports are their only public path.
pub use plumbing::registration::signatures::{
    BOOL_OUTPUTS, BYTES_TO_BYTES_INPUTS, BYTES_TO_BYTES_OUTPUTS, BYTES_TO_U32_OUTPUTS,
    F32_F32_F32_INPUTS, F32_F32_INPUTS, F32_INPUTS, F32_OUTPUTS, I32_OUTPUTS, U32_INPUTS,
    U32_OUTPUTS, U32_U32_INPUTS,
};
/// Owner-local byte fixtures for semantic operation registrations and tests.
pub(crate) mod fixture_bytes;

/// Dispatcher doubles and program sequencing for this crate's own unit tests.
#[cfg(test)]
mod test_parity_oracles;
