//! # vyre-libs  -  Category A composition ecosystem
//!
//! `vyre-libs` is the library layer that sits ON TOP of `vyre-ops`.
//!
//! Almost every function is a **pure Category A composition**: it returns a
//! [`vyre_foundation::ir::Program`] built entirely from existing vyre IR primitives. The
//! sole exception is the `math::atomic` family, which are **Category B**
//! (`Category::Intrinsic`) because they require the backend to own the
//! `Expr::Atomic` target builder emitter arm (F-IR-35).
//!
//! This is the ML/DSP/cryptographic ecosystem layer. Examples:
//!
//! ```ignore
//! use vyre_libs::nn::linear;
//! let program = linear(/* input_buf */ "x", /* weights */ "w", /* bias */ "b");
//! // `program` is a standard vyre_foundation::ir::Program you dispatch against any backend.
//! ```
//!
//! ## Why a single `vyre-libs` crate, not five?
//!
//! The initial proposal suggested `vyre-nn`, `vyre-math`, `vyre-match`,
//! `vyre-crypto`, `vyre-graph-stitch` as five standalone crates. That
//! is the right endpoint  -  each becomes its own crates.io identity
//! with its own community  -  but the migration cost at 0.6 is wrong.
//! This crate starts as one, with public modules for each domain; when
//! a module has its own consumer base + maturity, it promotes to a
//! dedicated crate without breaking downstream code (the
//! `vyre-libs::nn` path moves to `vyre-nn::` via a re-export shim).
//!
//! `vyre-graph-stitch` was deliberately omitted  -  "logical linker for
//! emitted graphs" is a `vyre-foundation` concern (IR composition),
//! not a library crate.
//!
//! ## Region wrapping
//!
//! Every public composition wraps its body in a
//! [`vyre_foundation::ir::Node::Region`] with a stable generator name. The
//! optimizer treats Regions as atomic by default (preserves
//! debuggability + source-mapping); explicit inline passes can unroll
//! them. This is LLVM's function-vs-always-inline split at IR level.
//!
//! ## Feature flags
//!
//! Each domain lives behind a feature flag so minimal consumers pay
//! for only what they use:
//!
//! - `math` (default)  -  linear algebra, scans, broadcasts
//! - `nn` (default, implies `math`)  -  neural-net primitives
//! - `matching` (default)  -  regex, DFA, substring, multi-pattern
//! - `crypto` (default)  -  hashing, MAC, checksums
//!
//! Turn defaults off with `default-features = false` and cherry-pick
//! what you need.

// P1.11 (closed): `OpEntry` is now POD over `&'static str` + `fn(...)`,
// so stdlib auto-traits give us `Send + Sync` for free. No `unsafe`
// anywhere in vyre-libs  -  `forbid` catches any future regression.
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

/// Build a trap-only program for registry fixtures or infallible composition wrappers.
#[allow(dead_code)]
pub(crate) fn invalid_program(
    op_id: &'static str,
    message: impl Into<String>,
) -> vyre_foundation::ir::Program {
    let message = message.into();
    vyre_foundation::ir::Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![region::wrap_anonymous(
            op_id,
            vec![vyre_foundation::ir::Node::trap(vyre_foundation::ir::Expr::u32(0), message)],
        )],
    )
}

/// Region builder  -  the shared helper every composition routes through.
pub mod region;

/// Domain-neutral byte-range ordering predicates. Previously lived inside
/// `vyre-libs::security::topology`; hoisted out so non-security callers
/// (a downstream analyzer's `Before`/`After` predicates, future dialects) do not pull the
/// security dialect through the import graph. See CRITIQUE_VISION_ALIGNMENT_2026-04-23 V5.
pub mod range_ordering;

/// `TensorRef`  -  typed buffer-argument wrapper used by every Cat-A
/// composition for dtype + shape + name-uniqueness validation.
pub mod tensor_ref;

pub use tensor_ref::{check_dtype, check_shape, check_unique_names, TensorRef, TensorRefError};

/// Shared builder helpers every Cat-A composition reuses.
pub mod builder;
mod substrate_catalog;

pub use builder::{check_tensors, BuildOptions};

pub mod buffer_names;

/// `ProgramDescriptor`  -  introspection surface for Cat-A Programs.
pub mod descriptor;

pub use descriptor::{BufferDescriptor, ProgramDescriptor};


#[cfg(feature = "math-linalg")]
pub use math::{matmul_bias_tiled, matmul_tiled, MatmulBias, MatmulBiasTiled, MatmulTiled};

/// Neutral program fixtures consumed by upper conformance harnesses.
#[doc(hidden)]
pub mod fixture_catalog;

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

/// GPU-native compiler middle-end (CFG and ELF emission helpers) for the C pipeline.
#[cfg(feature = "c-parser")]
pub mod compiler;

#[cfg(feature = "c-parser")]
pub use compiler::{
    cfg::c11_build_cfg_and_gotos, object_writer::opt_lower_elf,
    regalloc::opt_x86_64_register_allocation, stack_layout::opt_stack_layout_generation,
    types_layout::c11_compute_alignments,
};

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

/// Compatibility facade for GPU dataflow compositions.
/// This path remains for older `vyre-libs::dataflow::*` consumers and must
/// not grow a parallel dataflow implementation tree.
pub mod dataflow;

#[cfg(feature = "c-parser")]
pub(crate) use compiler::atomic_collect::atomic_collect_u32;
pub use dataflow::{
    validate_dynamic_pipeline, DynamicPrimitiveSoundness, DynamicSoundnessViolation,
    PrecisionContract, SharedFactHeader, SharedFactKind, Soundness, SoundnessTagged,
};
#[cfg(any(
    feature = "math-linalg",
    feature = "math-scan",
    feature = "math-broadcast"
))]
pub(crate) use math::elementwise::{f32_elementwise_mul, F32MulRhs};
#[cfg(feature = "nn-linear-4bit")]
pub(crate) use math::linalg::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, MatrixShape,
};

// vyre-libs::hardware removed (audit 2026-04-21 BLOCKER-1/6).
// Canonical Cat-C intrinsics live exclusively in the `vyre-intrinsics`
// crate; library compositions of atomic / clamp / lzcnt / tzcnt ops
// live in `vyre-libs::math::*` (which uses `Expr::Atomic`, `Expr::min`,
// `Expr::max`, `Expr::popcount` directly per library-tiers.md).
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
pub mod signatures;
/// Re-exports every type-signature constant at the crate root for convenient access.
pub use signatures::{
    BOOL_OUTPUTS, BYTES_TO_BYTES_INPUTS, BYTES_TO_BYTES_OUTPUTS, BYTES_TO_U32_OUTPUTS,
    F32_F32_F32_INPUTS, F32_F32_INPUTS, F32_INPUTS, F32_OUTPUTS, I32_OUTPUTS, U32_INPUTS,
    U32_OUTPUTS, U32_U32_INPUTS,
};
/// Pre-sweep shader snapshot migration entries, collected via inventory.
/// `pub(crate)` because the registry is an internal pre-sweep tool  -
/// downstream dialects do not submit through this path.
pub(crate) mod test_migration;
/// Test support components for vyre-libs.
pub mod test_support;


/// Re-export the small set of vyre types every composition function
/// returns. Consumers can `use vyre_libs::prelude::*` and get the API
/// plus the types it returns.
pub mod prelude {
    pub use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
    pub use vyre_foundation::ir::model::expr::GeneratorRef;

    // P2.1 / P2.2: the typed-tensor API + shared builder primitives.
    // Every Cat-A op ships with a TensorRef-accepting builder; the
    // prelude exposes the full construction surface so `use
    // vyre_libs::prelude::*;` is enough to author a new Cat-A op.
    pub use crate::builder::{check_tensors, BuildOptions};
    pub use crate::tensor_ref::{
        check_dtype, check_shape, check_unique_names, TensorRef, TensorRefError,
    };

    // Region wrapper  -  every composition emits its body through this.
    pub use crate::region::{wrap, wrap_anonymous, wrap_child};

    // Built-in Cat-A builders (gated on the relevant feature flags so
    // minimum-footprint consumers don't pay for the ones they skip).
    #[cfg(feature = "decode")]
    pub use crate::decode::{base64_decode, hex_decode, inflate, ziftsieve_gpu};
    #[cfg(feature = "crypto-blake3")]
    pub use crate::hash::blake3_compress;
    #[cfg(feature = "crypto-fnv")]
    pub use crate::hash::fnv1a32;
    #[cfg(feature = "logical")]
    pub use crate::logical::{and, nand, nor, or, xor};
    #[cfg(feature = "math-broadcast")]
    pub use crate::math::broadcast;
    #[cfg(feature = "math-scan")]
    pub use crate::math::scan_prefix_sum;
    #[cfg(feature = "math-algebra")]
    pub use crate::math::{
        bool_semiring_matmul, lattice_join, lattice_meet, semiring_min_plus_mul, sketch_mix,
        try_bool_semiring_matmul, try_lattice_join, try_lattice_meet, try_semiring_min_plus_mul,
        try_sketch_mix,
    };
    #[cfg(feature = "math-linalg")]
    pub use crate::math::{dot, matmul, matmul_tiled, Matmul, MatmulTiled};
    #[cfg(feature = "math-succinct")]
    pub use crate::math::{rank1_query, rank1_superblocks, try_rank1_query, try_rank1_superblocks};
    #[cfg(feature = "nn-linear")]
    pub use crate::nn::linear;
    #[cfg(feature = "nn-activation")]
    pub use crate::nn::relu;
    #[cfg(feature = "nn-attention")]
    pub use crate::nn::{attention, softmax, Attention, Softmax};
    #[cfg(feature = "nn-norm")]
    pub use crate::nn::{layer_norm, LayerNorm};
    #[cfg(feature = "matching-substring")]
    pub use crate::scan::substring_search;
    #[cfg(feature = "matching-dfa")]
    pub use crate::scan::{aho_corasick, dfa_compile, CompiledDfa, DfaCompileError};
}

