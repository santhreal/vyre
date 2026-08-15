// Crate policy: unsafe is DENIED by default (was `forbid`), so the crate stays
// unsafe-free everywhere except call sites that carry an explicit
// `#[allow(unsafe_code)]` plus a `// SAFETY:` proof. The sole current exception
// is `wire::fill_le_words_into`, where eliminating a redundant pre-copy
// zero-fill on the GPU-readback decode hot path is worth a single, audited
// uninitialized-write. `deny` (not `forbid`) is required so that one annotated
// exception can compile; every other `unsafe` in the crate still hard-errors.
#![deny(unsafe_code)]
//! `vyre-primitives`: the operations that cannot be composed.
//!
//! An operation is admitted here only when it cannot be expressed as a
//! composition, which means it requires its own arm in a backend emitter and
//! its own arm in the reference interpreter. That is the whole rule. How many
//! callers an operation has is not an admission criterion: something reused by
//! twenty dialects and expressible as `fn(..) -> Program` over existing IR is a
//! composition, and a composition belongs in `vyre-libs` no matter who calls
//! it.
//!
//! Two kinds of item satisfy that rule.
//!
//! 1. **Marker types** (`markers`, always on, no dependencies). The unit
//!    structs a backend emitter and the reference interpreter dispatch on. They
//!    are the names those two arms agree about, so they cannot live in a crate
//!    either side composes over.
//!
//! 2. **Hardware intrinsics** (`hardware`). Subgroup collectives, memory
//!    fences, bit instructions, fused multiply-add, inverse square root: each
//!    one is a target instruction with no IR spelling.
//!
//! The remaining domains are the substrate those two rest on: the wire format,
//! the safe-IR guards, and the per-domain kernels a backend must see as one
//! unit. Each domain is one directory and one feature, and a consumer enables
//! only the domains it needs.
//!
//! ```text
//! vyre-primitives/
//!   src/
//!     lib.rs              # this table
//!     markers.rs          # marker types, always on
//!     wire.rs             # host/device byte layout, always on
//!     ir_safe.rs          # guarded IR construction
//!     hardware/           # feature = "hardware"
//!     text/               # feature = "text"
//!     matching/           # feature = "matching"
//!     decode/             # feature = "decode"
//!     nfa/                # feature = "nfa"
//!     hash/               # feature = "hash"
//!     math/               # feature = "math"
//!     parsing/            # feature = "parsing"
//!     nn/                 # feature = "nn"
//!     graph/              # feature = "graph"
//!     geom/               # feature = "geom"
//!     opt/                # feature = "opt"
//!     topology/           # feature = "topology"
//!     visual/             # feature = "visual"
//!     bitset/             # feature = "bitset"
//!     reduce/             # feature = "reduce"
//!     label/              # feature = "label"
//!     predicate/          # feature = "predicate"
//!     fixpoint/           # feature = "fixpoint"
//!     vfs/                # feature = "vyre-foundation"
//! ```
//!
//! The path is the interface. A domain `mod.rs` exposes its sub-modules rather
//! than a flat namespace, so a call site reads
//! `vyre_primitives::text::char_class::char_class(..)` and names the operation
//! it reached.

mod dispatch_grid;
#[cfg(feature = "vyre-foundation")]
pub mod ir_safe;
mod markers;
mod operand_shape;
pub mod wire;

/// One classification of every Cargo feature this crate declares.
///
/// A domain feature that is in neither list is how a third admission category
/// returns, so the module's own tests hold the lists to the manifest. The lists
/// are crate-private: they describe which compositions are still parked here,
/// and nothing outside may depend on that.
mod organization;

pub use markers::{
    ArithAdd, ArithMul, BitwiseAnd, BitwiseOr, BitwiseXor, Clz, CombineOp, CompareEq, CompareLt,
    Gather, HashBlake3, HashFnv1a, PatternMatchDfa, PatternMatchLiteral, Popcount, Reduce,
    RegionId, Scan, Scatter, ShiftLeft, ShiftRight, Shuffle,
};
#[cfg(any(feature = "graph", feature = "math", feature = "geom", feature = "opt"))]
use vyre_foundation::ir::Expr;
#[cfg(feature = "vyre-foundation")]
use vyre_foundation::ir::Program;

#[cfg(feature = "vyre-foundation")]
pub(crate) fn demote_intermediate_outputs(program: Program, final_output: &str) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .map(|buffer| {
            let mut buffer = buffer.clone();
            if buffer.name() != final_output && buffer.is_output() {
                buffer.is_output = false;
                buffer.pipeline_live_out = true;
            }
            buffer
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}

/// Return `(left * right) >> 16` for unsigned 16.16 fixed-point lanes without
/// losing the high half of the product to 32-bit overflow.
#[cfg(any(feature = "graph", feature = "math", feature = "geom", feature = "opt"))]
pub(crate) fn fixed_mul_16_16_expr(left: Expr, right: Expr) -> Expr {
    // 16.16 fixed-point is a SIGNED number format: operands are two's-complement i32 in a u32, so a
    // negative value is stored wrapped (`-v` → `2^32 - |v|·2^16`). Extracting the 16.16 product as
    // `(low >> 16) | (high << 16)` requires the SIGNED 64-bit high word. `Expr::mulhi` is UNSIGNED, so
    // reconstruct the signed high word with the standard correction:
    //   signed_high = unsigned_high − (left < 0 ? right : 0) − (right < 0 ? left : 0)
    // (A wrong all-unsigned `mulhi` treats a negative operand as ~2^32 and produces a garbage giant
    // product, the exact silent-corruption bug that made the fixed-point AMG V-cycle diverge from its
    // f64 reference the moment a residual `b − A·x` went negative. See BACKLOG
    // `LIMITATION-amg-fixed-path-unsigned-mul-negatives`.) For NON-NEGATIVE operands (|v| < 2^31, i.e.
    // every legitimate 16.16 magnitude) both corrections are zero, so this is bit-identical to the old
    // unsigned form (a strict correctness superset that leaves the non-negative kernels unchanged).
    let low = Expr::mul(left.clone(), right.clone());
    let unsigned_high = Expr::mulhi(left.clone(), right.clone());
    // `0 - (x >> 31)` is an all-ones mask when `x`'s sign bit is set, else zero (logical u32 shift).
    let left_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(left.clone(), Expr::u32(31)));
    let right_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(right.clone(), Expr::u32(31)));
    let correction_left = Expr::bitand(left_sign_mask, right);
    let correction_right = Expr::bitand(right_sign_mask, left);
    let signed_high = Expr::sub(Expr::sub(unsigned_high, correction_left), correction_right);
    Expr::bitor(
        Expr::shr(low, Expr::u32(16)),
        Expr::shl(signed_high, Expr::u32(16)),
    )
}

/// SIGNED integer division of a two's-complement `numerator` by a KNOWN-POSITIVE `denominator`
/// (truncating toward zero), for use in fixed-point kernels whose numerator may be negative.
///
/// `Expr::div` is UNSIGNED, so dividing a wrapped-negative 16.16 numerator (e.g. a Jacobi residual
/// `b − A·x` that went negative) by a small positive integer yields garbage, the second half of the
/// silent-corruption bug behind `LIMITATION-amg-fixed-path-unsigned-mul-negatives` (the first half being
/// [`fixed_mul_16_16_expr`]). This computes `sign·(|numerator| / denominator)` via the branchless
/// mask-abs idiom: `mask = numerator >> 31` broadcast to all-ones on a negative value, `abs = (n ^ m) − m`,
/// `q = abs / d` (now a genuine unsigned divide of a non-negative magnitude), then reapply the sign
/// `(q ^ m) − m`. For a NON-NEGATIVE numerator `mask == 0`, so this reduces to plain `Expr::div`, a
/// strict correctness superset that leaves non-negative kernels unchanged. The denominator MUST be
/// positive (all callers pass `diag_units ≥ 1`); a negative denominator is not handled.
#[cfg(any(feature = "graph", feature = "math", feature = "geom", feature = "opt"))]
pub(crate) fn fixed_sdiv_by_positive_expr(numerator: Expr, denominator: Expr) -> Expr {
    // `numerator >> 31` is 0 or 1 (logical u32 shift); `0 - that` broadcasts to the all-ones sign mask.
    let sign_mask = Expr::sub(Expr::u32(0), Expr::shr(numerator.clone(), Expr::u32(31)));
    // abs(numerator) = (numerator ^ sign_mask) - sign_mask (two's-complement branchless absolute value).
    let magnitude = Expr::sub(
        Expr::bitxor(numerator, sign_mask.clone()),
        sign_mask.clone(),
    );
    let quotient = Expr::div(magnitude, denominator);
    // Reapply the original sign: (quotient ^ sign_mask) - sign_mask.
    Expr::sub(Expr::bitxor(quotient, sign_mask.clone()), sign_mask)
}

#[cfg(any(feature = "graph", feature = "math"))]
pub(crate) mod fixed_u32_matmul;

#[cfg(all(
    any(feature = "graph", feature = "math"),
    any(test, feature = "cpu-parity")
))]
pub(crate) mod chebyshev_recurrence;

#[cfg(any(feature = "label", feature = "predicate"))]
pub(crate) mod nodeset_filter;

/// Derived view over canonical primitive operation registrations.
#[cfg(feature = "inventory-registry")]
pub mod operation_catalog;

/// Text primitives.
#[cfg(feature = "text")]
pub mod text;

/// Pattern-matching primitives.
#[cfg(feature = "matching")]
pub mod matching;

/// Decode primitives.
#[cfg(feature = "decode")]
pub mod decode;

/// NFA primitives  -  subgroup-cooperative simulator (G1 GPU perf).
#[cfg(feature = "nfa")]
pub mod nfa;

/// Hash primitives (FNV-1a 32/64, CRC-32).
#[cfg(feature = "hash")]
pub mod hash;

/// Math primitives (dot, scan, reduce, broadcast).
#[cfg(feature = "math")]
pub mod math;

/// Parsing primitives (optimizer and AST scan kernels).
#[cfg(feature = "parsing")]
pub mod parsing;

/// Neural-network primitives (attention and normalization sub-kernels).
#[cfg(feature = "nn")]
pub mod nn;

/// Graph primitives (topological sort, reachability, CSR traversal,
/// SCC decomposition, path reconstruction  -  the Tier 2.5 substrate
/// that a external analyzer's stdlib rules compose against).
#[cfg(feature = "graph")]
pub mod graph;

/// Geometric / Clifford-algebra primitives (#8). Multivector products
/// for equivariant NNs, physics simulation, robotics, 3D vision.
#[cfg(feature = "geom")]
pub mod geom;

/// Optimization primitives (#9, #14, #46). Homotopy continuation,
/// SOS, matroid intersection. Self: vyre's megakernel scheduler.
#[cfg(feature = "opt")]
pub mod opt;

/// Topological-data-analysis primitives (#15, #32). Vietoris-Rips
/// filtration + simplicial complex operations. User: TDA, persistent
/// landscape features, call-graph topological signatures.
#[cfg(feature = "topology")]
pub mod topology;

/// Visual pixel-map primitives. Shared packed-RGBA invocation skeletons
/// reused by higher-level image-processing compositions.
#[cfg(feature = "visual")]
pub mod visual;

/// Category C hardware intrinsics. Ops that need a dedicated backend emitter
/// arm and a dedicated reference-interpreter arm: subgroup collectives,
/// memory fences, bit instructions, fused multiply-add, inverse square root.
#[cfg(feature = "hardware")]
pub mod hardware;

/// Bitset primitives  -  `and`/`or`/`not`/`xor`/`popcount`/`any`/
/// `contains` over packed u32 bitsets. The NodeSet / ValueSet
/// representation every graph primitive consumes.
#[cfg(feature = "bitset")]
pub mod bitset;

/// Reduction primitives  -  `count`/`min`/`max`/`sum` over bitsets and
/// fixed-width ValueSets. Backs source-query dialect aggregates.
#[cfg(feature = "reduce")]
pub mod reduce;

/// Label → NodeSet resolver  -  turn a TagFamily bitmask into a
/// NodeSet bitset. Implements the `@family` lookup that a external analyzer's
/// label surface surfaces.
#[cfg(feature = "label")]
pub mod label;

/// Frozen predicate primitives  -  the ~10 engine primitives (call_to,
/// return_value_of, arg_of, size_argument_of, edge, in_function,
/// in_file, in_package, literal_of, node_kind) that source-query dialect stdlib
/// rules compose into every higher-level query.
#[cfg(feature = "predicate")]
pub mod predicate;

/// Deterministic fixpoint primitive (ping-pong with convergence
/// flag). Composes `csr_forward_traverse` + bitset OR into the
/// transitive-closure driver every stdlib taint rule needs.
#[cfg(feature = "fixpoint")]
pub mod fixpoint;

/// Virtual File System DMA primitives. Uses `vyre_foundation::ir`
/// so it's gated behind the same set of features that pull
/// vyre-foundation in as an optional dep. Any of the domain
/// features enables vfs.
#[cfg(any(
    feature = "text",
    feature = "matching",
    feature = "decode",
    feature = "math",
    feature = "nn",
    feature = "hash",
    feature = "parsing",
    feature = "graph",
    feature = "bitset",
    feature = "reduce",
    feature = "label",
    feature = "predicate",
    feature = "fixpoint",
))]
pub mod vfs;

/// Wire-format envelope re-exported from vyre-foundation.
///
/// Every primitive that ships its own `to_bytes` / `from_bytes` (today:
/// `CompiledDfa`; future: serializable region tables, hash tables,
/// parser plans) composes this envelope. Re-exporting at the
/// vyre-primitives root keeps the import path uniform for consumers:
/// `vyre_primitives::serial_data::WireWriter` regardless of whether
/// the type lives at the primitive layer or higher up.
///
/// Available when any feature that pulls vyre-foundation is enabled
/// (every primitive domain enables it).
#[cfg(feature = "vyre-foundation")]
pub mod serial_data {
    pub use vyre_foundation::serial::wire_round_trip;
    pub use vyre_foundation::serial::{EnvelopeError, WireReader, WireWriter};
}

/// Curated prelude - the byte-pack/decode primitives every consumer
/// needs for GPU buffer construction and readback, plus the shared
/// envelope types when vyre-foundation is in play.
///
/// `use vyre_primitives::prelude::*;` should be the only import a
/// caller needs for the common pack/unpack surface. Adding new wire
/// primitives must keep this list in sync.
pub mod prelude {
    pub use crate::wire::{
        append_f32_slice_le_bytes, append_packed_byte_lane, append_u32_slice_le_bytes,
        decode_f32_le_bytes_all, decode_i32_le_bytes_all, decode_u16_le_bytes_all,
        decode_u32_le_bytes_all, decode_u64_le_bytes_all, pack_bytes_as_u32_slice,
        pack_bytes_as_u32_slice_min_words, pack_f32_slice, pack_f32_slice_into,
        pack_f32_slice_into_uninit, pack_i32_slice, pack_i32_slice_into, pack_u16_slice,
        pack_u16_slice_into, pack_u32_slice, pack_u32_slice_into, pack_u32_slice_into_uninit,
        pack_u32_slice_min_words_into, pack_u64_slice, pack_u64_slice_into, read_f32_le_word,
        read_u32_le_word, unpack_f32_slice, unpack_f32_slice_into, unpack_u32_slice_into,
    };
}
