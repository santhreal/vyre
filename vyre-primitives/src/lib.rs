// Crate policy: unsafe is DENIED by default (was `forbid`), so the crate stays
// unsafe-free everywhere except call sites that carry an explicit
// `#[allow(unsafe_code)]` plus a `// SAFETY:` proof. The sole current exception
// is `wire::fill_le_words_into`, where eliminating a redundant pre-copy
// zero-fill on the GPU-readback decode hot path is worth a single, audited
// uninitialized-write. `deny` (not `forbid`) is required so that one annotated
// exception can compile; every other `unsafe` in the crate still hard-errors.
#![deny(unsafe_code)]
//! `vyre-primitives`: marker types and uncomposable hardware
//! intrinsics.
//!
//! Two things belong here.
//!
//! 1. **Marker types** (`markers`, always on): unit structs the
//!    reference interpreter and backend emitters dispatch on.
//! 2. **Category C hardware** (`hardware`): ops that need a dedicated
//!    emitter arm and a dedicated reference-interpreter arm.
//!
//! Everything else in this crate is a composition that belongs in
//! `vyre-libs`. Reuse count is not an admission criterion. Those
//! domains keep their `vyre_primitives::<domain>` paths until they
//! move. [`organization`] is the one list of what is intrinsic, what
//! is parked, and what is crate support.
//!
//! The path is the interface. Callers write
//! `vyre_primitives::hardware::fma_f32` and, until the move,
//! `vyre_primitives::math::…`.
//!
//! Admission for a hardware intrinsic: nothing that composes over existing IR
//! is admitted. The complexity rule is enforced by gate1 in `xtask-registry`,
//! which walks each operation's registered exemplar and passes it on either of
//! two grounds: at most 4 loops and at most 200 nodes, or at least 60 percent
//! of its nodes living inside a `Region` whose `source_region` names another
//! registered operation. An anonymous `Region` is a local wrapper and does not
//! count as composition.

mod dispatch_grid;
#[cfg(feature = "vyre-foundation")]
pub mod ir_safe;
mod markers;
/// Feature classification: intrinsic, parked composition, or support.
pub mod organization;
pub mod wire;
#[cfg(feature = "vyre-foundation")]
use std::sync::Arc;

pub use markers::{
    ArithAdd, ArithMul, BitwiseAnd, BitwiseOr, BitwiseXor, Clz, CombineOp, CompareEq, CompareLt,
    Gather, HashBlake3, HashFnv1a, PatternMatchDfa, PatternMatchLiteral, Popcount, Reduce,
    RegionId, Scan, Scatter, ShiftLeft, ShiftRight, Shuffle,
};
#[cfg(feature = "vyre-foundation")]
use vyre_foundation::ir::model::expr::Ident;
#[cfg(feature = "vyre-foundation")]
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Build a scalar trap program for invalid primitive builder inputs.
///
/// Primitive constructors are intentionally infallible for composition with
/// registry fixtures and generated dialect code. Invalid user-controlled
/// shapes must therefore become explicit IR traps, not host panics.
#[cfg(feature = "vyre-foundation")]
pub(crate) fn invalid_output_program(
    op_id: &'static str,
    output: &str,
    data_type: DataType,
    message: String,
) -> Program {
    Program::wrapped(
        vec![BufferDecl::output(output, 0, data_type).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(vec![Node::trap(Expr::u32(0), message)]),
        }],
    )
}

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

/// Category C hardware intrinsics. The only domain that belongs here.
#[cfg(feature = "hardware")]
pub mod hardware;

// Parked compositions. Each is a `Program` builder that belongs in
// `vyre-libs`. Paths stay `vyre_primitives::<domain>` until the move.
// Do not add a domain here without listing it in `organization.rs`.

/// Text compositions (parked; belongs in `vyre-libs`).
#[cfg(feature = "text")]
pub mod text;

/// Pattern-matching compositions (parked).
#[cfg(feature = "matching")]
pub mod matching;

/// Decode compositions (parked).
#[cfg(feature = "decode")]
pub mod decode;

/// NFA compositions (parked).
#[cfg(feature = "nfa")]
pub mod nfa;

/// Hash compositions (parked).
#[cfg(feature = "hash")]
pub mod hash;

/// Math compositions (parked).
#[cfg(feature = "math")]
pub mod math;

/// Parsing compositions (parked).
#[cfg(feature = "parsing")]
pub mod parsing;

/// Neural-network compositions (parked).
#[cfg(feature = "nn")]
pub mod nn;

/// Graph compositions (parked): CSR, BFS, SCC, motif, toposort.
#[cfg(feature = "graph")]
pub mod graph;

/// Geometric / Clifford-algebra compositions (parked).
#[cfg(feature = "geom")]
pub mod geom;

/// Optimization compositions (parked).
#[cfg(feature = "opt")]
pub mod opt;

/// Topological-data-analysis compositions (parked).
#[cfg(feature = "topology")]
pub mod topology;

/// Visual pixel-map compositions (parked).
#[cfg(feature = "visual")]
pub mod visual;

/// Effects-typed pipeline compositions (parked).
#[cfg(feature = "effects")]
pub mod effects;

/// Type-discipline compositions (parked).
#[cfg(feature = "types")]
pub mod types;

/// Categorical compositions (parked).
#[cfg(feature = "cat")]
pub mod cat;

/// ZX-calculus rewrite compositions (parked).
#[cfg(feature = "zx")]
pub mod zx;

/// d-DNNF compiler compositions (parked).
#[cfg(feature = "dnnf")]
pub mod dnnf;

/// Bitset compositions (parked).
#[cfg(feature = "bitset")]
pub mod bitset;

/// Reduction compositions (parked).
#[cfg(feature = "reduce")]
pub mod reduce;

/// Label to NodeSet resolver (parked).
#[cfg(feature = "label")]
pub mod label;

/// Predicate compositions (parked).
#[cfg(feature = "predicate")]
pub mod predicate;

/// Fixpoint compositions (parked).
#[cfg(feature = "fixpoint")]
pub mod fixpoint;

/// Virtual-file-system DMA compositions (parked). Not its own Cargo
/// feature: any parked domain that pulls foundation enables it.
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
    pub use vyre_foundation::serial::envelope::{
        test_helpers, EnvelopeError, WireReader, WireWriter,
    };
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
