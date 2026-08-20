//! The `vyre-libs` parsing and AST building domain library.
//!
//! Exposes registered semantic operations for structural analysis and
//! full-grammar Shunting-Yard AST generation entirely on GPU.
//!
//! Architected as disjoint, language-isolated registered passes:
//!
//! - `core`  -  substrate-neutral parsing primitives (AST node kinds,
//!   delimiter handling, grammar table walkers).
//! - `lr_tables`  -  precomputed LR action/goto tables and the CPU reference
//!   parser that walks them.
//! - `vast`  -  the packed AST wire format and its host walks.
//! - `source_cache`  -  content-hash LRU cache for parsed source artifacts.
//! - `parallel_parse`  -  parallel corpus parse on top of that cache.
//! - `go`  -  Go 1.21 lex + structural extraction.
//!   Feature-gated behind `go-parser`.
//! - `python`  -  Python 3.12 sparse lex + structural extraction.
//!   Feature-gated behind `python-parser`.

/// Substrate-neutral parsing primitives (AST, delimiter, grammar).
///
/// Behind `parsing` because the three registrations under it carry
/// `vyre-libs::parsing::` op ids, and that is the feature the operation
/// schema routes those ids to. The substrate below (`composition`,
/// `source_cache`, `lr_tables`, `vast`, `parallel_parse`) registers nothing
/// and stays available to a build that only wants the kernels.
#[cfg(feature = "parsing")]
pub mod core;

/// Content-hash LRU cache for parsed source artifacts.
/// substrate; language-specific parse pipelines opt in via
/// `ParsedSourceLru::get_or_parse`.
pub mod source_cache;

/// Parallel corpus parse on top of the L2 LRU cache.
/// substrate; fans `get_or_parse` across cores with `rayon` while
/// preserving input order.
pub mod parallel_parse;

pub(crate) mod composition;

/// Precomputed LR action/goto tables and CPU reference parser.
pub mod lr_tables;

/// Packed AST (VAST) wire + host walks  -  re-export from `vyre-foundation`.
pub mod vast;

/// Go 1.21 pipeline (lex / structural parse / AST ops).
#[cfg(feature = "go-parser")]
pub mod go;

/// Python 3.12 pipeline (lex / structural parse / AST ops).
#[cfg(feature = "python-parser")]
pub mod python;

/// Generic delimiter-depth scan for paired delimiter token streams.
#[cfg(feature = "parsing-kernels")]
pub mod core_delimiter_match;

/// SSA dominance-frontier phi discovery scan.
#[cfg(feature = "parsing-kernels")]
pub mod ssa_dominance_scan;

/// Shared AST opcode constants.
#[cfg(feature = "parsing-kernels")]
pub mod ast_ops;

/// Pack an opcode to handler dispatch table into one u32 per entry for fast
/// GPU-side bytecode interpretation. Foundational primitive for
/// warp-specialized interpreter loops where every thread executes the same
/// opcode in the same warp.
#[cfg(feature = "parsing-kernels")]
pub mod bytecode_dispatch_table_pack;

/// Word-at-a-time whitespace classification (#P-PRIM-WS-CLASSIFY).
/// Foundational primitive for structural parsers (JSON, CSV, HTTP, INI):
/// loads 4 bytes per u32, emits a 4-bit per-word "is-whitespace" mask
/// using pure arithmetic (no per-byte branches, so no warp divergence).
/// Composes with `stream_compact` for the canonical simdjson-style
/// whitespace-skip pipeline.
#[cfg(feature = "parsing-kernels")]
pub mod whitespace_classify_word;

/// Per-byte kept-mask for C translation phase 2 (backslash-newline deletion).
/// One thread per input byte; a two-byte sliding window classifies each of
/// the five splice cases. Composes with `stream_compact` to materialise
/// the post-phase-2 byte stream and the original-offset map.
#[cfg(feature = "parsing-kernels")]
pub mod line_splice_classify;

/// AST-level constant-folding wave operating on packed-AST u32 buffers.
/// NOT the vyre-IR `optimizer::passes::fusion_cse::cse`: the `ast_` prefix
/// marks this as a parsing-domain primitive that runs against a packed-AST
/// representation, not against `Expr` / `Node` of the IR.
#[cfg(feature = "parsing-kernels")]
pub mod ast_cse_constant_fold;

/// AST-level structural-hash CSE probe/insert wave operating on packed-AST
/// u32 buffers. NOT the vyre-IR CSE; the `ast_` prefix disambiguates.
#[cfg(feature = "parsing-kernels")]
pub mod ast_cse_structural_hash;

/// 2D / planar grammar rewrite scheduler. Picks a maximal
/// non-overlapping set of `k x k` matches to apply in one wave.
#[cfg(feature = "parsing-kernels")]
pub mod planar_rewrite;
