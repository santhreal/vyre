//! One classification of every `vyre-primitives` Cargo feature.
//!
//! The workspace charter admits two things into this crate: marker types
//! (always on) and uncomposable hardware intrinsics (`hardware`). Every
//! other domain feature is a composition that belongs in `vyre-libs`.
//! Paths stay `vyre_primitives::<domain>` until that move. Do not add a
//! domain feature without putting it in one of these lists.
//!
//! `tests/feature_classification.rs` fails if `Cargo.toml` grows a
//! feature that is not classified here.

/// Domain feature that belongs in this crate: Category C hardware
/// intrinsics that need a dedicated emitter arm and a dedicated
/// reference-interpreter arm.
pub const INTRINSIC_FEATURES: &[&str] = &["hardware"];

/// Domain features that are compositions parked in this crate.
///
/// Reuse count is not an admission criterion. Each of these builds a
/// `Program` from existing IR and belongs in `vyre-libs`.
pub const COMPOSITION_FEATURES: &[&str] = &[
    "bitset",
    "cat",
    "decode",
    "dnnf",
    "effects",
    "fixpoint",
    "geom",
    "graph",
    "hash",
    "label",
    "matching",
    "math",
    "nfa",
    "nn",
    "opt",
    "parsing",
    "predicate",
    "reduce",
    "text",
    "topology",
    "types",
    "visual",
    "zx",
];

/// Crate-support features. Not domains.
pub const SUPPORT_FEATURES: &[&str] = &[
    "all-lego",
    "cpu-parity",
    "gpu",
    "inventory-registry",
    "vyre-foundation",
];
