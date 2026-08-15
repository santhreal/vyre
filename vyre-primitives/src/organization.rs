//! One classification of every `vyre-primitives` Cargo feature.
//!
//! An operation is admitted here only when it cannot be composed, which means
//! it needs its own arm in a backend emitter and its own arm in the reference
//! interpreter. Marker types are always on; `hardware` is the intrinsic domain.
//! Every other domain feature named below is a composition still resident here,
//! and each one leaving is a move to `vyre-libs`, not a reclassification. Do not
//! add a domain feature without putting it in one of these lists.
//!
//! `tests/feature_classification.rs` fails if `Cargo.toml` grows a
//! feature that is not classified here.

/// Domain feature that belongs in this crate: Category C hardware
/// intrinsics that need a dedicated emitter arm and a dedicated
/// reference-interpreter arm.
pub const INTRINSIC_FEATURES: &[&str] = &["hardware"];

/// Domain features that are compositions still resident in this crate.
///
/// Reuse count is not an admission criterion. Each of these builds a `Program`
/// from existing IR and belongs in `vyre-libs`. A name leaves this list only by
/// the domain moving; `cat`, `zx`, `dnnf`, `types` and `effects` left that way.
pub const COMPOSITION_FEATURES: &[&str] = &[
    "bitset",
    "decode",
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
    "visual",
];

/// Crate-support features. Not domains.
pub const SUPPORT_FEATURES: &[&str] = &[
    "all-lego",
    "cpu-parity",
    "gpu",
    "inventory-registry",
    "vyre-foundation",
];
