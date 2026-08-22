//! The one owner of "draw an operator: a builtin from the frozen table, or an
//! opaque extension id".
//!
//! Two suites needed the same four strategies. The random-IR corpus in
//! `vyre-foundation` and the wire property suite in `vyre-spec` each carried a
//! copy, and the copies already differed: one drew `TernaryOp` and the other
//! did not, one boxed its strategies and the other returned `impl Strategy`,
//! and the opaque-arm weighting rule was stated twice. A property that draws
//! from a different corpus in each suite covers a set nobody chose.
//!
//! Which builtins exist is `spec_variant_tables.rs`'s decision, and this file
//! owns only how a draw is weighted and what an opaque arm carries. The
//! assertion each suite makes about the drawn operator stays in that suite.

#![allow(dead_code)]

#[path = "spec_variant_tables.rs"]
mod spec_variant_tables;

use proptest::prelude::*;
use spec_variant_tables::{
    builtin_atomic_ops, builtin_bin_ops, builtin_ternary_ops, builtin_un_ops,
};
use vyre_spec::extension::{
    ExtensionAtomicOpId, ExtensionBinOpId, ExtensionTernaryOpId, ExtensionUnOpId,
};
use vyre_spec::{AtomicOp, BinOp, TernaryOp, UnOp};

/// An extension operator id: the high bit is what separates the extension space
/// from the builtin tags, so it is always set.
pub(crate) fn extension_raw_id() -> BoxedStrategy<u32> {
    any::<u32>().prop_map(|raw| raw | 0x8000_0000).boxed()
}

/// Draw uniformly from a builtin variant table, with one extra opaque arm.
///
/// The builtins keep the combined weight they had when each was its own
/// `prop_oneof!` arm, so folding a table into a single `select` does not hand
/// the opaque arm half the corpus.
pub(crate) fn builtin_arm_with_opaque<T: std::fmt::Debug + Clone + 'static>(
    builtins: Vec<T>,
    opaque: BoxedStrategy<T>,
) -> BoxedStrategy<T> {
    let weight = u32::try_from(builtins.len()).expect("Fix: variant tables stay small.");
    prop_oneof![
        weight => prop::sample::select(builtins),
        1 => opaque,
    ]
    .boxed()
}

pub(crate) fn arb_bin_op() -> BoxedStrategy<BinOp> {
    builtin_arm_with_opaque(
        builtin_bin_ops(),
        extension_raw_id()
            .prop_map(|raw| BinOp::Opaque(ExtensionBinOpId(raw)))
            .boxed(),
    )
}

pub(crate) fn arb_un_op() -> BoxedStrategy<UnOp> {
    builtin_arm_with_opaque(
        builtin_un_ops(),
        extension_raw_id()
            .prop_map(|raw| UnOp::Opaque(ExtensionUnOpId(raw)))
            .boxed(),
    )
}

pub(crate) fn arb_atomic_op() -> BoxedStrategy<AtomicOp> {
    builtin_arm_with_opaque(
        builtin_atomic_ops(),
        extension_raw_id()
            .prop_map(|raw| AtomicOp::Opaque(ExtensionAtomicOpId(raw)))
            .boxed(),
    )
}

pub(crate) fn arb_ternary_op() -> BoxedStrategy<TernaryOp> {
    builtin_arm_with_opaque(
        builtin_ternary_ops(),
        extension_raw_id()
            .prop_map(|raw| TernaryOp::Opaque(ExtensionTernaryOpId(raw)))
            .boxed(),
    )
}
