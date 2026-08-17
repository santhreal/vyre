//! One fixture per `Expr` variant, so traversals, validators, and emitters can be
//! tested against the entire expression universe derived from source.
//!
//! # Why this exists in a shared crate
//!
//! `Expr` is `#[non_exhaustive]`. Matches across the compiler and drivers rely on catch-all
//! arms or visitor dispatch. A variant added tomorrow without derived coverage would silently
//! fall into default paths, skipping folding, optimization, or lowering.
//!
//! [`expr_variant_samples`] covers every name in `vyre_foundation::ir::EXPR_VARIANT_NAMES`
//! (and the source declaration of `Expr`), and [`assert_covers_every_expr_variant`] guarantees
//! that any new variant turns all dependent suites RED until explicitly handled.

use std::collections::BTreeSet;
use std::sync::Arc;

use vyre_foundation::ir::{
    AtomicOp, BinOp, DataType, Expr, ExprNode, MemoryOrdering, SubgroupReduceOp, UnOp,
    EXPR_VARIANT_NAMES,
};

/// One `Expr` fixture together with its slot metadata.
#[derive(Debug, Clone)]
pub struct ExprSample {
    /// Declared variant name, matching `vyre_foundation::ir::EXPR_VARIANT_NAMES`.
    pub variant: &'static str,
    /// Which operand slot the marker was planted in, if any.
    pub slot: Option<&'static str>,
    /// The fixture expression.
    pub expr: Expr,
}

impl ExprSample {
    /// Human-readable label for assertion messages.
    #[must_use]
    pub fn label(&self) -> String {
        match self.slot {
            Some(slot) => format!("Expr::{}.{slot}", self.variant),
            None => format!("Expr::{}", self.variant),
        }
    }
}

crate::test_expr_extension!(
    FixtureExprExtension,
    kind: "vyre.test_support.fixture_expr",
    identity: "fixture-expr",
    result_type: Some(DataType::U32),
    cse_safe: true,
    fingerprint: 0x42,
);

fn sample(variant: &'static str, slot: Option<&'static str>, expr: Expr) -> ExprSample {
    ExprSample {
        variant,
        slot,
        expr,
    }
}

/// Every declared `Expr` variant exactly once, with default/inert payload.
#[must_use]
pub fn expr_variant_samples() -> Vec<ExprSample> {
    let mut out: Vec<ExprSample> = Vec::new();
    for mut candidate in inert_samples() {
        if out
            .iter()
            .any(|existing| existing.variant == candidate.variant)
        {
            continue;
        }
        candidate.slot = None;
        out.push(candidate);
    }
    out
}

/// Every child operand slot of every operand-nesting `Expr` variant with `marker` planted in it.
#[must_use]
pub fn expr_operand_slot_samples(marker: &Expr) -> Vec<ExprSample> {
    vec![
        sample("Load", Some("index"), Expr::load("buf", marker.clone())),
        sample(
            "BinOp",
            Some("left"),
            Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(marker.clone()),
                right: Box::new(Expr::u32(0)),
            },
        ),
        sample(
            "BinOp",
            Some("right"),
            Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::u32(0)),
                right: Box::new(marker.clone()),
            },
        ),
        sample(
            "UnOp",
            Some("operand"),
            Expr::UnOp {
                op: UnOp::Negate,
                operand: Box::new(marker.clone()),
            },
        ),
        sample(
            "Select",
            Some("cond"),
            Expr::select(marker.clone(), Expr::u32(1), Expr::u32(2)),
        ),
        sample(
            "Select",
            Some("true_val"),
            Expr::select(Expr::bool(true), marker.clone(), Expr::u32(2)),
        ),
        sample(
            "Select",
            Some("false_val"),
            Expr::select(Expr::bool(true), Expr::u32(1), marker.clone()),
        ),
        sample(
            "Cast",
            Some("value"),
            Expr::cast(DataType::U32, marker.clone()),
        ),
        sample(
            "Fma",
            Some("a"),
            Expr::fma(marker.clone(), Expr::f32(1.0), Expr::f32(0.0)),
        ),
        sample(
            "Fma",
            Some("b"),
            Expr::fma(Expr::f32(1.0), marker.clone(), Expr::f32(0.0)),
        ),
        sample(
            "Fma",
            Some("c"),
            Expr::fma(Expr::f32(1.0), Expr::f32(1.0), marker.clone()),
        ),
        sample(
            "Atomic",
            Some("index"),
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: "buf".into(),
                index: Box::new(marker.clone()),
                expected: None,
                value: Box::new(Expr::u32(1)),
                ordering: MemoryOrdering::Relaxed,
            },
        ),
        sample(
            "Atomic",
            Some("value"),
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: "buf".into(),
                index: Box::new(Expr::u32(0)),
                expected: None,
                value: Box::new(marker.clone()),
                ordering: MemoryOrdering::Relaxed,
            },
        ),
        sample(
            "SubgroupBallot",
            Some("cond"),
            Expr::subgroup_ballot(marker.clone()),
        ),
        sample(
            "SubgroupShuffle",
            Some("value"),
            Expr::subgroup_shuffle(marker.clone(), Expr::u32(0)),
        ),
        sample(
            "SubgroupShuffle",
            Some("lane"),
            Expr::subgroup_shuffle(Expr::u32(0), marker.clone()),
        ),
        sample(
            "SubgroupReduce",
            Some("value"),
            Expr::subgroup_reduce(SubgroupReduceOp::Add, marker.clone()),
        ),
    ]
}

fn inert_samples() -> Vec<ExprSample> {
    vec![
        sample("LitU32", None, Expr::u32(0)),
        sample("LitI32", None, Expr::i32(0)),
        sample("LitF32", None, Expr::f32(0.0)),
        sample("LitBool", None, Expr::bool(false)),
        sample("Var", None, Expr::var("x")),
        sample("BufferRef", None, Expr::buffer_ref("buf")),
        sample("Load", None, Expr::load("buf", Expr::u32(0))),
        sample("BufLen", None, Expr::buf_len("buf")),
        sample("InvocationId", None, Expr::InvocationId { axis: 0 }),
        sample("WorkgroupId", None, Expr::WorkgroupId { axis: 0 }),
        sample("LocalId", None, Expr::LocalId { axis: 0 }),
        sample(
            "BinOp",
            None,
            Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::u32(0)),
                right: Box::new(Expr::u32(0)),
            },
        ),
        sample(
            "UnOp",
            None,
            Expr::UnOp {
                op: UnOp::Negate,
                operand: Box::new(Expr::i32(0)),
            },
        ),
        sample("Call", None, Expr::call("vyre.test", vec![Expr::u32(0)])),
        sample(
            "Select",
            None,
            Expr::select(Expr::bool(true), Expr::u32(1), Expr::u32(0)),
        ),
        sample("Cast", None, Expr::cast(DataType::U32, Expr::u32(0))),
        sample(
            "Fma",
            None,
            Expr::fma(Expr::f32(0.0), Expr::f32(0.0), Expr::f32(0.0)),
        ),
        sample(
            "Atomic",
            None,
            Expr::Atomic {
                op: AtomicOp::Add,
                buffer: "buf".into(),
                index: Box::new(Expr::u32(0)),
                expected: None,
                value: Box::new(Expr::u32(1)),
                ordering: MemoryOrdering::Relaxed,
            },
        ),
        sample(
            "SubgroupBallot",
            None,
            Expr::subgroup_ballot(Expr::bool(true)),
        ),
        sample(
            "SubgroupShuffle",
            None,
            Expr::subgroup_shuffle(Expr::u32(0), Expr::u32(0)),
        ),
        sample(
            "SubgroupReduce",
            None,
            Expr::subgroup_reduce(SubgroupReduceOp::Add, Expr::u32(0)),
        ),
        sample("SubgroupLocalId", None, Expr::subgroup_local_id()),
        sample("SubgroupSize", None, Expr::subgroup_size()),
        sample("Opaque", None, Expr::Opaque(Arc::new(FixtureExprExtension))),
    ]
}

/// The variant names declared for `Expr`, read from the authoritative catalog.
#[must_use]
pub fn declared_expr_variants() -> BTreeSet<String> {
    EXPR_VARIANT_NAMES.iter().map(|&s| s.to_string()).collect()
}

/// Panic unless `samples` covers every declared `Expr` variant.
///
/// # Panics
/// Panics if any declared variant has no sample, or if a sample names an undeclared variant.
pub fn assert_covers_every_expr_variant(samples: &[ExprSample]) {
    let declared = declared_expr_variants();
    let covered: BTreeSet<String> = samples.iter().map(|s| s.variant.to_string()).collect();

    let missing: BTreeSet<_> = declared.difference(&covered).cloned().collect();
    assert!(
        missing.is_empty(),
        "missing Expr variant sample(s): {missing:?}; add them to expr_variant_samples"
    );

    let unexpected: BTreeSet<_> = covered.difference(&declared).cloned().collect();
    assert!(
        unexpected.is_empty(),
        "unexpected Expr variant sample(s) not in declared set: {unexpected:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_variant_samples_covers_every_declared_variant() {
        let samples = expr_variant_samples();
        assert_covers_every_expr_variant(&samples);
    }
}
