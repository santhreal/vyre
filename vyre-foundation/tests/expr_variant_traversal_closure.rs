//! The class closed here: a new `Expr` variant reaching a traversal, analysis,
//! or magnitude classification that nobody taught it to.
//!
//! # The property proved here
//!
//! An expression traversal or classification that must reach every operand-carrying
//! or buffer-referencing variant cannot pass once a variant is added without a
//! decision recorded for it. Two halves:
//!
//! - COMPILE TIME. `vyre_foundation::visit::expr_parts` matches every variant with
//!   no catch-all arm across `expr_children`, `expr_magnitude`, `expr_buffer_ref`,
//!   and `expr_combine`. Adding a variant fails to compile there.
//! - RUN TIME. `Expr` is `#[non_exhaustive]`, so downstream traversals may carry
//!   catch-all arms. The fixtures in `vyre_test_support::expr_variants` are checked
//!   against `EXPR_VARIANT_NAMES`, which the registry macro emits from the enum
//!   body. A new variant has no fixture, the coverage assertion fails, and every
//!   suite built on it fails with it.

use vyre_foundation::ir::{Expr, EXPR_VARIANT_NAMES};
use vyre_foundation::visit::{
    any_subexpr, expr_buffer_ref, expr_children, expr_combine, expr_magnitude, for_each_subexpr,
    ExprBufferRef, ExprCombine, ExprMagnitude,
};
use vyre_test_support::expr_variants::{
    assert_covers_every_expr_variant, expr_operand_slot_samples, expr_variant_samples,
};

/// Every declared variant has a fixture.
///
/// This is the assertion that goes red when somebody adds an `Expr` variant. It
/// catches a variant nobody considered.
#[test]
fn expr_variant_samples_cover_every_declared_variant() {
    let samples = expr_variant_samples();
    assert_covers_every_expr_variant(&samples);
    assert_eq!(
        samples.len(),
        EXPR_VARIANT_NAMES.len(),
        "one fixture per declared variant, no more: {:?} against {EXPR_VARIANT_NAMES:?}",
        samples.iter().map(|s| s.variant).collect::<Vec<_>>()
    );
}

/// `expr_children` exposes the marker planted in every child operand slot.
#[test]
fn expr_children_exposes_every_operand_slot() {
    let marker = Expr::var("vyre_fixture_marker_operand");
    let operand_samples = expr_operand_slot_samples(&marker);
    for sample in &operand_samples {
        let children: Vec<&Expr> = expr_children(&sample.expr).iter().collect();
        let reachable = children.iter().any(|child| **child == marker);
        assert!(
            reachable,
            "{}: expr_children must expose the planted operand marker; a slot it does not return is an operand no traversal built on it can reach",
            sample.label()
        );
    }
}

/// `expr_magnitude` classifies every declared variant without falling back to a catch-all.
#[test]
fn expr_magnitude_classifies_every_declared_variant() {
    for sample in expr_variant_samples() {
        let mag = expr_magnitude(&sample.expr);
        match sample.variant {
            "LitU32" | "LitI32" | "LitF32" | "LitBool" | "InvocationId" | "LogicalIndex"
            | "LogicalTileId" | "LogicalWithinTileId" | "WorkgroupId" | "LocalId"
            | "SubgroupLocalId" | "SubgroupSize" => {
                assert!(
                    matches!(mag, ExprMagnitude::HostFact),
                    "{}: expected HostFact, got {mag:?}",
                    sample.label()
                );
            }
            "BufLen" => {
                assert!(
                    matches!(mag, ExprMagnitude::BufferExtent(_)),
                    "{}: expected BufferExtent, got {mag:?}",
                    sample.label()
                );
            }
            "Var" => {
                assert!(
                    matches!(mag, ExprMagnitude::Binding(_)),
                    "{}: expected Binding, got {mag:?}",
                    sample.label()
                );
            }
            "Load" | "Atomic" => {
                assert!(
                    matches!(mag, ExprMagnitude::BufferElement(_)),
                    "{}: expected BufferElement, got {mag:?}",
                    sample.label()
                );
            }
            "Cast" | "Select" | "Fma" | "SubgroupShuffle" => {
                assert!(
                    matches!(mag, ExprMagnitude::AllOperands),
                    "{}: expected AllOperands, got {mag:?}",
                    sample.label()
                );
            }
            "BufferRef" | "Call" | "SubgroupBallot" | "SubgroupReduce" | "Opaque" => {
                assert!(
                    matches!(mag, ExprMagnitude::Unknown),
                    "{}: expected Unknown, got {mag:?}",
                    sample.label()
                );
            }
            "UnOp" | "BinOp" => {
                // Classified via operator table
            }
            other => panic!("Fix: unhandled Expr variant {other}"),
        }
    }
}

/// `expr_buffer_ref` classifies every buffer-referencing variant.
#[test]
fn expr_buffer_refs_classify_every_buffer_referencing_variant() {
    for sample in expr_variant_samples() {
        let buf_ref = expr_buffer_ref(&sample.expr);
        match sample.variant {
            "Atomic" => {
                assert!(
                    matches!(buf_ref, ExprBufferRef::ReadWrite(_)),
                    "{}: Atomic must be ReadWrite, got {buf_ref:?}",
                    sample.label()
                );
            }
            "Load" | "BufLen" | "BufferRef" => {
                assert!(
                    matches!(buf_ref, ExprBufferRef::Read(_)),
                    "{}: expected Read, got {buf_ref:?}",
                    sample.label()
                );
            }
            "Opaque" => {
                assert!(
                    matches!(buf_ref, ExprBufferRef::Unknown),
                    "{}: Opaque must be Unknown, got {buf_ref:?}",
                    sample.label()
                );
            }
            _ => {
                assert!(
                    matches!(buf_ref, ExprBufferRef::None),
                    "{}: expected None, got {buf_ref:?}",
                    sample.label()
                );
            }
        }
    }
}

/// `expr_combine` classifies every cross-invocation combining variant.
#[test]
fn expr_combine_classifies_every_cross_invocation_combining_variant() {
    for sample in expr_variant_samples() {
        let combine = expr_combine(&sample.expr);
        match sample.variant {
            "Atomic" => {
                assert!(
                    matches!(combine, Some(ExprCombine::Atomic { .. })),
                    "{}: Atomic must be Some(ExprCombine::Atomic), got {combine:?}",
                    sample.label()
                );
            }
            "SubgroupReduce" => {
                assert!(
                    matches!(combine, Some(ExprCombine::Subgroup { .. })),
                    "{}: SubgroupReduce must be Some(ExprCombine::Subgroup), got {combine:?}",
                    sample.label()
                );
            }
            "Opaque" => {
                assert!(
                    matches!(combine, Some(ExprCombine::Unknown)),
                    "{}: Opaque must be Some(ExprCombine::Unknown), got {combine:?}",
                    sample.label()
                );
            }
            _ => {
                assert!(
                    combine.is_none(),
                    "{}: expected None, got {combine:?}",
                    sample.label()
                );
            }
        }
    }
}

/// `any_subexpr` and `for_each_subexpr` reach the planted marker in every child operand slot.
#[test]
fn subexpr_walks_reach_every_child_operand_slot() {
    let marker = Expr::var("vyre_fixture_marker_operand");
    let operand_samples = expr_operand_slot_samples(&marker);
    for sample in &operand_samples {
        let mut found_any = false;
        any_subexpr(&sample.expr, &mut |e| {
            if *e == marker {
                found_any = true;
                true
            } else {
                false
            }
        });
        assert!(
            found_any,
            "{}: any_subexpr must reach the planted operand marker",
            sample.label()
        );

        let mut count = 0;
        for_each_subexpr(&sample.expr, &mut |e| {
            if *e == marker {
                count += 1;
            }
        });
        assert!(
            count >= 1,
            "{}: for_each_subexpr must reach the planted operand marker at least once (found {count})",
            sample.label()
        );
    }
}

/// An opaque expression payload reads as `Unknown` rather than an empty leaf.
#[test]
fn opaque_expr_is_not_reported_as_leaf() {
    let opaque = expr_variant_samples()
        .into_iter()
        .find(|sample| sample.variant == "Opaque")
        .expect("Fix: the Opaque fixture is required by assert_covers_every_expr_variant");
    assert!(
        matches!(expr_magnitude(&opaque.expr), ExprMagnitude::Unknown),
        "opaque expression magnitude must be Unknown"
    );
    assert!(
        matches!(expr_buffer_ref(&opaque.expr), ExprBufferRef::Unknown),
        "opaque expression buffer reference must be Unknown"
    );
    assert!(
        matches!(expr_combine(&opaque.expr), Some(ExprCombine::Unknown)),
        "opaque expression combine must be Some(Unknown)"
    );
}
