//! The canonical evaluator is compositional and iterative.
//!
//! WHY: `execution::hashmap` walks an expression on an explicit frame stack.
//! A frame stack reaches for operands by slot, so an off-by-one push, a slot
//! reused across sibling frames, or a branch that leaves its condition on the
//! stack produces a value that is wrong without being ill-typed, and every
//! fixed-input test in this directory still passes. The property that catches
//! that class is compositionality: evaluating a whole tree must equal folding
//! the same tree one node at a time, substituting each sub-result back as a
//! literal. Only an evaluator whose frames are independent satisfies it for
//! every generated shape.
//!
//! The second contract is reach: the evaluator must return a value for every
//! expression the validator admits, and the validator admits nesting up to
//! `DEFAULT_MAX_EXPR_DEPTH`. The bound is read from that constant rather than
//! copied, so raising the admitted depth without widening what the evaluator
//! can walk turns this red instead of turning a deep program into an abort.
//!
//! What this does not catch: whether any individual operator computes the
//! right arithmetic. That is pinned per operator by the `value_*` matrices and
//! the `composition_witness` suites. The operator set below is a generator of
//! tree shapes, chosen because every one of them is total over `u32`, so a
//! divergence here is the fold and never a refused operand.

use proptest::prelude::*;
use vyre_foundation::ir::{Expr, Program};
use vyre_foundation::validate::DEFAULT_MAX_EXPR_DEPTH;
use vyre_reference::value::Value;
use vyre_reference::workgroup::InvocationIds;
use vyre_reference::{reference_eval_expr, ReferenceMemory};

/// Evaluate one closed expression through the canonical evaluator.
fn evaluate(expr: &Expr) -> Value {
    let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let mut memory = ReferenceMemory::empty();
    reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, expr).unwrap_or_else(|error| {
        panic!("Fix: the canonical evaluator must evaluate a closed literal expression: {error}")
    })
}

/// The literal expression that denotes `value`.
fn literal(value: &Value) -> Expr {
    match value {
        Value::U32(value) => Expr::u32(*value),
        Value::I32(value) => Expr::i32(*value),
        Value::U64(value) => Expr::u64(*value),
        Value::Bool(value) => Expr::bool(*value),
        Value::Float(value) => Expr::f32(*value as f32),
        other => panic!(
            "Fix: the generated subset must produce a value with a literal form, got {other:?}"
        ),
    }
}

/// Evaluate `expr` bottom up, substituting each sub-result back as a literal.
///
/// A leaf is handed to the evaluator directly, which is what makes this a fold
/// over the same semantics rather than a second implementation of them.
fn fold_bottom_up(expr: &Expr) -> Value {
    match expr {
        Expr::BinOp { op, left, right } => {
            let left = literal(&fold_bottom_up(left));
            let right = literal(&fold_bottom_up(right));
            evaluate(&Expr::BinOp {
                op: *op,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            if fold_bottom_up(cond).truthy() {
                fold_bottom_up(true_val)
            } else {
                fold_bottom_up(false_val)
            }
        }
        Expr::Fma { a, b, c } => {
            let a = literal(&fold_bottom_up(a));
            let b = literal(&fold_bottom_up(b));
            let c = literal(&fold_bottom_up(c));
            evaluate(&Expr::Fma {
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            })
        }
        leaf => evaluate(leaf),
    }
}

/// Trees over operators that are total on `u32`, plus the two shapes whose
/// frames are not a plain binary pair.
fn total_u32_expression() -> impl Strategy<Value = Expr> {
    any::<u32>()
        .prop_map(Expr::u32)
        .prop_recursive(5, 64, 3, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(l, r)| l.wrapping_add(r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| l.wrapping_sub(r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::min(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::max(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::abs_diff(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitxor(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitand(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitor(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::rotate_left(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::mulhi(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::saturating_add(l, r)),
                (inner.clone(), inner.clone(), inner.clone()).prop_map(|(cond, t, f)| {
                    Expr::select(Expr::lt(cond.clone(), t.clone()), t, f)
                }),
            ]
        })
}

/// Float trees, so the fold is proved over the arm that canonicalizes.
fn float_expression() -> impl Strategy<Value = Expr> {
    any::<u32>()
        .prop_map(|bits| Expr::f32(f32::from_bits(bits)))
        .prop_recursive(4, 32, 3, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::min(l, r)),
                (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::max(l, r)),
                (inner.clone(), inner.clone(), inner.clone())
                    .prop_map(|(a, b, c)| Expr::fma(a, b, c)),
            ]
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn a_generated_integer_tree_folds_to_the_value_the_evaluator_returns(
        expr in total_u32_expression(),
    ) {
        prop_assert_eq!(evaluate(&expr), fold_bottom_up(&expr));
    }

    #[test]
    fn a_generated_float_tree_folds_to_the_value_the_evaluator_returns(
        expr in float_expression(),
    ) {
        prop_assert_eq!(evaluate(&expr), fold_bottom_up(&expr));
    }
}

/// The deepest expression the validator admits still evaluates.
#[test]
fn the_deepest_admitted_nesting_evaluates() {
    // One literal plus one binary node per level reaches the declared limit.
    let levels = u32::try_from(DEFAULT_MAX_EXPR_DEPTH - 1)
        .expect("Fix: the declared expression depth limit must fit in a u32");
    let mut expr = Expr::u32(0);
    for _ in 0..levels {
        expr = expr.wrapping_add(Expr::u32(1));
    }
    assert_eq!(evaluate(&expr), Value::U32(levels));
}
