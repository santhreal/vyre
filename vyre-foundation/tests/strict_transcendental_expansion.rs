//! What a strict f32 expansion is allowed to contain, and what it must cover.
//!
//! The parity claim behind `FloatLoweringMode::StrictIeee` is structural, not
//! statistical: a device and the reference produce identical bits because the
//! expanded program contains only operations both are required to compute
//! identically. Accuracy is measured elsewhere, against the reference oracle
//! executing the same expansion. Here the shape is judged, because a single
//! division, fused multiply-add, native transcendental or decimal f32 literal
//! reintroduces the freedom the mode exists to remove, and no ulp measurement
//! on one device would reveal it.
//!
//! What this does not catch: whether the arithmetic computes the right function.
//! `vyre-reference/tests/strict_transcendental_accuracy.rs` owns that.

use std::collections::BTreeSet;

use vyre_foundation::fp_expansion::{
    expand_strict_transcendentals, is_strict_expandable, strict_expansion,
    strict_unexpandable_unary_ops,
};
use vyre_foundation::fp_parity::{approximable_unary_ops, is_approximable_unary_op};
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_foundation::visit::{for_each_expr, for_each_subexpr};

/// The binary operators a strict expansion may use.
///
/// Every one is required by IEEE-754 and by every shader dialect vyre emits to
/// be correctly rounded on f32, or is exact integer work on a bit pattern. A
/// `Div` or an `Fma` is absent because neither is: WGSL defers f32 division to
/// the hardware, and a fused multiply-add rounds once where the reference
/// rounds twice.
const PERMITTED_BINARY: &[BinOp] = &[
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Min,
    BinOp::Max,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::Eq,
    BinOp::Lt,
    BinOp::Gt,
    BinOp::Or,
];

/// The unary operators a strict expansion may use: the two bitcasts, and
/// nothing else. Every arithmetic unary operator either is one of the five being
/// expanded or is another approximable one.
const PERMITTED_UNARY: &[UnOp] = &[UnOp::BitcastF32ToU32, UnOp::BitcastU32ToF32];

/// The cast targets a strict expansion may use. An integer-to-float or
/// float-to-integer conversion of a value already known to be a small exact
/// integer is exact; no other width appears.
const PERMITTED_CAST: &[DataType] = &[DataType::U32, DataType::F32];

/// The five operators row 136 covers.
fn expandable() -> Vec<UnOp> {
    approximable_unary_ops()
        .into_iter()
        .filter(is_strict_expandable)
        .collect()
}

/// One expansion applied to an opaque argument, so nothing folds away.
fn expansion_of(op: &UnOp) -> Expr {
    strict_expansion(op, &Expr::var("x"))
        .unwrap_or_else(|| panic!("Fix: {op:?} must have a strict expansion"))
}

/// `out[0] = op(in[0])`, the smallest program that carries one operator.
fn unary_program(op: UnOp) -> Program {
    let index = Expr::u32(0);
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            index.clone(),
            Expr::UnOp {
                op,
                operand: Box::new(Expr::load("in", index)),
            },
        )],
    )
}

/// Every operator reached by `expr`, as two sorted sets plus the cast targets.
fn operators_in(expr: &Expr) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
    let mut binary = BTreeSet::new();
    let mut unary = BTreeSet::new();
    let mut casts = BTreeSet::new();
    for_each_subexpr(expr, &mut |candidate| match candidate {
        Expr::BinOp { op, .. } => {
            binary.insert(format!("{op:?}"));
        }
        Expr::UnOp { op, .. } => {
            unary.insert(format!("{op:?}"));
        }
        Expr::Cast { target, .. } => {
            casts.insert(format!("{target:?}"));
        }
        _ => {}
    });
    (binary, unary, casts)
}

fn names(items: impl IntoIterator<Item = impl core::fmt::Debug>) -> BTreeSet<String> {
    items.into_iter().map(|item| format!("{item:?}")).collect()
}

/// No expansion reaches an operator outside the permitted set.
///
/// Stated as a subset test against a list of what is allowed rather than a list
/// of what is forbidden, so an operator added to the IR is rejected until
/// someone admits it here on purpose.
#[test]
fn every_expansion_uses_only_correctly_rounded_operators() {
    let permitted_binary = names(PERMITTED_BINARY.iter());
    let permitted_unary = names(PERMITTED_UNARY.iter());
    let permitted_cast = names(PERMITTED_CAST.iter());

    for op in expandable() {
        let (binary, unary, casts) = operators_in(&expansion_of(&op));
        let stray_binary: Vec<_> = binary.difference(&permitted_binary).collect();
        let stray_unary: Vec<_> = unary.difference(&permitted_unary).collect();
        let stray_cast: Vec<_> = casts.difference(&permitted_cast).collect();
        assert!(
            stray_binary.is_empty() && stray_unary.is_empty() && stray_cast.is_empty(),
            "the strict expansion of {op:?} reaches operators outside the correctly-rounded set: \
             binary {stray_binary:?}, unary {stray_unary:?}, casts {stray_cast:?}. Any of those \
             may be computed differently by a device than by the reference, which is the whole \
             claim of FloatLoweringMode::StrictIeee"
        );
    }
}

/// Every permitted operator is actually reached by some expansion.
///
/// The subset test above passes trivially if the permitted list grows without
/// anyone using the entries. This is the other direction: an entry that no
/// expansion needs is permission nobody asked for.
#[test]
fn the_permitted_operator_set_has_no_unused_entry() {
    let mut binary_used = BTreeSet::new();
    let mut unary_used = BTreeSet::new();
    let mut casts_used = BTreeSet::new();
    for op in expandable() {
        let (binary, unary, casts) = operators_in(&expansion_of(&op));
        binary_used.extend(binary);
        unary_used.extend(unary);
        casts_used.extend(casts);
    }
    assert_eq!(
        names(PERMITTED_BINARY.iter()),
        binary_used,
        "the permitted binary set and the set the expansions reach must be the same set"
    );
    assert_eq!(names(PERMITTED_UNARY.iter()), unary_used);
    assert_eq!(names(PERMITTED_CAST.iter()), casts_used);
}

/// No expansion contains an f32 literal.
///
/// A decimal literal is rounded by the Rust parser and rounded again by the
/// shader compiler that reads the emitted text, and the two roundings are not
/// required to agree. Every f32 constant is instead stated as the exact 32 bits
/// it must be and reinterpreted, so this holds for the coefficients as well as
/// for the domain boundaries.
#[test]
fn no_expansion_contains_a_decimal_float_literal() {
    for op in expandable() {
        let mut literals = Vec::new();
        for_each_subexpr(&expansion_of(&op), &mut |candidate| {
            if let Expr::LitF32(value) = candidate {
                literals.push(*value);
            }
        });
        assert!(
            literals.is_empty(),
            "the strict expansion of {op:?} carries f32 literals {literals:?}; state each as its \
             bit pattern through UnOp::BitcastU32ToF32 instead"
        );
    }
}

/// An expanded program reaches no approximable operation.
///
/// The postcondition of the rewrite, judged through
/// `fp_parity::approximable_operations`, the same owner the ULP policy reads. A
/// position the walk failed to visit would show up here as a survivor.
#[test]
fn expansion_leaves_no_approximable_operation_in_the_program() {
    for op in expandable() {
        let program = unary_program(op.clone());
        assert_eq!(
            vyre_foundation::fp_parity::approximable_operations(&program).len(),
            1,
            "the fixture for {op:?} must contain exactly the one approximable operation"
        );
        let expanded = expand_strict_transcendentals(&program)
            .expect("Fix: an expandable operator must not be rejected")
            .expect("Fix: a program containing an expandable operator must be rewritten");
        assert!(
            vyre_foundation::fp_parity::approximable_operations(&expanded).is_empty(),
            "the expansion of {op:?} left an approximable operation in the program"
        );
    }
}

/// The rewrite reaches an operation nested inside a loop body and a branch.
///
/// The shallow case is the one every implementation gets right. This is the
/// position class: an operand of a node inside a child body, which is where a
/// hand-written per-variant walk drops a rewrite.
#[test]
fn expansion_reaches_an_operation_nested_in_a_child_body() {
    let index = Expr::u32(0);
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(1)),
            vec![Node::store(
                "out",
                index.clone(),
                Expr::sin(Expr::load("in", index)),
            )],
        )],
    );
    let expanded = expand_strict_transcendentals(&program)
        .expect("Fix: sin must not be rejected")
        .expect("Fix: a nested operation must still be rewritten");
    assert!(
        vyre_foundation::fp_parity::approximable_operations(&expanded).is_empty(),
        "a sin inside a branch body survived the expansion"
    );
}

/// A program with nothing to expand is not copied.
#[test]
fn a_program_with_no_approximable_operation_is_left_alone() {
    let index = Expr::u32(0);
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", index.clone(), Expr::f32(1.0))],
    );
    assert_eq!(
        expand_strict_transcendentals(&program).expect("Fix: nothing to reject"),
        None,
        "a program with no approximable operation must report no rewrite rather than a clone"
    );
}

/// Every approximable operator is either expanded or explicitly refused.
///
/// The closure test for this row. The variant space is
/// `fp_parity::approximable_unary_ops`, derived at run time from the frozen
/// builtin tag table, so an operator added to the approximability policy lands
/// in neither list only if someone deletes this assertion. The two lists must
/// partition the space, which is what makes "silently left approximate" an
/// unreachable state rather than a promise.
#[test]
fn every_approximable_operator_is_expanded_or_recorded_as_refused() {
    let approximable = approximable_unary_ops();
    let expanded = expandable();
    let refused = strict_unexpandable_unary_ops();

    assert_eq!(
        expanded.len() + refused.len(),
        approximable.len(),
        "expanded {expanded:?} plus refused {refused:?} must partition {approximable:?}"
    );
    assert!(
        names(expanded.iter())
            .intersection(&names(refused.iter()))
            .next()
            .is_none(),
        "no operator may be both expanded and refused"
    );
    for op in &approximable {
        assert_eq!(
            is_strict_expandable(op),
            strict_expansion(op, &Expr::var("x")).is_some(),
            "the classifier and the builder disagree about {op:?}, so one of them is a lie"
        );
    }
    assert_eq!(
        names(expanded.iter()),
        names([UnOp::Sin, UnOp::Cos, UnOp::Sqrt, UnOp::Exp, UnOp::Log].iter()),
        "row 136 covers exactly these five; extending the set is a deliberate change"
    );
}

/// A refused operator fails the dispatch instead of reaching the device.
///
/// The alternative, leaving it approximate, answers a bit-identity request with
/// the backend-dependent bits the mode exists to eliminate, and does it
/// silently.
#[test]
fn a_refused_operator_is_an_error_naming_itself() {
    for op in strict_unexpandable_unary_ops() {
        assert!(
            is_approximable_unary_op(&op),
            "the refused list must be a subset of the approximability policy"
        );
        let error = expand_strict_transcendentals(&unary_program(op.clone()))
            .expect_err("Fix: an operator with no expansion must be refused, not passed through");
        assert_eq!(error.operation(), format!("{op:?}"));
        assert!(
            error.to_string().contains(&format!("{op:?}")),
            "the diagnostic must name the operator that has no expansion: {error}"
        );
    }
}

/// The expansion of a nested pair reaches both operators.
///
/// `sin(exp(x))` is the adversarial shape for a rewrite that replaces a node and
/// then declines to descend into the replacement, or descends into it and
/// expands the same operator forever.
#[test]
fn a_nested_pair_of_operations_is_fully_expanded() {
    let index = Expr::u32(0);
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            index.clone(),
            Expr::sin(Expr::exp(Expr::load("in", index))),
        )],
    );
    let expanded = expand_strict_transcendentals(&program)
        .expect("Fix: both operators are expandable")
        .expect("Fix: a nested pair must be rewritten");
    assert!(
        vyre_foundation::fp_parity::approximable_operations(&expanded).is_empty(),
        "one of the two nested operations survived"
    );
    let mut count = 0usize;
    for_each_expr(expanded.entry(), |expr| {
        if matches!(expr, Expr::UnOp { op, .. } if matches!(op, UnOp::BitcastF32ToU32)) {
            count += 1;
        }
    });
    assert!(
        count > 0,
        "an expanded program must reach the exponent field through a bitcast"
    );
}
