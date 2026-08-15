//! The class closed here: two consumers each writing out their own list of
//! "which binary operators are arithmetic".
//!
//! `validate::typecheck` asked the question twice, once to decide an
//! expression's static type and once to decide what its operands must be, and
//! wrote a separate operator list for each. The lists had already drifted on
//! `AbsDiff`, which is in the operand list and not the type list. That is
//! correct, nothing said so, and nothing would have caught it wrong in the
//! other direction. `BinOp::result_class` is now the one answer and
//! `BinOp::takes_numeric_operands` is derived from it.
//!
//! Every property below runs over `builtin_bin_ops`, the frozen variant table,
//! rather than a list typed here: a list typed here would be the third copy
//! this change exists to remove, and it would go stale in silence the first
//! time an operator was added. What the table cannot prove is that each
//! operator got the RIGHT class, so the one case where the two answers differ
//! is pinned by name.

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use spec_variant_tables::builtin_bin_ops;
use vyre_spec::bin_op::{BinOp, BinOpResult};
use vyre_spec::extension::ExtensionBinOpId;

/// The case that made two lists necessary and then let them drift.
#[test]
fn abs_diff_takes_numeric_operands_and_still_produces_an_integer() {
    assert_eq!(
        BinOp::AbsDiff.result_class(),
        BinOpResult::Integer,
        "AbsDiff is an unsigned difference: its result is not its operand type"
    );
    assert!(
        BinOp::AbsDiff.takes_numeric_operands(),
        "AbsDiff still rejects a Bool operand, which is what the operand check is for"
    );
}

/// `Extension` is the answer for an out-of-tree operator only.
///
/// A builtin classified `Extension` would tell every consumer that core does
/// not know its result type, which for an operator core defines is a refusal to
/// decide rather than a decision.
#[test]
fn no_builtin_operator_defers_its_result_class_to_an_extension() {
    let deferred: Vec<BinOp> = builtin_bin_ops()
        .into_iter()
        .filter(|op| op.result_class() == BinOpResult::Extension)
        .collect();
    assert!(
        deferred.is_empty(),
        "Fix: classify these builtin operators in BinOp::result_class: {deferred:?}"
    );
}

/// An operator whose result is its operand type must accept numeric operands,
/// or the two answers contradict each other: the type walker would propagate an
/// operand type the operand checker had just rejected.
#[test]
fn every_operator_that_returns_its_operand_type_accepts_numeric_operands() {
    for op in builtin_bin_ops()
        .into_iter()
        .filter(|op| op.result_class() == BinOpResult::Numeric)
    {
        assert!(
            op.takes_numeric_operands(),
            "{op:?} returns its operand type but its operands are not numeric"
        );
    }
}

/// A comparison or logical connective evaluates to `Bool` whatever its operands
/// were, so the numeric-arithmetic operand rules must not claim it. Were one to
/// answer `true`, the arithmetic arm of the operand checker would swallow it and
/// the comparison arm, which requires both sides to have the SAME type rather
/// than a numeric one, would become unreachable for it.
#[test]
fn a_predicate_operator_is_not_numeric_arithmetic() {
    for op in builtin_bin_ops()
        .into_iter()
        .filter(|op| op.result_class() == BinOpResult::Predicate)
    {
        assert!(
            !op.takes_numeric_operands(),
            "{op:?} evaluates to Bool and must reach the comparison arm, not the arithmetic one"
        );
    }
}

/// `AbsDiff` is the only operator whose result class and operand discipline
/// disagree.
///
/// This is the pin that would have caught the original drift in either
/// direction. A second such operator is legitimate, and it has to be recorded
/// here when it arrives rather than appearing by accident: a bit-level operator
/// that quietly starts accepting `f32` operands is how the shift arm, which
/// rejects a non-integer shift amount, stops being reached.
#[test]
fn abs_diff_is_the_only_integer_typed_operator_taking_numeric_operands() {
    let mixed: Vec<BinOp> = builtin_bin_ops()
        .into_iter()
        .filter(|op| op.result_class() == BinOpResult::Integer && op.takes_numeric_operands())
        .collect();
    assert_eq!(
        mixed,
        vec![BinOp::AbsDiff],
        "Fix: an integer-typed operator that accepts numeric operands bypasses its own \
         operand arm. Record it here with the reason, or correct its classification."
    );
}

/// An out-of-tree operator declares its own result type, so this contract
/// declines to guess one for it rather than folding it in with the integers.
#[test]
fn an_extension_operator_is_its_own_class() {
    let op = BinOp::Opaque(ExtensionBinOpId(7));
    assert_eq!(op.result_class(), BinOpResult::Extension);
    assert!(
        !op.takes_numeric_operands(),
        "core cannot know an extension operator's operand discipline"
    );
}
