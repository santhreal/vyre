//! The class closed here: two consumers each writing out their own list of
//! "which binary operators are arithmetic".
//!
//! `validate::typecheck` asked the question twice, once to decide an
//! expression's static type and once to decide what its operands must be, and
//! wrote a separate operator list for each. The lists had already drifted:
//! `AbsDiff` was in the operand list and not in the type list, which is
//! correct, but nothing said so and nothing would have caught it if it were
//! wrong in the other direction. `BinOp::result_class` is now the one answer
//! and `BinOp::takes_numeric_operands` is derived from it.
//!
//! What this file proves is the discriminating cases: an operator whose two
//! answers differ, one from each of the other three classes, and the extension
//! variant. What it does NOT prove is that every operator is classified, since
//! that is a compile error inside `result_class`'s exhaustive match rather than
//! a run-time property, and restating the variant list here would be the third
//! list this change exists to remove.

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

#[test]
fn an_arithmetic_operator_carries_its_operand_type_through() {
    for op in [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Min,
        BinOp::Max,
        BinOp::SaturatingAdd,
        BinOp::SaturatingSub,
        BinOp::SaturatingMul,
    ] {
        assert_eq!(
            op.result_class(),
            BinOpResult::Numeric,
            "{op:?} produces whatever its operands were"
        );
        assert!(op.takes_numeric_operands(), "{op:?} rejects Bool operands");
    }
}

#[test]
fn a_comparison_or_a_logical_connective_produces_a_predicate() {
    for op in [
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Le,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
    ] {
        assert_eq!(
            op.result_class(),
            BinOpResult::Predicate,
            "{op:?} evaluates to Bool whatever its operands were"
        );
        assert!(
            !op.takes_numeric_operands(),
            "{op:?} is not numeric arithmetic, so the arithmetic operand rules must not fire"
        );
    }
}

/// A bit-level operator is integer-typed and is NOT numeric arithmetic.
///
/// The distinction matters: were `Shl` to answer `true` here, the arithmetic
/// arm of the operand checker would claim it and its own arm, which rejects a
/// non-integer shift amount, would become unreachable.
#[test]
fn a_bit_level_operator_is_integer_typed_and_not_numeric_arithmetic() {
    for op in [
        BinOp::Mod,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::RotateLeft,
        BinOp::RotateRight,
        BinOp::MulHigh,
        BinOp::WrappingAdd,
        BinOp::WrappingSub,
    ] {
        assert_eq!(op.result_class(), BinOpResult::Integer, "{op:?} is integer-typed");
        assert!(
            !op.takes_numeric_operands(),
            "{op:?} must reach its own operand arm, not the arithmetic one"
        );
    }
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
