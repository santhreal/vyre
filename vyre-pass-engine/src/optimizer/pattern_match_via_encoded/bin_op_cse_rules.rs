//! The literal rules plus the rules that need canonical identity.
//!
//! `x ^ x`, `x - x`, `x & x` and `x | x` only fire when both operands are the
//! same value after CSE, which the `canonical` column answers. Every binding
//! the literal rules make is in scope here so an equality rule can read it
//! without recomputing the operand. The rules are tables over the shapes in
//! `rule_shapes`, the same shapes the literal-only body uses.

use vyre_foundation::ir::{Expr, Node};

use super::bin_op_rules::bin_op_match_body;
use super::rewrite_action::{
    REPLACE_WITH_LEFT, REPLACE_WITH_LEFT_INNER_LEFT, REPLACE_WITH_LEFT_INNER_RIGHT,
    REPLACE_WITH_LIT_FALSE, REPLACE_WITH_LIT_TRUE, REPLACE_WITH_LIT_ZERO, REPLACE_WITH_RIGHT,
};
use super::rule_shapes::Literal::U32;
use super::rule_shapes::Operand::{Left, Right};
use super::rule_shapes::{
    equal_operands, left_child_cancellation, literal, operator_flag, step_node, Step,
};

/// Operator tags only the canonical rules read.
const OPERATOR_FLAGS: &[(&str, u32)] = &[
    ("is_min", 0x15),
    ("is_max", 0x16),
    ("is_absdiff", 0x14),
    ("is_sat_add", 0x17),
    ("is_sat_sub", 0x18),
    ("is_sat_mul", 0x19),
    ("is_wrap_add", 0x20),
    ("is_wrap_sub", 0x21),
];

/// Rewrites that fire when one operator holds and the operands are one value.
///
/// `(Add ?x ?x)` is deliberately absent: doubling has no canonical winner here,
/// and a pass that prefers a shift wants to see the pattern intact.
const EQUAL_OPERAND_RULES: &[(&str, u32)] = &[
    ("is_sat_sub", REPLACE_WITH_LIT_ZERO), // (SaturatingSub ?x ?x) → 0u
    ("is_wrap_sub", REPLACE_WITH_LIT_ZERO), // (WrappingSub ?x ?x) → 0u
    ("is_min", REPLACE_WITH_LEFT),         // (Min ?x ?x) → ?x
    ("is_max", REPLACE_WITH_LEFT),         // (Max ?x ?x) → ?x
    ("is_absdiff", REPLACE_WITH_LIT_ZERO), // (AbsDiff ?x ?x) → 0u32
    ("is_sub", REPLACE_WITH_LIT_ZERO),     // (Sub ?x ?x) → 0u32
    ("is_bitxor", REPLACE_WITH_LIT_ZERO),  // (BitXor ?x ?x) → 0u32
    ("is_bitand", REPLACE_WITH_LEFT),      // (BitAnd ?x ?x) → ?x
    ("is_bitor", REPLACE_WITH_LEFT),       // (BitOr ?x ?x) → ?x
    ("is_bool_and", REPLACE_WITH_LEFT),    // (And ?x ?x) → ?x
    ("is_bool_or", REPLACE_WITH_LEFT),     // (Or ?x ?x) → ?x
];

/// Rewrites where the left child is a binary node that cancels the right.
///
/// Each row is the outer operator, the inner operator tag, the canonical
/// binding of the inner operand that must match the right child, and the
/// surviving operand.
const CANCELLATION_RULES: &[(&str, u32, &str, u32)] = &[
    // (Sub (Add a b) c) with a == c takes b, and with b == c takes a.
    (
        "is_sub",
        0x01,
        "l_inner_left_canon",
        REPLACE_WITH_LEFT_INNER_RIGHT,
    ),
    (
        "is_sub",
        0x01,
        "l_inner_right_canon",
        REPLACE_WITH_LEFT_INNER_LEFT,
    ),
    // (Add (Sub a b) b) → a.
    (
        "is_add",
        0x02,
        "l_inner_right_canon",
        REPLACE_WITH_LEFT_INNER_LEFT,
    ),
    // BitXor is its own inverse, so a repeated operand cancels either way.
    (
        "is_bitxor",
        0x08,
        "l_inner_right_canon",
        REPLACE_WITH_LEFT_INNER_LEFT,
    ),
    (
        "is_bitxor",
        0x08,
        "l_inner_left_canon",
        REPLACE_WITH_LEFT_INNER_RIGHT,
    ),
];

/// Literal identities on the operators only this body binds.
const LITERAL_RULES: &[Step] = &[
    // For u32, 0 is the absolute minimum and MAX the absolute maximum.
    literal("is_min", Right, U32(0), REPLACE_WITH_LIT_ZERO), // (Min ?x 0u) → 0u
    literal("is_min", Left, U32(0), REPLACE_WITH_LIT_ZERO),  // (Min 0u ?x) → 0u
    literal("is_min", Right, U32(u32::MAX), REPLACE_WITH_LEFT), // (Min ?x MAX) → ?x
    literal("is_min", Left, U32(u32::MAX), REPLACE_WITH_RIGHT), // (Min MAX ?x) → ?x
    literal("is_max", Right, U32(u32::MAX), REPLACE_WITH_RIGHT), // (Max ?x MAX) → MAX
    literal("is_max", Left, U32(u32::MAX), REPLACE_WITH_LEFT), // (Max MAX ?x) → MAX
    literal("is_max", Right, U32(0), REPLACE_WITH_LEFT),     // (Max ?x 0u) → ?x
    literal("is_max", Left, U32(0), REPLACE_WITH_RIGHT),     // (Max 0u ?x) → ?x
    // Saturating and wrapping zero and one identities.
    literal("is_sat_add", Right, U32(0), REPLACE_WITH_LEFT), // (SaturatingAdd ?x 0) → ?x
    literal("is_sat_add", Left, U32(0), REPLACE_WITH_RIGHT), // (SaturatingAdd 0 ?x) → ?x
    literal("is_sat_sub", Right, U32(0), REPLACE_WITH_LEFT), // (SaturatingSub ?x 0) → ?x
    literal("is_sat_mul", Right, U32(0), REPLACE_WITH_LIT_ZERO), // (SaturatingMul ?x 0) → 0
    literal("is_sat_mul", Left, U32(0), REPLACE_WITH_LIT_ZERO), // (SaturatingMul 0 ?x) → 0
    literal("is_sat_mul", Right, U32(1), REPLACE_WITH_LEFT), // (SaturatingMul ?x 1) → ?x
    literal("is_sat_mul", Left, U32(1), REPLACE_WITH_RIGHT), // (SaturatingMul 1 ?x) → ?x
    literal("is_wrap_add", Right, U32(0), REPLACE_WITH_LEFT), // (WrappingAdd ?x 0) → ?x
    literal("is_wrap_add", Left, U32(0), REPLACE_WITH_RIGHT), // (WrappingAdd 0 ?x) → ?x
    literal("is_wrap_sub", Right, U32(0), REPLACE_WITH_LEFT), // (WrappingSub ?x 0) → ?x
];

/// CSE-aware variant of `bin_op_match_body`.
///
/// The literal-only body comes first, so `l`, `r` and every `is_*` flag it
/// binds are in scope and an equality rule reads them without recomputing the
/// operand. The canonical column then answers whether two operands are one
/// value.
pub(super) fn bin_op_match_body_with_cse() -> Vec<Node> {
    let mut body = bin_op_match_body();
    body.push(Node::let_bind(
        "can_l",
        Expr::load("canonical", Expr::var("l")),
    ));
    body.push(Node::let_bind(
        "can_r",
        Expr::load("canonical", Expr::var("r")),
    ));
    body.push(Node::let_bind(
        "operands_equal",
        Expr::eq(Expr::var("can_l"), Expr::var("can_r")),
    ));
    body.extend(
        OPERATOR_FLAGS
            .iter()
            .map(|(name, tag)| operator_flag(name, *tag)),
    );
    body.extend(
        EQUAL_OPERAND_RULES
            .iter()
            .map(|(flag, action)| equal_operands(Expr::var(*flag), *action)),
    );
    // (Eq ?x ?x), (Le ?x ?x) and (Ge ?x ?x) are all true of one value.
    body.push(equal_operands(
        Expr::or(
            Expr::or(Expr::var("is_cmp_eq"), Expr::var("is_cmp_le")),
            Expr::var("is_cmp_ge"),
        ),
        REPLACE_WITH_LIT_TRUE,
    ));
    // (Ne ?x ?x), (Lt ?x ?x) and (Gt ?x ?x) are all false of one value.
    body.push(equal_operands(
        Expr::or(
            Expr::or(Expr::var("is_cmp_ne"), Expr::var("is_cmp_lt")),
            Expr::var("is_cmp_gt"),
        ),
        REPLACE_WITH_LIT_FALSE,
    ));
    body.extend(left_child_bindings());
    body.extend(
        CANCELLATION_RULES
            .iter()
            .map(|(flag, inner_tag, inner_canonical, action)| {
                left_child_cancellation(flag, *inner_tag, inner_canonical, *action)
            }),
    );
    body.extend(LITERAL_RULES.iter().map(step_node));
    body
}

/// The left child's operator, its two operands, and their canonical values.
///
/// A cancellation rule reads inside the left child, which the literal rules
/// never do, so these bindings belong to this body alone.
fn left_child_bindings() -> Vec<Node> {
    vec![
        Node::let_bind("l_kind_full", Expr::load("arena_kinds", Expr::var("l"))),
        Node::let_bind("l_op", Expr::load("arena_arg0", Expr::var("l"))),
        Node::let_bind("l_inner_left", Expr::load("arena_arg1", Expr::var("l"))),
        Node::let_bind("l_inner_right", Expr::load("arena_arg2", Expr::var("l"))),
        Node::let_bind(
            "l_inner_left_canon",
            Expr::load("canonical", Expr::var("l_inner_left")),
        ),
        Node::let_bind(
            "l_inner_right_canon",
            Expr::load("canonical", Expr::var("l_inner_right")),
        ),
    ]
}
