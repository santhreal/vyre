//! The literal-operand rules, as IR the kernel evaluates per Expr.
//!
//! Each rule reads the encoded operands and writes one action. The set is a
//! table because every one of them is the same shape over a different operator,
//! side and literal; `rule_shapes` owns that shape. The order is the order the
//! kernel evaluates in, and a flag sits above the rules that read it.

use vyre_foundation::ir::{Expr, Node};

use super::rewrite_action::{
    REPLACE_WITH_LEFT, REPLACE_WITH_LIT_FALSE, REPLACE_WITH_LIT_TRUE, REPLACE_WITH_LIT_ZERO,
    REPLACE_WITH_RIGHT,
};
use super::rule_shapes::Literal::{Bool, U32};
use super::rule_shapes::Operand::{Left, Right};
use super::rule_shapes::{flag, literal, operator_flag, step_node, Step};
use crate::optimizer::expr_arena::expr_kind;

/// Operator tags the literal rules gate on, bound before the first rule.
const OPERATOR_FLAGS: &[(&str, u32)] = &[
    ("is_add", 0x01),
    ("is_sub", 0x02),
    ("is_mul", 0x03),
    ("is_bitand", 0x06),
    ("is_bitor", 0x07),
    ("is_bitxor", 0x08),
    ("is_cmp_eq", 0x0B),
    ("is_cmp_ne", 0x0C),
    ("is_cmp_lt", 0x0D),
    ("is_cmp_gt", 0x0E),
    ("is_cmp_le", 0x10),
    ("is_cmp_ge", 0x11),
    ("is_bool_and", 0x12),
    ("is_bool_or", 0x13),
];

/// Whether each operand is a literal of a given type.
const OPERAND_KIND_FLAGS: &[(&str, &str, u32)] = &[
    ("l_is_lit_bool", "l_kind", expr_kind::LIT_BOOL),
    ("r_is_lit_bool", "r_kind", expr_kind::LIT_BOOL),
    ("l_is_lit_u32", "l_kind", expr_kind::LIT_U32),
    ("r_is_lit_u32", "r_kind", expr_kind::LIT_U32),
];

/// Every rewrite that fires on a literal operand alone, in evaluation order.
const LITERAL_RULES: &[Step] = &[
    literal("is_add", Left, U32(0), REPLACE_WITH_RIGHT), // (Add 0 ?x) → ?x
    literal("is_add", Right, U32(0), REPLACE_WITH_LEFT), // (Add ?x 0) → ?x
    literal("is_mul", Left, U32(1), REPLACE_WITH_RIGHT), // (Mul 1 ?x) → ?x
    literal("is_mul", Right, U32(1), REPLACE_WITH_LEFT), // (Mul ?x 1) → ?x
    literal("is_mul", Left, U32(0), REPLACE_WITH_LIT_ZERO), // (Mul 0 ?x) → 0u32
    literal("is_mul", Right, U32(0), REPLACE_WITH_LIT_ZERO), // (Mul ?x 0) → 0u32
    literal("is_sub", Right, U32(0), REPLACE_WITH_LEFT),  // (Sub ?x 0) → ?x
    // A mask of every bit is identity; a mask of none is zero.
    literal("is_bitand", Right, U32(u32::MAX), REPLACE_WITH_LEFT), // (BitAnd ?x MAX) → ?x
    literal("is_bitand", Left, U32(u32::MAX), REPLACE_WITH_RIGHT), // (BitAnd MAX ?x) → ?x
    // Or against every bit saturates, so the result is the literal itself.
    literal("is_bitor", Right, U32(u32::MAX), REPLACE_WITH_RIGHT), // (BitOr ?x MAX) → MAX
    literal("is_bitor", Left, U32(u32::MAX), REPLACE_WITH_LEFT),   // (BitOr MAX ?x) → MAX
    literal("is_bitand", Left, U32(0), REPLACE_WITH_LIT_ZERO),     // (BitAnd 0 ?x) → 0u32
    literal("is_bitand", Right, U32(0), REPLACE_WITH_LIT_ZERO),    // (BitAnd ?x 0) → 0u32
    literal("is_bitor", Left, U32(0), REPLACE_WITH_RIGHT),         // (BitOr 0 ?x) → ?x
    literal("is_bitor", Right, U32(0), REPLACE_WITH_LEFT),         // (BitOr ?x 0) → ?x
    literal("is_bitxor", Left, U32(0), REPLACE_WITH_RIGHT),        // (BitXor 0 ?x) → ?x
    literal("is_bitxor", Right, U32(0), REPLACE_WITH_LEFT),        // (BitXor ?x 0) → ?x
    // Division by one is identity. A zero divisor fires no rule.
    flag("is_div", 0x04),
    literal("is_div", Right, U32(1), REPLACE_WITH_LEFT), // (Div ?x 1) → ?x
    flag("is_mod", 0x05),
    literal("is_mod", Right, U32(1), REPLACE_WITH_LIT_ZERO), // (Mod ?x 1) → 0
    // Shifting by zero keeps the value; shifting zero by anything stays zero.
    flag("is_shl", 0x09),
    flag("is_shr", 0x0A),
    literal("is_shl", Right, U32(0), REPLACE_WITH_LEFT), // (Shl ?x 0) → ?x
    literal("is_shr", Right, U32(0), REPLACE_WITH_LEFT), // (Shr ?x 0) → ?x
    literal("is_shl", Left, U32(0), REPLACE_WITH_LIT_ZERO), // (Shl 0 ?x) → 0
    literal("is_shr", Left, U32(0), REPLACE_WITH_LIT_ZERO), // (Shr 0 ?x) → 0
    // Bool And/Or identities. The arena encodes LitBool(true) as 1.
    literal("is_bool_and", Right, Bool(false), REPLACE_WITH_LIT_FALSE), // (And ?x false) → false
    literal("is_bool_and", Left, Bool(false), REPLACE_WITH_LIT_FALSE),  // (And false ?x) → false
    literal("is_bool_and", Right, Bool(true), REPLACE_WITH_LEFT),       // (And ?x true) → ?x
    literal("is_bool_and", Left, Bool(true), REPLACE_WITH_RIGHT),       // (And true ?x) → ?x
    literal("is_bool_or", Right, Bool(true), REPLACE_WITH_LIT_TRUE),    // (Or ?x true) → true
    literal("is_bool_or", Left, Bool(true), REPLACE_WITH_LIT_TRUE),     // (Or true ?x) → true
    literal("is_bool_or", Right, Bool(false), REPLACE_WITH_LEFT),       // (Or ?x false) → ?x
    literal("is_bool_or", Left, Bool(false), REPLACE_WITH_RIGHT),       // (Or false ?x) → ?x
];

/// Load the operator tag, the child ids, and each child's kind and value.
///
/// Every rule below reads these bindings, and the CSE variant reads them too,
/// so they are bound once at the top of the body.
fn operand_prologue() -> Vec<Node> {
    let mut nodes = vec![
        Node::let_bind("op", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind("l", Expr::load("arena_arg1", Expr::var("i"))),
        Node::let_bind("r", Expr::load("arena_arg2", Expr::var("i"))),
        Node::let_bind("l_kind", Expr::load("arena_kinds", Expr::var("l"))),
        Node::let_bind("r_kind", Expr::load("arena_kinds", Expr::var("r"))),
        Node::let_bind("l_val", Expr::load("arena_arg0", Expr::var("l"))),
        Node::let_bind("r_val", Expr::load("arena_arg0", Expr::var("r"))),
    ];
    nodes.extend(
        OPERATOR_FLAGS
            .iter()
            .map(|(name, tag)| operator_flag(*name, *tag)),
    );
    nodes.extend(OPERAND_KIND_FLAGS.iter().map(|(name, operand_kind, kind)| {
        Node::let_bind(*name, Expr::eq(Expr::var(*operand_kind), Expr::u32(*kind)))
    }));
    nodes
}

/// The per-Expr body for a binary node: the operand bindings, then every rule
/// that fires on a literal operand alone.
pub(super) fn bin_op_match_body() -> Vec<Node> {
    let mut body = operand_prologue();
    body.extend(LITERAL_RULES.iter().map(step_node));
    body
}
