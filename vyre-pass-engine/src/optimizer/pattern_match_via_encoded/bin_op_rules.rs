//! The literal-operand rules, as IR the kernel evaluates per Expr.
//!
//! Each rule reads the encoded operands and writes one action. They are written
//! out rather than looped because the kernel has no pattern bank yet; the shape
//! is the fixed instance of the general engine described in the module above.

use vyre_foundation::ir::{Expr, Node};

use super::rewrite_action;
use crate::optimizer::expr_arena::expr_kind;

pub(super) fn bin_op_match_body() -> Vec<Node> {
    // Look up op tag, child ids, and child kind+value.
    vec![
        Node::let_bind("op", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind("l", Expr::load("arena_arg1", Expr::var("i"))),
        Node::let_bind("r", Expr::load("arena_arg2", Expr::var("i"))),
        Node::let_bind("l_kind", Expr::load("arena_kinds", Expr::var("l"))),
        Node::let_bind("r_kind", Expr::load("arena_kinds", Expr::var("r"))),
        Node::let_bind("l_val", Expr::load("arena_arg0", Expr::var("l"))),
        Node::let_bind("r_val", Expr::load("arena_arg0", Expr::var("r"))),
        // Op tags: Add=0x01, Sub=0x02, Mul=0x03, BitAnd=0x06,
        // BitOr=0x07, BitXor=0x08, Eq=0x0B, Ne=0x0C, Lt=0x0D,
        // Gt=0x0E, Le=0x10, Ge=0x11, And=0x12, Or=0x13.
        Node::let_bind("is_add", Expr::eq(Expr::var("op"), Expr::u32(0x01))),
        Node::let_bind("is_sub", Expr::eq(Expr::var("op"), Expr::u32(0x02))),
        Node::let_bind("is_mul", Expr::eq(Expr::var("op"), Expr::u32(0x03))),
        Node::let_bind("is_bitand", Expr::eq(Expr::var("op"), Expr::u32(0x06))),
        Node::let_bind("is_bitor", Expr::eq(Expr::var("op"), Expr::u32(0x07))),
        Node::let_bind("is_bitxor", Expr::eq(Expr::var("op"), Expr::u32(0x08))),
        Node::let_bind("is_cmp_eq", Expr::eq(Expr::var("op"), Expr::u32(0x0B))),
        Node::let_bind("is_cmp_ne", Expr::eq(Expr::var("op"), Expr::u32(0x0C))),
        Node::let_bind("is_cmp_lt", Expr::eq(Expr::var("op"), Expr::u32(0x0D))),
        Node::let_bind("is_cmp_gt", Expr::eq(Expr::var("op"), Expr::u32(0x0E))),
        Node::let_bind("is_cmp_le", Expr::eq(Expr::var("op"), Expr::u32(0x10))),
        Node::let_bind("is_cmp_ge", Expr::eq(Expr::var("op"), Expr::u32(0x11))),
        Node::let_bind("is_bool_and", Expr::eq(Expr::var("op"), Expr::u32(0x12))),
        Node::let_bind("is_bool_or", Expr::eq(Expr::var("op"), Expr::u32(0x13))),
        Node::let_bind(
            "l_is_lit_bool",
            Expr::eq(Expr::var("l_kind"), Expr::u32(expr_kind::LIT_BOOL)),
        ),
        Node::let_bind(
            "r_is_lit_bool",
            Expr::eq(Expr::var("r_kind"), Expr::u32(expr_kind::LIT_BOOL)),
        ),
        Node::let_bind(
            "l_is_lit_u32",
            Expr::eq(Expr::var("l_kind"), Expr::u32(expr_kind::LIT_U32)),
        ),
        Node::let_bind(
            "r_is_lit_u32",
            Expr::eq(Expr::var("r_kind"), Expr::u32(expr_kind::LIT_U32)),
        ),
        // (Add 0 ?x) → ?x   (left child is LitU32(0); replace with right)
        Node::if_then(
            Expr::and(
                Expr::var("is_add"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (Add ?x 0) → ?x   (right child is LitU32(0); replace with left)
        Node::if_then(
            Expr::and(
                Expr::var("is_add"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Mul 1 ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_mul"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (Mul ?x 1) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_mul"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Mul 0 ?x) → 0u32
        Node::if_then(
            Expr::and(
                Expr::var("is_mul"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // (Mul ?x 0) → 0u32
        Node::if_then(
            Expr::and(
                Expr::var("is_mul"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // (Sub ?x 0) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_sub"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (BitAnd ?x MAX) → ?x   (mask-everything And is identity)
        Node::if_then(
            Expr::and(
                Expr::var("is_bitand"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(u32::MAX)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (BitAnd MAX ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bitand"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(u32::MAX)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (BitOr ?x MAX) → MAX  (saturated). Replace with right
        // (the literal MAX itself).
        Node::if_then(
            Expr::and(
                Expr::var("is_bitor"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(u32::MAX)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (BitOr MAX ?x) → MAX  (saturated). Replace with left.
        Node::if_then(
            Expr::and(
                Expr::var("is_bitor"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(u32::MAX)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (BitAnd 0 ?x) → 0u32
        Node::if_then(
            Expr::and(
                Expr::var("is_bitand"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // (BitAnd ?x 0) → 0u32
        Node::if_then(
            Expr::and(
                Expr::var("is_bitand"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // (BitOr 0 ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bitor"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (BitOr ?x 0) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bitor"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (BitXor 0 ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bitxor"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (BitXor ?x 0) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bitxor"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Div ?x 1) → ?x   -  division by 1 is identity. Op tag for
        // Div is 0x04. Reject divisor zero (no rule fires there).
        Node::let_bind("is_div", Expr::eq(Expr::var("op"), Expr::u32(0x04))),
        Node::if_then(
            Expr::and(
                Expr::var("is_div"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Mod ?x 1) → 0   -  modulo 1 is always zero. Op tag 0x05.
        Node::let_bind("is_mod", Expr::eq(Expr::var("op"), Expr::u32(0x05))),
        Node::if_then(
            Expr::and(
                Expr::var("is_mod"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // Shl=0x09, Shr=0x0A. Shift-by-zero keeps the value; shift
        // of zero is always zero (any positive shift count).
        Node::let_bind("is_shl", Expr::eq(Expr::var("op"), Expr::u32(0x09))),
        Node::let_bind("is_shr", Expr::eq(Expr::var("op"), Expr::u32(0x0A))),
        // (Shl ?x 0) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_shl"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Shr ?x 0) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_shr"),
                Expr::and(
                    Expr::var("r_is_lit_u32"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Shl 0 ?x) → 0  (zero left-shifted by anything stays 0)
        Node::if_then(
            Expr::and(
                Expr::var("is_shl"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // (Shr 0 ?x) → 0  (zero right-shifted is still 0)
        Node::if_then(
            Expr::and(
                Expr::var("is_shr"),
                Expr::and(
                    Expr::var("l_is_lit_u32"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
            )],
        ),
        // Bool And/Or identity rules. LitBool(true) is encoded as
        // arg0=1; LitBool(false) as arg0=0 in the arena.
        // (And ?x false) → false
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_and"),
                Expr::and(
                    Expr::var("r_is_lit_bool"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_FALSE),
            )],
        ),
        // (And false ?x) → false
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_and"),
                Expr::and(
                    Expr::var("l_is_lit_bool"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_FALSE),
            )],
        ),
        // (And ?x true) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_and"),
                Expr::and(
                    Expr::var("r_is_lit_bool"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (And true ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_and"),
                Expr::and(
                    Expr::var("l_is_lit_bool"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
        // (Or ?x true) → true
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_or"),
                Expr::and(
                    Expr::var("r_is_lit_bool"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_TRUE),
            )],
        ),
        // (Or true ?x) → true
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_or"),
                Expr::and(
                    Expr::var("l_is_lit_bool"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(1)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LIT_TRUE),
            )],
        ),
        // (Or ?x false) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_or"),
                Expr::and(
                    Expr::var("r_is_lit_bool"),
                    Expr::eq(Expr::var("r_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
            )],
        ),
        // (Or false ?x) → ?x
        Node::if_then(
            Expr::and(
                Expr::var("is_bool_or"),
                Expr::and(
                    Expr::var("l_is_lit_bool"),
                    Expr::eq(Expr::var("l_val"), Expr::u32(0)),
                ),
            ),
            vec![Node::store(
                "rewrite_action",
                Expr::var("i"),
                Expr::u32(rewrite_action::REPLACE_WITH_RIGHT),
            )],
        ),
    ]
}
