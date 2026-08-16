//! The literal rules plus the rules that need canonical identity.
//!
//! `x ^ x`, `x - x`, `x & x` and `x | x` only fire when both operands are the
//! same value after CSE, which the `canonical` column answers. Every binding
//! the literal rules make is in scope here so an equality rule can read it
//! without recomputing the operand.

use vyre_foundation::ir::{Expr, Node};

use super::rewrite_action;
use crate::optimizer::expr_arena::expr_kind;

/// CSE-aware variant of `bin_op_match_body`. Inlines all the literal
/// rules from the base body, then adds structural-equality rules
/// using the `canonical` buffer. All bindings (l, r, is_*) live in
/// the same scope so the canonical-equality rules can reference them
/// directly.
pub(super) fn bin_op_match_body_with_cse() -> Vec<Node> {
    let mut body = bin_op_match_body();
    // Append CSE-aware rules using the same scope (l, r, is_*
    // already bound in `body`). Fetch canonical[l] / canonical[r]
    // and gate on equality.
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
    // Min/Max/AbsDiff op tag flags  -  needed for the next rule batch.
    body.push(Node::let_bind(
        "is_min",
        Expr::eq(Expr::var("op"), Expr::u32(0x15)),
    ));
    body.push(Node::let_bind(
        "is_max",
        Expr::eq(Expr::var("op"), Expr::u32(0x16)),
    ));
    body.push(Node::let_bind(
        "is_absdiff",
        Expr::eq(Expr::var("op"), Expr::u32(0x14)),
    ));
    body.push(Node::let_bind(
        "is_sat_add",
        Expr::eq(Expr::var("op"), Expr::u32(0x17)),
    ));
    body.push(Node::let_bind(
        "is_sat_sub",
        Expr::eq(Expr::var("op"), Expr::u32(0x18)),
    ));
    body.push(Node::let_bind(
        "is_sat_mul",
        Expr::eq(Expr::var("op"), Expr::u32(0x19)),
    ));
    body.push(Node::let_bind(
        "is_wrap_add",
        Expr::eq(Expr::var("op"), Expr::u32(0x20)),
    ));
    body.push(Node::let_bind(
        "is_wrap_sub",
        Expr::eq(Expr::var("op"), Expr::u32(0x21)),
    ));
    // (SaturatingSub ?x ?x) → 0u
    body.push(Node::if_then(
        Expr::and(Expr::var("is_sat_sub"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
        )],
    ));
    // (WrappingSub ?x ?x) → 0u
    body.push(Node::if_then(
        Expr::and(Expr::var("is_wrap_sub"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
        )],
    ));
    // (Min ?x ?x) → ?x   -  idempotent under operand equality.
    body.push(Node::if_then(
        Expr::and(Expr::var("is_min"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (Max ?x ?x) → ?x   -  idempotent under operand equality.
    body.push(Node::if_then(
        Expr::and(Expr::var("is_max"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (AbsDiff ?x ?x) → 0u32   -  |x - x| = 0.
    body.push(Node::if_then(
        Expr::and(Expr::var("is_absdiff"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
        )],
    ));
    // (Sub ?x ?x) → 0u32
    body.push(Node::if_then(
        Expr::and(Expr::var("is_sub"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
        )],
    ));
    // (BitXor ?x ?x) → 0u32
    body.push(Node::if_then(
        Expr::and(Expr::var("is_bitxor"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_ZERO),
        )],
    ));
    // (BitAnd ?x ?x) → ?x
    body.push(Node::if_then(
        Expr::and(Expr::var("is_bitand"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (BitOr ?x ?x) → ?x
    body.push(Node::if_then(
        Expr::and(Expr::var("is_bitor"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (And ?x ?x) → ?x   -  bool-level idempotency.
    body.push(Node::if_then(
        Expr::and(Expr::var("is_bool_and"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (Or ?x ?x) → ?x   -  bool-level idempotency.
    body.push(Node::if_then(
        Expr::and(Expr::var("is_bool_or"), Expr::var("operands_equal")),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT),
        )],
    ));
    // (Add ?x ?x): no canonical simplification  -  keep as-is
    //   (no rewrite). Skipped here intentionally so other passes
    //   that prefer doubling-as-shift can inspect the pattern.

    // (Eq ?x ?x), (Le ?x ?x), (Ge ?x ?x) → LitBool(true)
    body.push(Node::if_then(
        Expr::and(
            Expr::or(
                Expr::or(Expr::var("is_cmp_eq"), Expr::var("is_cmp_le")),
                Expr::var("is_cmp_ge"),
            ),
            Expr::var("operands_equal"),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_TRUE),
        )],
    ));
    // (Ne ?x ?x), (Lt ?x ?x), (Gt ?x ?x) → LitBool(false)
    body.push(Node::if_then(
        Expr::and(
            Expr::or(
                Expr::or(Expr::var("is_cmp_ne"), Expr::var("is_cmp_lt")),
                Expr::var("is_cmp_gt"),
            ),
            Expr::var("operands_equal"),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LIT_FALSE),
        )],
    ));

    // Sub-Add cancellation: `(Sub (Add a b) c)` where canonical
    // identifies `a == c` or `b == c` collapses to the unmatched
    // operand. Detect by inspecting the left child (`l`)  -  if its
    // kind is BIN_OP with op == 0x01 (Add), and one of its
    // canonical operand-children matches `canonical[r]`, fire.
    body.push(Node::let_bind(
        "l_kind_full",
        Expr::load("arena_kinds", Expr::var("l")),
    ));
    body.push(Node::let_bind(
        "l_op",
        Expr::load("arena_arg0", Expr::var("l")),
    ));
    body.push(Node::let_bind(
        "l_inner_left",
        Expr::load("arena_arg1", Expr::var("l")),
    ));
    body.push(Node::let_bind(
        "l_inner_right",
        Expr::load("arena_arg2", Expr::var("l")),
    ));
    body.push(Node::let_bind(
        "l_inner_left_canon",
        Expr::load("canonical", Expr::var("l_inner_left")),
    ));
    body.push(Node::let_bind(
        "l_inner_right_canon",
        Expr::load("canonical", Expr::var("l_inner_right")),
    ));
    // `(Sub (Add a b) c)` and canonical[a] == canonical[c]: take b
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sub"),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    Expr::eq(Expr::var("l_op"), Expr::u32(0x01)),
                    Expr::eq(Expr::var("l_inner_left_canon"), Expr::var("can_r")),
                ),
            ),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT_INNER_RIGHT),
        )],
    ));
    // `(Sub (Add a b) c)` and canonical[b] == canonical[c]: take a
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sub"),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    Expr::eq(Expr::var("l_op"), Expr::u32(0x01)),
                    Expr::eq(Expr::var("l_inner_right_canon"), Expr::var("can_r")),
                ),
            ),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT_INNER_LEFT),
        )],
    ));

    // Add-Sub cancellation: `(Add (Sub a b) b) → a` and the
    // commutative variant `(Add (Sub a b) b)` after canon. Op tag
    // for Sub is 0x02; the left's op must be Sub for this rule.
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_add"),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    // Left's op == Sub (0x02)
                    Expr::eq(Expr::var("l_op"), Expr::u32(0x02)),
                    // canonical[(left's right operand b)] == canonical[r]
                    Expr::eq(Expr::var("l_inner_right_canon"), Expr::var("can_r")),
                ),
            ),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT_INNER_LEFT),
        )],
    ));

    // BitXor self-cancellation through a chain: BitXor is its own
    // inverse, so `(BitXor (BitXor a b) b) → a` and the symmetric
    // `(BitXor (BitXor a b) a) → b`. Op tag for BitXor is 0x08.
    // `(BitXor (BitXor a b) c)` and canonical[b] == canonical[c]: take a
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_bitxor"),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    Expr::eq(Expr::var("l_op"), Expr::u32(0x08)),
                    Expr::eq(Expr::var("l_inner_right_canon"), Expr::var("can_r")),
                ),
            ),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT_INNER_LEFT),
        )],
    ));
    // `(BitXor (BitXor a b) c)` and canonical[a] == canonical[c]: take b
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_bitxor"),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    Expr::eq(Expr::var("l_op"), Expr::u32(0x08)),
                    Expr::eq(Expr::var("l_inner_left_canon"), Expr::var("can_r")),
                ),
            ),
        ),
        vec![Node::store(
            "rewrite_action",
            Expr::var("i"),
            Expr::u32(rewrite_action::REPLACE_WITH_LEFT_INNER_RIGHT),
        )],
    ));

    // Min/Max literal-identity rules. For u32: 0 is the absolute
    // minimum (any value is >= 0), MAX is the absolute maximum.
    // (Min ?x 0u) → 0u
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_min"),
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
    ));
    // (Min 0u ?x) → 0u
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_min"),
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
    ));
    // (Min ?x MAX) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_min"),
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
    ));
    // (Min MAX ?x) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_min"),
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
    ));
    // (Max ?x MAX) → MAX  (replace with right; right IS the literal MAX)
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_max"),
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
    ));
    // (Max MAX ?x) → MAX
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_max"),
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
    ));
    // (Max ?x 0u) → ?x  (max with 0 is identity for u32)
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_max"),
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
    ));
    // (Max 0u ?x) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_max"),
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
    ));
    // Saturating/Wrapping zero/one identities (literal cases).
    // (SaturatingAdd ?x 0) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_add"),
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
    ));
    // (SaturatingAdd 0 ?x) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_add"),
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
    ));
    // (SaturatingSub ?x 0) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_sub"),
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
    ));
    // (SaturatingMul ?x 0) → 0
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_mul"),
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
    ));
    // (SaturatingMul 0 ?x) → 0
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_mul"),
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
    ));
    // (SaturatingMul ?x 1) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_mul"),
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
    ));
    // (SaturatingMul 1 ?x) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_sat_mul"),
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
    ));
    // (WrappingAdd ?x 0) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_wrap_add"),
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
    ));
    // (WrappingAdd 0 ?x) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_wrap_add"),
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
    ));
    // (WrappingSub ?x 0) → ?x
    body.push(Node::if_then(
        Expr::and(
            Expr::var("is_wrap_sub"),
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
    ));
    body
}
