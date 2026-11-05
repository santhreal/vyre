//! The per-Expr decision the kernel writes and the decoder reads.
//!
//! One `u32` per Expr id. The kernel writes it, the host applies it, and
//! nothing else may invent a value: a discriminant added on one side and not
//! the other silently rewrites a program into the wrong shape.

/// No rewrite applies  -  keep the Expr as-is.
pub const NONE: u32 = 0;
/// Replace the Expr with its left child (the operand at `arg1`).
pub const REPLACE_WITH_LEFT: u32 = 1;
/// Replace with the right child (the operand at `arg2`).
pub const REPLACE_WITH_RIGHT: u32 = 2;
/// Replace with `LitU32(0)`.
pub const REPLACE_WITH_LIT_ZERO: u32 = 3;
/// For a `UnOp(op, UnOp(op, x))`: replace with `x` (the
/// grand-child at `arg1->arg1`). Fires for `~~x = x`, `--x = x`,
/// `!!x = x`.
pub const REPLACE_WITH_GRAND_OPERAND: u32 = 4;
/// Replace with `LitBool(true)`. Fires for `x == x`, `x <= x`,
/// `x >= x` after CSE proves the operands are equivalent.
pub const REPLACE_WITH_LIT_TRUE: u32 = 5;
/// Replace with `LitBool(false)`. Fires for `x != x`, `x < x`,
/// `x > x` (irreflexive comparisons of equal operands).
pub const REPLACE_WITH_LIT_FALSE: u32 = 6;
/// For a `BinOp(_, BinOp(_, a, b), _)`: replace with `a`
/// (the outer's left child's left grand-child). Fires for
/// `(Sub (Add a b) b) → a` after CSE confirms operand equality.
pub const REPLACE_WITH_LEFT_INNER_LEFT: u32 = 7;
/// Same as above but pulls the left child's right grand-child.
/// Fires for `(Sub (Add a b) a) → b`.
pub const REPLACE_WITH_LEFT_INNER_RIGHT: u32 = 8;
