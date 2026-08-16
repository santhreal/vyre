//! The analysis programs the kernel is built from.
//!
//! Two shapes: the base program, and the CSE-aware one that carries a
//! `canonical` column so structural-equality rules can fire. Both are the same
//! per-Expr body over a dispatched grid; the rule bodies live beside them.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::bin_op_cse_rules::bin_op_match_body_with_cse;
use super::bin_op_rules::bin_op_match_body;
use super::rewrite_action;
use crate::optimizer::arena_kernel::WORKGROUP_X;
use crate::optimizer::expr_arena::expr_kind;

/// Build a CSE-aware pattern-match Program. Identical to
/// `build_pattern_match_program` but with an extra `canonical` (RO,
/// binding 5) buffer that lets the kernel fire structural-equality
/// rules: `x ^ x → 0`, `x - x → 0`, `x & x → x`, `x | x → x`. These
/// rules only fire when `canonical[arg1] == canonical[arg2]` after
/// CSE. Caller must populate `canonical` by running
/// `gpu_cse_canonicals` first.
#[must_use]
pub fn build_pattern_match_program_with_cse(expr_count: u32) -> Program {
    let mut buffers = crate::optimizer::arena_kernel::arena_row_buffers(expr_count, 0);
    buffers.extend([
        BufferDecl::storage("rewrite_action", 4, BufferAccess::ReadWrite, DataType::U32)
            .with_count(expr_count.max(1)),
        BufferDecl::storage("canonical", 5, BufferAccess::ReadOnly, DataType::U32)
            .with_count(expr_count.max(1)),
    ]);

    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            vec![
                Node::let_bind("kind", Expr::load("arena_kinds", Expr::var("i"))),
                Node::if_then(
                    Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BIN_OP)),
                    bin_op_match_body_with_cse(),
                ),
                Node::if_then(
                    Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::UN_OP)),
                    un_op_match_body(),
                ),
            ],
        ),
    ];

    Program::wrapped(buffers, [WORKGROUP_X, 1, 1], body)
}

/// UnOp double-application matcher. Fires when:
///   `Expr i = UnOp(op, UnOp(op, x))` and `op` is involutive
/// (i.e. `op(op(x)) == x` for all x). Writes
/// `REPLACE_WITH_GRAND_OPERAND` so the rewriter collapses to `x`.
///
/// Restricted to the three truly involutive UnOps: `Negate` (0x01),
/// `BitNot` (0x02), `LogicalNot` (0x03). NOT `Abs`/`Sign`/`Floor`/
/// `Ceil`/`Round`/`Trunc` etc., which are idempotent (`f(f(x)) ==
/// f(x)`) but NOT identity. Folding those to `x` would change
/// behaviour when `x` lies outside the op's range.
fn un_op_match_body() -> Vec<Node> {
    vec![
        Node::let_bind("u_op", Expr::load("arena_arg0", Expr::var("i"))),
        Node::let_bind("u_child", Expr::load("arena_arg1", Expr::var("i"))),
        Node::let_bind(
            "u_child_kind",
            Expr::load("arena_kinds", Expr::var("u_child")),
        ),
        Node::let_bind(
            "u_op_is_involutive",
            Expr::or(
                Expr::or(
                    Expr::eq(Expr::var("u_op"), Expr::u32(0x01)),
                    Expr::eq(Expr::var("u_op"), Expr::u32(0x02)),
                ),
                Expr::eq(Expr::var("u_op"), Expr::u32(0x03)),
            ),
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("u_child_kind"), Expr::u32(expr_kind::UN_OP)),
                Expr::var("u_op_is_involutive"),
            ),
            vec![
                Node::let_bind("u_child_op", Expr::load("arena_arg0", Expr::var("u_child"))),
                Node::if_then(
                    Expr::eq(Expr::var("u_child_op"), Expr::var("u_op")),
                    vec![Node::store(
                        "rewrite_action",
                        Expr::var("i"),
                        Expr::u32(rewrite_action::REPLACE_WITH_GRAND_OPERAND),
                    )],
                ),
            ],
        ),
    ]
}

/// Build the pattern-match analysis Program. Parallel kernel: each
/// GPU thread handles one Expr id via `gid_x()`. The orchestrator
/// dispatches `ceil(expr_count / 256)` workgroups.
pub fn build_pattern_match_program(expr_count: u32) -> Program {
    crate::optimizer::build_encoded_analysis_program(expr_count, "rewrite_action", per_expr_body())
}

fn per_expr_body() -> Vec<Node> {
    vec![
        Node::let_bind("kind", Expr::load("arena_kinds", Expr::var("i"))),
        Node::if_then(
            Expr::eq(Expr::var("kind"), Expr::u32(expr_kind::BIN_OP)),
            bin_op_match_body(),
        ),
    ]
}
