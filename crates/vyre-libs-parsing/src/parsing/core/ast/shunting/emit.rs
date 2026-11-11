//! The node bodies the shunting-yard reducer emits: value leaves, binary
//! reductions, closing parentheses, and the final sweep.

use crate::parsing::core::ast::node::{AST_CONST_INT, AST_VAR};
use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::c11_token::{TOK_INTEGER, TOK_LPAREN};

use super::operator::{ast_opcode, precedence, should_pop_cached};
use super::{AST_SHUNTING_YARD_REDUCE_OP_ID, OP_ID, STACK_SLOTS_PER_STATEMENT};

pub(super) fn emit_value_leaf(
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind(
            "ast_idx",
            Expr::atomic_add(out_ast_count, Expr::u32(0), Expr::u32(4)),
        ),
        Node::let_bind(
            "opcode",
            Expr::select(
                Expr::eq(Expr::var("tok"), Expr::u32(TOK_INTEGER)),
                Expr::u32(AST_CONST_INT),
                Expr::u32(AST_VAR),
            ),
        ),
        Node::store(out_ast_nodes, Expr::var("ast_idx"), Expr::var("opcode")),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(1)),
            Expr::u32(u32::MAX),
        ),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(2)),
            Expr::u32(u32::MAX),
        ),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(3)),
            Expr::var("tok_idx"),
        ),
        Node::store(
            scratch_val_stack,
            Expr::add(val_stack_base, Expr::var("v_sp")),
            Expr::var("ast_idx"),
        ),
        Node::assign("v_sp", Expr::add(Expr::var("v_sp"), Expr::u32(1))),
    ]
}

fn reduce_loaded_operator(
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    opcode: Expr,
) -> Vec<Node> {
    vec![
        Node::assign("v_sp", Expr::sub(Expr::var("v_sp"), Expr::u32(1))),
        Node::let_bind(
            "right_child",
            Expr::load(
                scratch_val_stack,
                Expr::add(val_stack_base.clone(), Expr::var("v_sp")),
            ),
        ),
        Node::assign("v_sp", Expr::sub(Expr::var("v_sp"), Expr::u32(1))),
        Node::let_bind(
            "left_child",
            Expr::load(
                scratch_val_stack,
                Expr::add(val_stack_base.clone(), Expr::var("v_sp")),
            ),
        ),
        Node::let_bind(
            "ast_idx",
            Expr::atomic_add(out_ast_count, Expr::u32(0), Expr::u32(4)),
        ),
        Node::store(out_ast_nodes, Expr::var("ast_idx"), opcode),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(1)),
            Expr::var("left_child"),
        ),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(2)),
            Expr::var("right_child"),
        ),
        Node::store(
            out_ast_nodes,
            Expr::add(Expr::var("ast_idx"), Expr::u32(3)),
            Expr::u32(u32::MAX),
        ),
        Node::store(
            scratch_val_stack,
            Expr::add(val_stack_base, Expr::var("v_sp")),
            Expr::var("ast_idx"),
        ),
        Node::assign("v_sp", Expr::add(Expr::var("v_sp"), Expr::u32(1))),
    ]
}

fn reduce_if_allowed(
    scratch_op_stack: &str,
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    op_stack_base: Expr,
) -> Vec<Node> {
    let mut body = vec![Node::assign(
        "o_sp",
        Expr::sub(Expr::var("o_sp"), Expr::u32(1)),
    )];
    body.extend(reduce_loaded_operator(
        out_ast_nodes,
        out_ast_count,
        scratch_val_stack,
        val_stack_base,
        Expr::var("top_ast_opcode"),
    ));

    vec![
        Node::let_bind(
            "top_op",
            Expr::load(
                scratch_op_stack,
                Expr::add(op_stack_base, Expr::sub(Expr::var("o_sp"), Expr::u32(1))),
            ),
        ),
        Node::let_bind("top_op_prec", precedence(Expr::var("top_op"))),
        Node::let_bind("top_ast_opcode", ast_opcode(Expr::var("top_op"))),
        Node::let_bind(
            "reduce_now",
            Expr::and(
                should_pop_cached(
                    Expr::var("top_op"),
                    Expr::var("top_op_prec"),
                    Expr::var("tok_prec"),
                    Expr::var("tok_is_assignment"),
                ),
                Expr::ge(Expr::var("v_sp"), Expr::u32(2)),
            ),
        ),
        Node::if_then(Expr::var("reduce_now"), body),
        Node::if_then(
            Expr::not(Expr::var("reduce_now")),
            vec![Node::assign("done_bin", Expr::u32(1))],
        ),
    ]
}

pub(super) fn binary_token_body(
    scratch_op_stack: &str,
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    op_stack_base: Expr,
) -> Vec<Node> {
    let reduce_one = reduce_if_allowed(
        scratch_op_stack,
        out_ast_nodes,
        out_ast_count,
        scratch_val_stack,
        val_stack_base,
        op_stack_base.clone(),
    );

    let reduce_loop = vec![
        Node::let_bind("done_bin", Expr::u32(0)),
        Node::loop_for(
            "pop",
            Expr::u32(0),
            Expr::u32(STACK_SLOTS_PER_STATEMENT),
            vec![Node::if_then(
                Expr::eq(Expr::var("done_bin"), Expr::u32(0)),
                vec![
                    Node::if_then(
                        Expr::eq(Expr::var("o_sp"), Expr::u32(0)),
                        vec![Node::assign("done_bin", Expr::u32(1))],
                    ),
                    Node::if_then(Expr::ne(Expr::var("o_sp"), Expr::u32(0)), reduce_one),
                ],
            )],
        ),
    ];

    vec![
        wrap_child_region(
            AST_SHUNTING_YARD_REDUCE_OP_ID,
            Ident::from(OP_ID),
            reduce_loop,
        ),
        Node::store(
            scratch_op_stack,
            Expr::add(op_stack_base, Expr::var("o_sp")),
            Expr::var("tok"),
        ),
        Node::assign("o_sp", Expr::add(Expr::var("o_sp"), Expr::u32(1))),
    ]
}

fn operator_sweep_body(
    scratch_op_stack: &str,
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    op_stack_base: Expr,
    done_name: &str,
    stop_at_lparen: bool,
) -> Vec<Node> {
    let mut pop_body = vec![
        Node::assign("o_sp", Expr::sub(Expr::var("o_sp"), Expr::u32(1))),
        Node::let_bind(
            "top_op",
            Expr::load(
                scratch_op_stack,
                Expr::add(op_stack_base, Expr::var("o_sp")),
            ),
        ),
        Node::let_bind("top_ast_opcode", ast_opcode(Expr::var("top_op"))),
    ];
    if stop_at_lparen {
        pop_body.push(Node::if_then(
            Expr::eq(Expr::var("top_op"), Expr::u32(TOK_LPAREN)),
            vec![Node::assign(done_name, Expr::u32(1))],
        ));
    }
    pop_body.push(Node::if_then(
        Expr::and(
            Expr::ne(Expr::var("top_op"), Expr::u32(TOK_LPAREN)),
            Expr::ge(Expr::var("v_sp"), Expr::u32(2)),
        ),
        reduce_loaded_operator(
            out_ast_nodes,
            out_ast_count,
            scratch_val_stack,
            val_stack_base,
            Expr::var("top_ast_opcode"),
        ),
    ));
    let sweep_loop = vec![
        Node::let_bind(done_name, Expr::u32(0)),
        Node::loop_for(
            "pop",
            Expr::u32(0),
            Expr::u32(STACK_SLOTS_PER_STATEMENT),
            vec![Node::if_then(
                Expr::eq(Expr::var(done_name), Expr::u32(0)),
                vec![
                    Node::if_then(
                        Expr::eq(Expr::var("o_sp"), Expr::u32(0)),
                        vec![Node::assign(done_name, Expr::u32(1))],
                    ),
                    Node::if_then(Expr::ne(Expr::var("o_sp"), Expr::u32(0)), pop_body),
                ],
            )],
        ),
    ];
    vec![wrap_child_region(
        AST_SHUNTING_YARD_REDUCE_OP_ID,
        Ident::from(OP_ID),
        sweep_loop,
    )]
}
pub(super) fn rparen_body(
    scratch_op_stack: &str,
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    op_stack_base: Expr,
) -> Vec<Node> {
    operator_sweep_body(
        scratch_op_stack,
        out_ast_nodes,
        out_ast_count,
        scratch_val_stack,
        val_stack_base,
        op_stack_base,
        "done_rp",
        true,
    )
}

pub(super) fn final_sweep_body(
    scratch_op_stack: &str,
    out_ast_nodes: &str,
    out_ast_count: &str,
    scratch_val_stack: &str,
    val_stack_base: Expr,
    op_stack_base: Expr,
) -> Vec<Node> {
    operator_sweep_body(
        scratch_op_stack,
        out_ast_nodes,
        out_ast_count,
        scratch_val_stack,
        val_stack_base,
        op_stack_base,
        "done_fs",
        false,
    )
}

/// Build the standalone operator reduction sub-operation.
#[must_use]
pub fn ast_shunting_yard_reduce_program() -> Program {
    let out_ast_nodes = "out_ast_nodes";
    let out_ast_count = "out_ast_count";
    let scratch_val_stack = "scratch_val_stack";
    let scratch_op_stack = "scratch_op_stack";
    let mut body = vec![
        Node::let_bind("o_sp", Expr::u32(1)),
        Node::let_bind("v_sp", Expr::u32(2)),
        Node::let_bind("tok", Expr::u32(vyre_spec::c11_token::TOK_PLUS)),
        Node::let_bind("tok_prec", precedence(Expr::var("tok"))),
        Node::let_bind("tok_is_assignment", Expr::bool(false)),
    ];
    body.extend(binary_token_body(
        scratch_op_stack,
        out_ast_nodes,
        out_ast_count,
        scratch_val_stack,
        Expr::u32(0),
        Expr::u32(0),
    ));
    let guarded = vec![Node::if_then(
        Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
        body,
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(scratch_op_stack, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(STACK_SLOTS_PER_STATEMENT),
            BufferDecl::storage(scratch_val_stack, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(STACK_SLOTS_PER_STATEMENT),
            BufferDecl::storage(out_ast_nodes, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(64),
            BufferDecl::storage(out_ast_count, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            AST_SHUNTING_YARD_REDUCE_OP_ID,
            guarded,
        )],
    )
}
