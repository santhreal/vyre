use crate::parsing::composition::child_phase;
use emit::{binary_token_body, emit_value_leaf, final_sweep_body, rparen_body};
use operator::{is_assignment_token, is_value_token, precedence};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::c11_token::*;

mod emit;
mod operator;
mod shunting_witness;

const OP_ID: &str = "vyre-libs::parsing::ast_shunting_yard";
// Phase boundary inside the one operation, not an operation of its own. The
// `anonymous::` prefix is what says so: see
// `vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES`.
const STATEMENT_PASS_GENERATOR: &str = "anonymous::shunting_yard_statement_pass";
const MAX_TOK_SCAN: u32 = 65_536;
const STACK_SLOTS_PER_STATEMENT: u32 = 64;

/// Data-parallel shunting-yard AST builder.
///
/// Each invocation owns one statement boundary and emits a flat node stream
/// where every AST node is four `u32` words: `(opcode, left, right, value_ref)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ast_shunting_yard(
    tok_types: &str,
    statements: &str,
    num_statements: Expr,
    out_ast_nodes: &str,
    out_ast_count: &str,
    out_statement_roots: &str,
    scratch_val_stack: &str,
    scratch_op_stack: &str,
) -> Program {
    ast_shunting_yard_program(
        tok_types,
        statements,
        num_statements,
        out_ast_nodes,
        out_ast_count,
        out_statement_roots,
        scratch_val_stack,
        scratch_op_stack,
        MAX_TOK_SCAN,
        None,
    )
}

/// Data-parallel shunting-yard AST builder with caller-bounded capacities.
///
/// This preserves [`ast_shunting_yard`]'s semantics while avoiding the
/// release-path cost of uploading and allocating fixed 65k-token buffers for
/// small translation units. `token_capacity` sizes the four-word AST-node
/// stream, while `statement_capacity` sizes the per-statement root table.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ast_shunting_yard_with_capacity(
    tok_types: &str,
    statements: &str,
    num_statements: Expr,
    out_ast_nodes: &str,
    out_ast_count: &str,
    out_statement_roots: &str,
    scratch_val_stack: &str,
    scratch_op_stack: &str,
    token_capacity: u32,
    statement_capacity: u32,
) -> Program {
    ast_shunting_yard_program(
        tok_types,
        statements,
        num_statements,
        out_ast_nodes,
        out_ast_count,
        out_statement_roots,
        scratch_val_stack,
        scratch_op_stack,
        token_capacity,
        Some(statement_capacity),
    )
}

#[allow(clippy::too_many_arguments)]
fn ast_shunting_yard_program(
    tok_types: &str,
    statements: &str,
    num_statements: Expr,
    out_ast_nodes: &str,
    out_ast_count: &str,
    out_statement_roots: &str,
    scratch_val_stack: &str,
    scratch_op_stack: &str,
    token_capacity: u32,
    statement_capacity: Option<u32>,
) -> Program {
    let token_capacity = token_capacity.clamp(1, MAX_TOK_SCAN);
    let statement_capacity = statement_capacity.map(|capacity| capacity.clamp(1, MAX_TOK_SCAN));
    let t = Expr::InvocationId { axis: 0 };
    let val_stack_base = Expr::mul(t.clone(), Expr::u32(STACK_SLOTS_PER_STATEMENT));
    let op_stack_base = Expr::mul(t.clone(), Expr::u32(STACK_SLOTS_PER_STATEMENT));

    let loop_body = vec![
        Node::let_bind(
            "stmt_start",
            Expr::load(statements, Expr::mul(t.clone(), Expr::u32(2))),
        ),
        Node::let_bind(
            "stmt_end",
            Expr::load(
                statements,
                Expr::add(Expr::mul(t.clone(), Expr::u32(2)), Expr::u32(1)),
            ),
        ),
        Node::let_bind("v_sp", Expr::u32(0)),
        Node::let_bind("o_sp", Expr::u32(0)),
        Node::loop_for(
            "tok_idx",
            Expr::var("stmt_start"),
            Expr::var("stmt_end"),
            vec![
                Node::let_bind("tok", Expr::load(tok_types, Expr::var("tok_idx"))),
                Node::let_bind("tok_prec", precedence(Expr::var("tok"))),
                Node::let_bind("tok_is_assignment", is_assignment_token(Expr::var("tok"))),
                Node::if_then(
                    is_value_token(Expr::var("tok")),
                    emit_value_leaf(
                        out_ast_nodes,
                        out_ast_count,
                        scratch_val_stack,
                        val_stack_base.clone(),
                    ),
                ),
                Node::if_then(
                    Expr::ne(Expr::var("tok_prec"), Expr::u32(0)),
                    binary_token_body(
                        scratch_op_stack,
                        out_ast_nodes,
                        out_ast_count,
                        scratch_val_stack,
                        val_stack_base.clone(),
                        op_stack_base.clone(),
                    ),
                ),
                Node::if_then(
                    Expr::eq(Expr::var("tok"), Expr::u32(TOK_LPAREN)),
                    vec![
                        Node::store(
                            scratch_op_stack,
                            Expr::add(op_stack_base.clone(), Expr::var("o_sp")),
                            Expr::var("tok"),
                        ),
                        Node::assign("o_sp", Expr::add(Expr::var("o_sp"), Expr::u32(1))),
                    ],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("tok"), Expr::u32(TOK_RPAREN)),
                    rparen_body(
                        scratch_op_stack,
                        out_ast_nodes,
                        out_ast_count,
                        scratch_val_stack,
                        val_stack_base.clone(),
                        op_stack_base.clone(),
                    ),
                ),
            ],
        ),
        Node::Block(final_sweep_body(
            scratch_op_stack,
            out_ast_nodes,
            out_ast_count,
            scratch_val_stack,
            val_stack_base.clone(),
            op_stack_base,
        )),
        Node::if_then(
            Expr::gt(Expr::var("v_sp"), Expr::u32(0)),
            vec![Node::store(
                out_statement_roots,
                t.clone(),
                Expr::load(
                    scratch_val_stack,
                    Expr::add(val_stack_base, Expr::sub(Expr::var("v_sp"), Expr::u32(1))),
                ),
            )],
        ),
        Node::if_then(
            Expr::eq(Expr::var("v_sp"), Expr::u32(0)),
            vec![Node::store(
                out_statement_roots,
                t.clone(),
                Expr::u32(u32::MAX),
            )],
        ),
    ];

    let statement_limit = statement_capacity
        .map(Expr::u32)
        .unwrap_or_else(|| num_statements.clone());
    let out_statement_roots_decl = {
        let decl = BufferDecl::storage(
            out_statement_roots,
            4,
            BufferAccess::ReadWrite,
            DataType::U32,
        );
        if let Some(statement_capacity) = statement_capacity {
            decl.with_count(statement_capacity)
        } else {
            decl
        }
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(statements, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_ast_nodes, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(token_capacity.saturating_mul(4)),
            BufferDecl::storage(out_ast_count, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            out_statement_roots_decl,
            BufferDecl::storage(scratch_val_stack, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(
                    statement_capacity
                        .unwrap_or(MAX_TOK_SCAN)
                        .saturating_mul(STACK_SLOTS_PER_STATEMENT),
                ),
            BufferDecl::storage(scratch_op_stack, 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(
                    statement_capacity
                        .unwrap_or(MAX_TOK_SCAN)
                        .saturating_mul(STACK_SLOTS_PER_STATEMENT),
                ),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(
                    t.clone(),
                    Expr::min(
                        Expr::div(Expr::buf_len(statements), Expr::u32(2)),
                        statement_limit,
                    ),
                ),
                vec![child_phase(OP_ID, STATEMENT_PASS_GENERATOR, loop_body)],
            )],
        )],
    )
    .with_entry_op_id(OP_ID)
    .with_non_composable_with_self(true)
}
