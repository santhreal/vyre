//! The one registration and fixture set for the shunting-yard AST builder.
//!
//! This file used to carry a second `inventory::submit!` for `OP_ID` that was
//! never compiled: `shunting.rs` declared only `emit` and `operator`, so the
//! module was orphaned and its copy of the fixtures drifted. It built the
//! unbounded `ast_shunting_yard` instead of the capacity-bounded builder and
//! expected empty scratch buffers back. The compiled registration in
//! `shunting.rs` is the one kept here.

use crate::parsing::c::lex::tokens::TOK_IDENTIFIER;
use crate::parsing::core::ast::node::AST_VAR;
use vyre_foundation::ir::Expr;
use vyre_primitives::wire::pack_u32_slice as pack_u32;

use super::{ast_shunting_yard_with_capacity, MAX_TOK_SCAN, OP_ID};

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || ast_shunting_yard_with_capacity(
            "tok_types", "statements", Expr::u32(100),
            "out_ast_nodes", "out_ast_count", "out_statement_roots",
            "scratch_val_stack", "scratch_op_stack",
            MAX_TOK_SCAN, 100
        ),
        Some(|| vec![vec![
            shunting_token_fixture(),
            shunting_statement_fixture(),
            vec![0u8; MAX_TOK_SCAN as usize * 4 * 4],
            vec![0u8; 4],
            vec![0u8; 100 * 4],
            vec![0u8; 6_400 * 4],
            vec![0u8; 6_400 * 4],
        ]]),
        Some(shunting_expected_output),
    )
    .with_category("parsing")
}

fn shunting_token_fixture() -> Vec<u8> {
    let mut tokens = vec![0u32; MAX_TOK_SCAN as usize];
    tokens[0] = TOK_IDENTIFIER;
    pack_u32(&tokens)
}

fn shunting_statement_fixture() -> Vec<u8> {
    let mut statements = vec![0u32; 200];
    statements[1] = 1;
    pack_u32(&statements)
}

fn shunting_expected_output() -> Vec<Vec<Vec<u8>>> {
    let mut ast_nodes = vec![0u32; MAX_TOK_SCAN as usize * 4];
    ast_nodes[0..4].copy_from_slice(&[AST_VAR, u32::MAX, u32::MAX, 0]);
    let mut roots = vec![u32::MAX; 100];
    roots[0] = 0;
    vec![vec![
        pack_u32(&ast_nodes),
        pack_u32(&[4]),
        pack_u32(&roots),
        vec![0u8; 6_400 * 4],
        vec![0u8; 6_400 * 4],
    ]]
}
