//! The one registration and fixture set for the shunting-yard AST builder.
//!
//! This file used to carry a second `inventory::submit!` for `OP_ID` that was
//! never compiled: `shunting.rs` declared only `emit` and `operator`, so the
//! module was orphaned and its copy of the fixtures drifted. It built the
//! unbounded `ast_shunting_yard` instead of the capacity-bounded builder and
//! expected empty scratch buffers back. The compiled registration in
//! `shunting.rs` is the one kept here.

use vyre_foundation::ir::Expr;
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_spec::c11_token::TOK_IDENTIFIER;

use super::{ast_shunting_yard_with_capacity, MAX_TOK_SCAN, OP_ID};
const AST_NODES_BYTE_LEN: usize = (MAX_TOK_SCAN as usize) * 16;
static EXPECTED_SHUNTING_AST_NODES_BYTES: [u8; AST_NODES_BYTE_LEN] = {
    let mut arr = [0u8; AST_NODES_BYTE_LEN];
    arr[0] = 2;
    arr[4] = 255;
    arr[5] = 255;
    arr[6] = 255;
    arr[7] = 255;
    arr[8] = 255;
    arr[9] = 255;
    arr[10] = 255;
    arr[11] = 255;
    arr
};
const EXPECTED_SHUNTING_COUNT_BYTES: [u8; 4] = [4, 0, 0, 0];
static EXPECTED_SHUNTING_ROOTS_BYTES: [u8; 400] = {
    let mut arr = [255u8; 400];
    arr[0] = 0;
    arr[1] = 0;
    arr[2] = 0;
    arr[3] = 0;
    arr
};
static EXPECTED_SHUNTING_SCRATCH_VAL_STACK_BYTES: [u8; 25_600] = [0u8; 25_600];
static EXPECTED_SHUNTING_SCRATCH_OP_STACK_BYTES: [u8; 25_600] = [0u8; 25_600];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || ast_shunting_yard_with_capacity(
            "tok_types", "statements", Expr::u32(100),
            "out_ast_nodes", "out_ast_count", "out_statement_roots",
            "scratch_val_stack", "scratch_op_stack",
            MAX_TOK_SCAN, 100
        ),
        Some(|| {
            vec![vec![
                shunting_token_fixture(),
                shunting_statement_fixture(),
            ]]
        }),
        Some(|| vec![vec![
            EXPECTED_SHUNTING_AST_NODES_BYTES.to_vec(),
            EXPECTED_SHUNTING_COUNT_BYTES.to_vec(),
            EXPECTED_SHUNTING_ROOTS_BYTES.to_vec(),
            EXPECTED_SHUNTING_SCRATCH_VAL_STACK_BYTES.to_vec(),
            EXPECTED_SHUNTING_SCRATCH_OP_STACK_BYTES.to_vec(),
        ]]),
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
