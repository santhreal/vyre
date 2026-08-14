//! High-quality integration tests for GNU C and Linux-kernel-shaped constructs in
//! the C AST/VAST parser.
//!
//! Coverage:
//!   - GNU extended asm (templates, input/output operands, clobbers, goto labels)
//!   - GNU attributes: cleanup, alias, aligned, section
//!   - computed goto (`&&label`)
//!   - `__builtin_*` forms: expect, constant_p, choose_expr,
//!     types_compatible_p, plus unrecognized builtins as generic calls
//!   - `_Atomic` qualifier and type specifier
//!   - `typeof_unqual` / `__typeof_unqual__`
//!   - `__auto_type`
//!   - `__int128`
//!   - declarator ambiguity (pointer vs array precedence)
//!   - C99 for-loop declarations
//!   - abstract function pointers in parameter position
//!   - Linux-kernel-shaped declarations (attributes + typeof + function pointers)
//!
//! Tests assert the *intended contract* (distinct VAST kinds, correct tree
//! parentage, no collapse into generic CALL/BINARY) rather than snapshotting
//! current output.
//!
//! A missing GPU adapter is a configuration failure; tests do not skip.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/fixtures/asm_extended_operands.rs"]
mod asm_extended_operands;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, classify, row_indices, void_fn_fixture, word_at, Fixture,
    FixtureToken, VAST_STRIDE_U32,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::gnu_builtins::try_classify_gnu_builtin_name;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_DECL, C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_GOTO_LABELS,
    C_AST_KIND_ASM_INPUT_OPERAND, C_AST_KIND_ASM_OUTPUT_OPERAND, C_AST_KIND_ASM_TEMPLATE,
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_ATTRIBUTE_ALIAS, C_AST_KIND_ATTRIBUTE_ALIGNED,
    C_AST_KIND_ATTRIBUTE_CLEANUP, C_AST_KIND_ATTRIBUTE_SECTION, C_AST_KIND_BUILTIN_CHOOSE_EXPR,
    C_AST_KIND_BUILTIN_CONSTANT_P_EXPR, C_AST_KIND_BUILTIN_EXPECT_EXPR,
    C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR, C_AST_KIND_CAST_EXPR, C_AST_KIND_FOR_STMT,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_GNU_LABEL_ADDRESS_EXPR, C_AST_KIND_GOTO_STMT, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_LABEL_STMT, C_AST_KIND_POINTER_DECL,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parent_of(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 1)
}

// ---------------------------------------------------------------------------
// 1. GNU asm
// ---------------------------------------------------------------------------

use asm_extended_operands::{asm_goto_two_labels, asm_volatile_mov_one_clobber};

fn fixture_asm_goto_with_labels() -> Fixture {
    asm_goto_two_labels("__asm__", TOK_IDENTIFIER, "\"jmp %l0\"")
}

fn fixture_asm_extended_io_clobbers() -> Fixture {
    asm_volatile_mov_one_clobber("\"memory\"")
}

#[path = "c_ast_gnu_and_kernel_construct_integration/asm_goto_classifies_template_and_labels.rs"]
mod asm_goto_classifies_template_and_labels;
#[path = "c_ast_gnu_and_kernel_construct_integration/gnu_type_extensions_and_declarator_precedence.rs"]
mod gnu_type_extensions_and_declarator_precedence;
