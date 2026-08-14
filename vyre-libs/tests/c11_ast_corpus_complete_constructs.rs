//! CPU reference C11 AST construction across a corpus covering every declaration, statement, and
//! expression construct.
#![cfg(feature = "c-parser")]
#![allow(clippy::type_complexity)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/complete_construct_corpus.rs"]
mod complete_construct_corpus;

use crate::c_frontend::rows::{row_indices as typed_indices, word_at, VAST_STRIDE_U32};
use complete_construct_corpus::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_IF_STMT,
    C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT, C_AST_KIND_SIZEOF_EXPR,
};
use vyre_primitives::predicate::node_kind;

#[path = "c11_ast_corpus_complete_constructs/cpu_corpus_macros_and_anonymous_aggregates.rs"]
mod cpu_corpus_macros_and_anonymous_aggregates;
#[path = "c11_ast_corpus_complete_constructs/cpu_enums_sizeof_and_statement_expressions.rs"]
mod cpu_enums_sizeof_and_statement_expressions;
#[path = "c11_ast_corpus_complete_constructs/cpu_function_pointers_designators_and_asm.rs"]
mod cpu_function_pointers_designators_and_asm;
