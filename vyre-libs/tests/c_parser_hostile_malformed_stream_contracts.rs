//! Hostile C parser tests for malformed-but-lexically-valid token streams.
//!
//! Covers constructs that survive the lexer but violate C grammar, exposing
//! parser stage behavior through concrete VAST/PG/semantic-graph contracts.
//!
//! Targets:
//!   * unmatched delimiters
//!   * malformed declarations
//!   * unterminated attribute argument lists after lexing
//!   * bad asm operands
//!   * invalid designator nesting
//!   * case/default outside switch (observable via semantic edges)
//!   * label/goto mismatches (observable via semantic edges)
//!   * pathological nesting / resource bounds

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::rows::{row_indices, word_at, VAST_STRIDE_U32};
use c_frontend::semantic_graph::{semantic_edge_word, semantic_node_word};
use c_frontend::token_fixture::{build_fixture, classify, FixtureToken};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{
    reference_ast_to_pg_nodes, reference_ast_to_pg_semantic_graph, C_AST_PG_EDGE_GOTO_TARGET,
    C_AST_PG_EDGE_NONE, C_AST_PG_EDGE_STRIDE_U32, C_AST_PG_EDGE_SWITCH_CASE,
    C_AST_PG_EDGE_SWITCH_DEFAULT, C_AST_PG_ROLE_CASE, C_AST_PG_ROLE_DEFAULT, C_AST_PG_ROLE_GOTO,
};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_OUTPUT_OPERAND, C_AST_KIND_CASE_STMT,
    C_AST_KIND_DEFAULT_STMT, C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GOTO_STMT, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_INLINE_ASM, C_AST_KIND_LABEL_STMT, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_SWITCH_STMT,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    reference_ast_to_pg_nodes(typed_vast)
}

fn semantic_lower(typed_vast: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let sg = reference_ast_to_pg_semantic_graph(typed_vast);
    (sg.nodes, sg.edges)
}

// ---------------------------------------------------------------------------
// 1. Unmatched delimiters  -  must not crash, must emit structural rows
// ---------------------------------------------------------------------------

#[path = "c_parser_hostile_malformed_stream_contracts/duplicate_cases_and_malformed_labels.rs"]
mod duplicate_cases_and_malformed_labels;
#[path = "c_parser_hostile_malformed_stream_contracts/semantic_edges_and_token_boundaries.rs"]
mod semantic_edges_and_token_boundaries;
#[path = "c_parser_hostile_malformed_stream_contracts/unmatched_delimiters_and_malformed_declarations.rs"]
mod unmatched_delimiters_and_malformed_declarations;
