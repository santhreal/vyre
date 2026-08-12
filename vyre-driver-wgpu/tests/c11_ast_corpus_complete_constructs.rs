//! Generated wrapper test crate for c11 ast corpus complete constructs.
//!
//! Implementation lives in `contract_cases/` chunks.
#![cfg(feature = "c-parser")]
#![allow(clippy::type_complexity)]
#![allow(deprecated)]
use std::sync::OnceLock;
use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    c11_build_vast_nodes, c11_classify_vast_node_kinds, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_ENUMERATOR_DECL,
    C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_INLINE_ASM, C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_POINTER_DECL,
    C_AST_KIND_RETURN_STMT, C_AST_KIND_SIZEOF_EXPR,
};
use vyre_primitives::predicate::node_kind;
const VAST_STRIDE_U32: usize = 10;
const PG_STRIDE_U32: usize = 6;
fn bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}
fn word_at(buf: &[u8], word: usize) -> u32 {
    let off = word * 4;
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn starts_for_lens(lens: &[u32]) -> Vec<u32> {
    let mut cursor = 0u32;
    lens.iter()
        .map(|len| {
            let start = cursor;
            cursor = cursor.saturating_add(*len).saturating_add(1);
            start
        })
        .collect()
}
#[allow(dead_code)]
fn assert_vast_row(
    rows: &[u8],
    idx: usize,
    kind: u32,
    parent: u32,
    first_child: u32,
    next_sibling: u32,
) {
    let row = idx * VAST_STRIDE_U32;
    assert_eq!(word_at(rows, row), kind, "kind[{idx}]");
    assert_eq!(word_at(rows, row + 1), parent, "parent[{idx}]");
    assert_eq!(word_at(rows, row + 2), first_child, "first_child[{idx}]");
    assert_eq!(word_at(rows, row + 3), next_sibling, "next_sibling[{idx}]");
}
fn assert_kind(rows: &[u8], idx: usize, kind: u32) {
    assert_eq!(word_at(rows, idx * VAST_STRIDE_U32), kind, "kind[{idx}]");
}
fn typed_indices(rows: &[u8], kind: u32) -> Vec<usize> {
    rows.chunks_exact(VAST_STRIDE_U32 * 4)
        .enumerate()
        .filter_map(|(idx, row)| {
            let row_kind = u32::from_le_bytes(row[0..4].try_into().unwrap());
            (row_kind == kind).then_some(idx)
        })
        .collect()
}
fn pg_word_at(buf: &[u8], idx: usize, field: usize) -> u32 {
    word_at(buf, idx * PG_STRIDE_U32 + field)
}
fn gpu_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "WgpuBackend::acquire failed on a machine that must have a GPU. \
             Per project GPU rule, this is a configuration bug, not a graceful skip.",
        )
    })
}
fn run_gpu_vast_builder(tok_types: &[u32], tok_starts: &[u32], tok_lens: &[u32]) -> Vec<u8> {
    let program = c11_build_vast_nodes(
        "tok_types",
        "tok_starts",
        "tok_lens",
        Expr::u32(tok_types.len() as u32),
        "out_vast_nodes",
        "out_count",
    );
    let tok_type_bytes = bytes(tok_types);
    let tok_start_bytes = bytes(tok_starts);
    let tok_len_bytes = bytes(tok_lens);
    let inputs: Vec<&[u8]> = vec![&tok_type_bytes, &tok_start_bytes, &tok_len_bytes];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU VAST builder dispatch must succeed");
    assert_eq!(outputs.len(), 2);
    outputs[0].clone()
}
fn run_gpu_classifier(raw_vast: &[u8], num_nodes: u32) -> Vec<u8> {
    let program =
        c11_classify_vast_node_kinds("vast_nodes", Expr::u32(num_nodes), "typed_vast_nodes");
    let inputs: Vec<&[u8]> = vec![raw_vast];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU classifier dispatch must succeed");
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}
fn run_gpu_pg_lower(typed_vast: &[u8], num_nodes: u32) -> Vec<u8> {
    let program = c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(num_nodes), "out_pg_nodes");
    let inputs: Vec<&[u8]> = vec![typed_vast];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU PG lowerer dispatch must succeed");
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}
fn fixture_macro_shaped_decl_after_preproc() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_PREPROC,
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_STRING,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_INT,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![7, 13, 1, 1, 10, 1, 10, 1, 1, 1, 3, 1, 5, 1, 3, 2, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_nested_anonymous_aggregates() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_STRUCT,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_UNION,
        TOK_LBRACE,
        TOK_FLOAT_KW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_ENUM,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_function_pointer_array() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STATIC,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_CONST,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_VOID,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_nested_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_STRING,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_RBRACE,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_attribute_and_asm() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_GNU_ASM,
        TOK_VOLATILE,
        TOK_LPAREN,
        TOK_STRING,
        TOK_COLON,
        TOK_COLON,
        TOK_COLON,
        TOK_STRING,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RETURN,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_enum_values() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_ENUM,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_sizeof_type_vs_expr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
fn fixture_stmt_expr_nesting() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RETURN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_GT,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_QUESTION,
        TOK_LPAREN,
        TOK_LBRACE,
        TOK_IF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_GT,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_RETURN,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_RPAREN,
        TOK_COLON,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
struct CorpusCase {
    name: &'static str,
    fixture: fn() -> (Vec<u32>, Vec<u32>, Vec<u32>),
}
const CORPUS_CASES: &[CorpusCase] = &[
    CorpusCase {
        name: "macro_shaped_decl_after_preproc",
        fixture: fixture_macro_shaped_decl_after_preproc,
    },
    CorpusCase {
        name: "nested_anonymous_aggregates",
        fixture: fixture_nested_anonymous_aggregates,
    },
    CorpusCase {
        name: "function_pointer_array",
        fixture: fixture_function_pointer_array,
    },
    CorpusCase {
        name: "nested_designated_init",
        fixture: fixture_nested_designated_init,
    },
    CorpusCase {
        name: "attribute_and_asm",
        fixture: fixture_attribute_and_asm,
    },
    CorpusCase {
        name: "enum_values",
        fixture: fixture_enum_values,
    },
    CorpusCase {
        name: "sizeof_type_vs_expr",
        fixture: fixture_sizeof_type_vs_expr,
    },
    CorpusCase {
        name: "stmt_expr_nesting",
        fixture: fixture_stmt_expr_nesting,
    },
];
mod c11_ast_corpus_complete_constructs_cpu_reference_all_corpus_fixtures_build_vast_without_panic {
    include!("contract_cases/c11_ast_corpus_complete_constructs__cpu_reference_all_corpus_fixtures_build_vast_without_panic.rs");
}
mod c11_ast_corpus_complete_constructs_cpu_reference_function_pointer_array_with_qualifiers {
    include!("contract_cases/c11_ast_corpus_complete_constructs__cpu_reference_function_pointer_array_with_qualifiers.rs");
}
mod c11_ast_corpus_complete_constructs_cpu_reference_enum_values_typed_correctly {
    include!("contract_cases/c11_ast_corpus_complete_constructs__cpu_reference_enum_values_typed_correctly.rs");
}
mod c11_ast_corpus_complete_constructs_pg_lower_preserves_corpus_kinds_and_spans {
    include!("contract_cases/c11_ast_corpus_complete_constructs__pg_lower_preserves_corpus_kinds_and_spans.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_vast_builder_nested_designated_init {
    include!("contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_vast_builder_nested_designated_init.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_classifier_nested_anonymous_aggregates {
    include!("contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_classifier_nested_anonymous_aggregates.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_classifier_attribute_and_asm {
    include!("contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_classifier_attribute_and_asm.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_classifier_stmt_expr_nesting {
    include!("contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_classifier_stmt_expr_nesting.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_pg_lower_function_pointer_array {
    include!("contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_pg_lower_function_pointer_array.rs");
}
mod c11_ast_corpus_complete_constructs_gpu_parity_pg_lower_enum_values {
    include!(
        "contract_cases/c11_ast_corpus_complete_constructs__gpu_parity_pg_lower_enum_values.rs"
    );
}
