//! Kernel-style C AST corpus coverage for parser constructs that are common in
//! systems code but are not Linux-specific.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_gpu_parity_support/mod.rs"]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    build_fixture, kind_at, lexeme_indices, row_indices, run_gpu_classifier,
    run_gpu_scoped_typedef_annotation, run_gpu_vast_builder_from_parts, Fixture, FixtureToken,
};
use c_grammar_gen::lex_c11_max_munch::lex_c11_max_munch_kinds;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_BREAK_STMT, C_AST_KIND_CASE_STMT,
    C_AST_KIND_CAST_EXPR, C_AST_KIND_DEFAULT_STMT, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GOTO_STMT, C_AST_KIND_LABEL_STMT, C_AST_KIND_POINTER_DECL,
    C_AST_KIND_RETURN_STMT, C_AST_KIND_SIZEOF_EXPR, C_AST_KIND_SWITCH_STMT,
};
use vyre_libs::predicate::node_kind;

use c_frontend::rows::{
    assert_words_eq, flags_at, ORDINARY_FLAG_DECL, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};

fn fixture_token_stream() -> Fixture {
    let tokens = [
        FixtureToken::new(
            "#define READ_ONCE(x) (*(volatile typeof(x) *)&(x))\n",
            TOK_PREPROC,
        ),
        FixtureToken::new(
            "#define fallthrough __attribute__((fallthrough))\n",
            TOK_PREPROC,
        ),
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("probe_cb_t", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("device", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("_Atomic", TOK_IDENTIFIER),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("state", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("static", TOK_IDENTIFIER),
        FixtureToken::new("inline", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("always_inline", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("dispatch", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("device", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("probe_cb_t", TOK_IDENTIFIER),
        FixtureToken::new("cb", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new("saved", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("sizeof", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("sizeof", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("tmp", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("tmp", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("word_t", TOK_IDENTIFIER),
        FixtureToken::new("restored", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("saved", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("sizeof", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("restored", TOK_IDENTIFIER),
        FixtureToken::new("+", TOK_PLUS),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("again", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("switch", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("case", TOK_IDENTIFIER),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("cb", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("break", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("case", TOK_IDENTIFIER),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("goto", TOK_IDENTIFIER),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("default", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("READ_ONCE", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("fallthrough", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("return", TOK_IDENTIFIER),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new("?", TOK_QUESTION),
        FixtureToken::new("expr_sz", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("saved", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ];

    build_fixture(&tokens)
}

fn assert_flag(rows: &[u8], idx: usize, flag: u32, message: &str) {
    assert_ne!(flags_at(rows, idx) & flag, 0, "{message} at row {idx}");
}

fn assert_no_flag(rows: &[u8], idx: usize, flag: u32, message: &str) {
    assert_eq!(flags_at(rows, idx) & flag, 0, "{message} at row {idx}");
}

fn run_gpu_vast_builder(fix: &Fixture) -> Vec<u8> {
    run_gpu_vast_builder_from_parts(&fix.tok_types, &fix.tok_starts, &fix.tok_lens)
}

fn run_gpu_typedef_annotation(fix: &Fixture, raw_vast: &[u8]) -> Vec<u8> {
    run_gpu_scoped_typedef_annotation(fix.source.as_bytes(), raw_vast)
}

#[path = "c_ast_kernel_style_corpus/kernel_style_fixture.rs"]
mod kernel_style_fixture;
