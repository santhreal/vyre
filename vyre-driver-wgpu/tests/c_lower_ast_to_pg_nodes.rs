//! CPU, WGSL, and GPU parity tests for C VAST-to-PG lowering.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#![allow(
    clippy::useless_conversion,
    clippy::if_same_then_else,
    clippy::unnecessary_cast
)]
#[path = "../../tests/support/c_frontend/fixtures/vast_builder_token_streams.rs"]
mod vast_builder_token_streams;
use c_grammar_gen::lex_c11_max_munch::lex_c11_max_munch_kinds;
use proptest::prelude::*;
use std::sync::OnceLock;
use vast_builder_token_streams::{
    declarator_initializer_fixture, expression_operator_fixture,
    function_pointer_array_prototype_fixture,
};
use vyre::ir::{Expr, Program};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_emit_naga::program as naga_emit;
use vyre_foundation::operation::SemanticOperation;
use vyre_foundation::optimizer::optimize;
use vyre_libs::operation_catalog::all_entries;
use vyre_libs::parsing::c::lex::keyword::reference_c_keyword_types;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_TEMPLATE,
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CAST_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR,
    C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT,
    C_AST_KIND_SIZEOF_EXPR, C_AST_KIND_UNARY_EXPR,
};
use vyre_libs::predicate::node_kind;
use vyre_reference::value::Value;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::rows::{
    bytes, node_count_from_vast, row_indices_by_stride as row_indices, starts_for_lens, word_at,
};
use c_frontend::token_fixture::FixtureToken;

const VAST_STRIDE_U32: u32 = 10;
const VAST_STRIDE_BYTES: usize = (VAST_STRIDE_U32 as usize) * core::mem::size_of::<u32>();
const PG_STRIDE_U32: u32 = 6;
const TEST_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
const OP_ID: &str = "vyre-libs::parsing::c::lower::ast_to_pg_nodes";
fn gnu_c_stress_fixture_source_and_tokens() -> (String, Vec<u32>, Vec<u32>, Vec<u32>) {
    let tokens = [
        FixtureToken::new(
            "#define likely(x) __builtin_expect(!!(x), 1)\n",
            TOK_PREPROC,
        ),
        FixtureToken::new("typedef", TOK_TYPEDEF),
        FixtureToken::new("long", TOK_LONG),
        FixtureToken::new("fault_cb_t", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("file", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("static", TOK_STATIC),
        FixtureToken::new("inline", TOK_INLINE),
        FixtureToken::new("long", TOK_LONG),
        FixtureToken::new("handle_fault", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("file", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("const", TOK_CONST),
        FixtureToken::new("char", TOK_CHAR_KW),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("name", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("always_inline", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("trace_fault", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("name", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("if", TOK_IF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("likely", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new("&", TOK_AMP),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("asm", TOK_GNU_ASM),
        FixtureToken::new("volatile", TOK_VOLATILE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mfence\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"memory\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new("do_fault", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ];
    let mut source = String::new();
    let mut starts = Vec::with_capacity(tokens.len());
    let mut lens = Vec::with_capacity(tokens.len());
    let mut raw_kinds = Vec::with_capacity(tokens.len());
    for token in tokens {
        if !source.is_empty() && !source.ends_with('\n') {
            source.push(' ');
        }
        starts.push(source.len() as u32);
        source.push_str(token.lexeme);
        lens.push(token.lexeme.len() as u32);
        raw_kinds.push(token.raw_kind);
    }
    (source, raw_kinds, starts, lens)
}
fn entry() -> SemanticOperation {
    all_entries()
        .find(|entry| entry.id == OP_ID)
        .unwrap_or_else(|| panic!("Fix: missing canonical operation registration for {OP_ID}"))
}
fn assert_reference_witnesses(
    program: &Program,
    inputs: &[Vec<Vec<u8>>],
    expected: &[Vec<Vec<u8>>],
) {
    assert_eq!(
        inputs.len(),
        expected.len(),
        "Fix: every witness input case must have an expected output case"
    );
    for (case_idx, (case_inputs, case_expected)) in inputs.iter().zip(expected).enumerate() {
        let actual = run_reference_eval(program, case_inputs);
        assert_eq!(
            actual, *case_expected,
            "Fix: witness case {case_idx} must match CPU reference output"
        );
    }
}
fn run_reference_eval(program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let owned_inputs;
    let inputs = if inputs.len() == 1 {
        let output_len = node_count_from_vast(&inputs[0])
            .saturating_mul(PG_STRIDE_U32)
            .max(1) as usize
            * 4;
        owned_inputs = vec![inputs[0].clone(), vec![0; output_len]];
        owned_inputs.as_slice()
    } else {
        inputs
    };
    let values = inputs.iter().cloned().map(Value::from).collect::<Vec<_>>();
    vyre_reference::reference_eval(program, &values)
        .unwrap_or_else(|error| panic!("Fix: CPU reference must execute: {error}"))
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}
fn emit_wgsl(program: &Program) -> String {
    let module = naga_emit::emit_module(program, TEST_WORKGROUP_SIZE)
        .expect("Fix: program must lower to a Naga module");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Fix: emitted Naga module must validate");
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("Fix: Naga module must serialize to WGSL")
}
fn build_vast_node(
    kind: u32,
    parent_idx: u32,
    span_start: u32,
    span_len: u32,
    attr_off: u32,
    attr_len: u32,
) -> Vec<u32> {
    vec![
        kind,
        parent_idx,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        span_start,
        span_len,
        attr_off,
        attr_len,
        u32::MAX,
    ]
}
fn build_vast(nodes: &[Vec<u32>]) -> Vec<u8> {
    nodes.iter().flat_map(|node| bytes(node)).collect()
}
fn assert_pg_row(
    rows: &[u8],
    idx: usize,
    kind: u32,
    parent: u32,
    first_child: u32,
    next_sibling: u32,
) {
    let row = idx * PG_STRIDE_U32 as usize;
    assert_eq!(word_at(rows, row), kind, "pg kind[{idx}]");
    assert_eq!(word_at(rows, row + 3), parent, "pg parent[{idx}]");
    assert_eq!(word_at(rows, row + 4), first_child, "pg first_child[{idx}]");
    assert_eq!(
        word_at(rows, row + 5),
        next_sibling,
        "pg next_sibling[{idx}]"
    );
}

fn adversarial_vast_cases() -> Vec<Vec<u8>> {
    let mut cases = Vec::with_capacity(64);
    for case_idx in 0..60 {
        let node_count = (case_idx % 16) + 1;
        let mut nodes = Vec::new();
        let seed = u32::try_from(case_idx).unwrap_or_default();
        for node_idx in 0..node_count {
            let kind = match (seed + node_idx) % 6 {
                0 => node_kind::VARIABLE,
                1 => node_kind::CALL,
                2 => node_kind::IMPORT,
                3 => node_kind::LITERAL,
                4 => node_kind::SSA,
                _ => node_kind::BASIC_BLOCK,
            };
            let parent = if node_idx == 0 {
                u32::MAX
            } else if node_idx % 3 == 0 {
                u32::MAX
            } else {
                seed.wrapping_mul(0x9E37_79B9)
                    .wrapping_add(u32::try_from(node_idx).unwrap_or_default())
            };
            let span_start = seed
                .rotate_left((node_idx % 32) as u32)
                .wrapping_add(u32::try_from(node_idx).unwrap_or_default().wrapping_mul(17));
            let span_len = if node_idx % 4 == 0 {
                u32::MAX
            } else {
                seed.wrapping_mul(97)
                    .wrapping_add(u32::try_from(node_idx).unwrap_or_default())
            };
            let attr_off = seed
                .wrapping_mul(31)
                .wrapping_add(u32::try_from(node_idx).unwrap_or_default() * 13);
            let attr_len = if node_idx % 2 == 0 {
                0
            } else {
                seed.wrapping_mul(9)
                    .wrapping_add(u32::try_from(node_idx).unwrap_or_default() * 7)
            };
            nodes.push(build_vast_node(
                kind, parent, span_start, span_len, attr_off, attr_len,
            ));
        }
        cases.push(build_vast(&nodes));
    }
    cases
}
#[path = "c_lower_ast_to_pg_nodes/adversarial_fixtures_and_gpu_dispatch.rs"]
mod adversarial_fixtures_and_gpu_dispatch;
#[path = "c_lower_ast_to_pg_nodes/declarator_and_function_pointer_prototypes.rs"]
mod declarator_and_function_pointer_prototypes;
#[path = "c_lower_ast_to_pg_nodes/registration_parity_and_tree_links.rs"]
mod registration_parity_and_tree_links;
#[path = "c_lower_ast_to_pg_nodes/stress_fixture_wgsl_and_certificate.rs"]
mod stress_fixture_wgsl_and_certificate;
#[path = "c_lower_ast_to_pg_nodes/typed_vast_lowering.rs"]
mod typed_vast_lowering;
