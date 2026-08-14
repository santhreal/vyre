//! CPU reference C11 AST construction across a corpus covering every declaration, statement, and
//! expression construct.
#![cfg(feature = "c-parser")]
#![allow(clippy::type_complexity)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/complete_construct_corpus.rs"]
mod complete_construct_corpus;

use c_frontend::rows::{
    assert_kind, bytes, pg_word_at, row_indices as typed_indices, word_at, PG_STRIDE_U32,
    VAST_STRIDE_U32,
};
use complete_construct_corpus::*;
use std::sync::OnceLock;
use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    c11_build_vast_nodes, c11_classify_vast_node_kinds, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT, C_AST_KIND_SIZEOF_EXPR,
};
use vyre_primitives::predicate::node_kind;

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

#[path = "c11_ast_corpus_complete_constructs/gpu_classifier_aggregates_and_designators.rs"]
mod gpu_classifier_aggregates_and_designators;
#[path = "c11_ast_corpus_complete_constructs/gpu_classifier_asm_enums_and_sizeof.rs"]
mod gpu_classifier_asm_enums_and_sizeof;
#[path = "c11_ast_corpus_complete_constructs/gpu_pg_lowering_enums_sizeof_and_statement_expressions.rs"]
mod gpu_pg_lowering_enums_sizeof_and_statement_expressions;
#[path = "c11_ast_corpus_complete_constructs/gpu_pg_lowering_function_pointers_and_asm.rs"]
mod gpu_pg_lowering_function_pointers_and_asm;
#[path = "c11_ast_corpus_complete_constructs/gpu_statement_expressions_and_macro_declarations.rs"]
mod gpu_statement_expressions_and_macro_declarations;
#[path = "c11_ast_corpus_complete_constructs/gpu_vast_builder_designators_and_statement_expressions.rs"]
mod gpu_vast_builder_designators_and_statement_expressions;
#[path = "c11_ast_corpus_complete_constructs/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
