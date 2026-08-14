// Integration test module for the containing Vyre package.
//
// The CPU half of this harness (fixtures, row accessors, row assertions) is
// shared with `vyre-libs/tests`, which owns the same C frontend's CPU-only
// contracts, so it lives in `tests/support/c_frontend` and is included here.
// What stays is the GPU half: adapter acquisition, dispatch, and the parity
// assertions that compare a dispatched buffer against the CPU reference.
#![allow(deprecated)]

use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use vyre::ir::{Expr, Program};
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lex::tokens::{
    TOK_ASSIGN, TOK_COLON, TOK_COMMA, TOK_IDENTIFIER, TOK_LBRACE, TOK_LBRACKET, TOK_LPAREN,
    TOK_RBRACE, TOK_RBRACKET, TOK_RPAREN, TOK_SEMICOLON, TOK_TYPEDEF, TOK_VOID,
};
use vyre_libs::parsing::c::lower::{
    c_lower_ast_to_pg_nodes, c_lower_ast_to_pg_semantic_graph, reference_ast_to_pg_nodes,
};
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_global_typedef_names_fast, c11_annotate_typedef_names,
    c11_annotate_typedef_names_precomputed_scope, c11_build_expression_shape_nodes,
    c11_build_vast_nodes, c11_classify_vast_node_kinds, c11_precompute_vast_scopes,
    c11_prehash_vast_identifiers, reference_c11_annotate_typedef_names,
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
};
use vyre_libs::parsing::c::sema::c_sema_scope;

pub(crate) use crate::c_frontend::rows::*;
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::semantic_graph::*;
pub(crate) use crate::c_frontend::token_fixture::*;
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::{expression_pipeline, scope_fixture};

const VAST_TYPEDEF_SYMBOL_FIELD: usize = 9;

mod gpu_dispatch_support;
mod gpu_pipeline_support;
pub(crate) mod scope_gpu_support;
pub(crate) mod semantic_dispatch_edges;
mod typedef_gpu_support;

pub(crate) use gpu_dispatch_support::*;
#[allow(unused_imports)]
pub(crate) use gpu_pipeline_support::*;
#[allow(unused_imports)]
pub(crate) use semantic_dispatch_edges::*;
pub(crate) use typedef_gpu_support::*;

/// Build a fixture for `void f() { <body> }`.
///
/// Every statement-level C fixture needs the same function shell, and the shell
/// is five of the eight tokens a parser fixture usually carries, so restating it
/// per fixture is most of each fixture's text.
pub(crate) fn void_fn_fixture(body: &[FixtureToken]) -> Fixture {
    let mut tokens = Vec::with_capacity(body.len() + 6);
    tokens.extend_from_slice(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
    ]);
    tokens.extend_from_slice(body);
    tokens.push(FixtureToken::new("}", TOK_RBRACE));
    build_fixture(&tokens)
}

/// Build the shared `void f() { __builtin_unreachable(); }` parser fixture.
pub(crate) fn fixture_builtin_unreachable() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("__builtin_unreachable", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}
