//! GPU arms for the scope-fixture layout.
//!
//! The CPU reference passes over a [`ScopeFixture`] live in the shared
//! `tests/support/c_frontend/scope_fixture.rs`; these are the dispatched
//! counterparts each `c_ast_sema_scope_*` parity module compares against. They
//! are deliberately not glob re-exported: `run_gpu_pg_lower` here takes a node
//! count, while the lexeme-fixture arm of the same name does not.

use super::scope_fixture::ScopeFixture;
use super::{
    run_gpu_c_sema_scope_from_parts, run_gpu_classifier_with_count,
    run_gpu_full_typedef_annotation, run_gpu_pg_lower_with_count, run_gpu_vast_builder_from_parts,
};

pub(crate) fn run_gpu_scope_tree(fix: &ScopeFixture) -> Vec<u8> {
    run_gpu_c_sema_scope_from_parts(
        &fix.tok_types,
        &fix.tok_starts,
        &fix.tok_lens,
        &fix.haystack,
    )
}

pub(crate) fn run_gpu_annotate(fix: &ScopeFixture) -> Vec<u8> {
    let raw = super::scope_fixture::raw_vast(fix);
    run_gpu_full_typedef_annotation(&fix.haystack, &raw)
}

pub(crate) fn run_gpu_classify(annotated: &[u8], node_count: usize) -> Vec<u8> {
    run_gpu_classifier_with_count(annotated, node_count as u32)
}

pub(crate) fn run_gpu_vast_builder(fix: &ScopeFixture) -> Vec<u8> {
    run_gpu_vast_builder_from_parts(&fix.tok_types, &fix.tok_starts, &fix.tok_lens)
}

pub(crate) fn run_gpu_pg_lower(vast: &[u8], node_count: usize) -> Vec<u8> {
    run_gpu_pg_lower_with_count(vast, node_count as u32)
}
