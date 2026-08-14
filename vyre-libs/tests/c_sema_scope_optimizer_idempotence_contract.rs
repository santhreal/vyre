//! Optimizer idempotence contract for the GPU C semantic-scope program.

#![cfg(feature = "c-parser")]

mod support;

use support::optimizer::assert_optimizer_is_idempotent;
use vyre::ir::Expr;
use vyre_libs::parsing::c::sema::registry::c_sema_scope;

#[test]
fn c_sema_scope_pre_lowering_optimizer_is_idempotent() {
    assert_optimizer_is_idempotent(c_sema_scope(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(16),
        Expr::u32(14),
        "out_scope_tree",
    ));
}
