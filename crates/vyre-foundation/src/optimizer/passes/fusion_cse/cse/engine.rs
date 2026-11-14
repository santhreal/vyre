use super::CseCtx;
use crate::ir_inner::model::program::Program;
use crate::optimizer::PassResult;
use std::borrow::Cow;

/// Run local value-numbering CSE over pure expressions, reusing an existing
/// scratchpad context.
///
/// The context is automatically [`clear`](CseCtx::clear)ed before use so that
/// callers can amortize allocation costs across multiple programs.
#[must_use]
#[inline]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CSE consumes Program to reuse its allocation-preserving with_rewritten_entry path"
)]
pub fn cse_into(program: Program, ctx: &mut CseCtx) -> Program {
    cse_pass_into(program, ctx).program
}

/// Run local value-numbering CSE and report whether any node changed without
/// cloning an unchanged `Program`.
#[must_use]
#[inline]
pub(crate) fn cse_pass_into(program: Program, ctx: &mut CseCtx) -> PassResult {
    ctx.clear();
    match ctx.nodes_cow(program.entry()) {
        Cow::Borrowed(_) => PassResult::unchanged(program),
        Cow::Owned(rewritten) => PassResult::changed(program.with_rewritten_entry(rewritten)),
    }
}

// Thread-local CseCtx scratchpad. Each `cse(program)` call would otherwise
// alloc fresh hashmaps / vectors / arena that immediately get dropped on
// return; the context's `clear()` already preserves allocated capacity, so
// reusing one per thread converts the cost from O(N programs) → O(1)
// initial allocation.
thread_local! {
    static CSE_CTX: std::cell::RefCell<CseCtx> = std::cell::RefCell::new(CseCtx::default());
}

/// Run local value-numbering CSE over pure expressions.
#[must_use]
#[inline]
pub fn cse(program: Program) -> Program {
    cse_pass(program).program
}

/// Run local value-numbering CSE using the thread-local scratchpad context.
#[must_use]
#[inline]
pub(crate) fn cse_pass(program: Program) -> PassResult {
    CSE_CTX.with(|cell| cse_pass_into(program, &mut cell.borrow_mut()))
}
