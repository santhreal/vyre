//! Thread-local counting of arithmetic IR operations executed by the reference
//! interpreter, a backend-agnostic dynamic operation count for roofline / complexity
//! analysis.
//!
//! The reference interpreter executes the *same* vyre IR with the *same* data-dependent
//! control flow that any backend (GPU or CPU) would, so the arithmetic-op count it
//! reports for a `(program, inputs)` pair equals the dynamic IR-op count the GPU would
//! execute for the same inputs, at the **vyre-IR** granularity, which is distinct from
//! (and coarser than) hardware SASS instructions. This gives an honest, non-root
//! operational-intensity measurement (`ops / bytes`) for the roofline without
//! Nsight-Compute; the SASS-level dynamic count remains the ncu refinement.
//!
//! Counting is a no-op unless a [`count_ops`] scope is active on the current thread, so
//! ordinary reference evaluation (the vast majority of interpreter use, all in tests)
//! pays only one thread-local read per arithmetic op and no allocation.

use std::cell::Cell;

thread_local! {
    /// `Some(n)` while a [`count_ops`] scope is active on this thread; `None` otherwise.
    static OP_COUNTER: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Record one arithmetic IR op if counting is active on this thread (a cheap
/// thread-local read otherwise). Called by the interpreter's `BinOp` / `UnOp` / `Fma`
/// evaluation arms (the arithmetic operations that make up the roofline's compute term).
#[inline]
pub(crate) fn record_op() {
    OP_COUNTER.with(|counter| {
        if let Some(count) = counter.get() {
            counter.set(Some(count.saturating_add(1)));
        }
    });
}

/// Run `f` with arithmetic-op counting active and return `(f's result, arithmetic IR
/// ops executed)`. The count is every `BinOp` / `UnOp` / `Fma` the reference interpreter
/// evaluates during `f`: a backend-agnostic dynamic operation count.
///
/// Re-entrant-safe: a nested `count_ops` reports its own inner ops AND propagates them to
/// the enclosing scope, so an outer scope's total still includes work done inside inner
/// scopes.
pub fn count_ops<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let saved = OP_COUNTER.with(|counter| counter.replace(Some(0)));
    let result = f();
    let count = OP_COUNTER
        .with(|counter| counter.replace(saved))
        .unwrap_or(0);
    // Propagate our ops to an enclosing scope, if any, so nesting is additive.
    if saved.is_some() {
        OP_COUNTER.with(|counter| {
            if let Some(outer) = counter.get() {
                counter.set(Some(outer.saturating_add(count)));
            }
        });
    }
    (result, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_ops_is_zero_when_no_ops_run() {
        let (value, ops) = count_ops(|| 42u32);
        assert_eq!(value, 42);
        assert_eq!(ops, 0, "a closure that runs no interpreter ops counts zero");
    }

    #[test]
    fn record_op_is_inert_outside_a_count_scope() {
        // No active scope: record_op must not panic and must not accumulate anywhere.
        record_op();
        record_op();
        let (_, ops) = count_ops(|| {
            record_op();
            record_op();
            record_op();
        });
        assert_eq!(ops, 3, "only ops inside the scope are counted");
    }

    #[test]
    fn nested_scopes_are_additive() {
        let ((_, inner_ops), outer_ops) = count_ops(|| {
            record_op(); // outer: 1
            let inner = count_ops(|| {
                record_op();
                record_op(); // inner: 2
            });
            record_op(); // outer: another 1
            inner
        });
        assert_eq!(inner_ops, 2, "inner scope counts only its own ops");
        assert_eq!(
            outer_ops, 4,
            "outer scope counts its own 2 ops plus the 2 propagated from the inner scope"
        );
    }
}
