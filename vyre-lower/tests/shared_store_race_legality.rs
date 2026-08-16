//! Tests for shared store race legality analysis.

use vyre_foundation::ir::{AtomicOp, BinOp, DataType, MemoryOrdering};
use vyre_lower::analyses::shared_store_race::{
    analyze as analyze_shared_store_race, SharedStoreLegality,
};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, effect, if_then, lit, op, shared_rw,
};
use vyre_lower::{KernelOpKind, LiteralValue};

#[test]
fn single_invocation_constant_store_is_legal() {
    let kernel = descriptor("single_invocation_ok")
        .slot(shared_rw(0, DataType::U32, 1024, "sh_mem"))
        .dispatch(1, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(42)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreShared, [0, 0, 1])),
        )
        .build();

    let report = analyze_shared_store_race(&kernel);
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].legality,
        SharedStoreLegality::RaceFreeSingleInvocation
    );
}

#[test]
fn multi_invocation_guarded_thread_zero_store_is_legal() {
    let child = body()
        .literals([LiteralValue::U32(42)])
        .op(lit(0, 3))
        .op(effect(KernelOpKind::StoreShared, [0, 0, 3]));

    let kernel = descriptor("multi_invocation_guarded_ok")
        .slot(shared_rw(0, DataType::U32, 1024, "sh_mem"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0)])
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Eq, 0, 1, 2))
                .op(if_then(2, 0))
                .child(child),
        )
        .build();

    let report = analyze_shared_store_race(&kernel);
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].legality,
        SharedStoreLegality::RaceFreeSingleInvocation
    );
}

#[test]
fn multi_invocation_thread_varying_store_is_legal() {
    let kernel = descriptor("multi_invocation_varying_ok")
        .slot(shared_rw(0, DataType::U32, 1024, "sh_mem"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(42)])
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(lit(0, 1))
                .op(effect(KernelOpKind::StoreShared, [0, 0, 1])),
        )
        .build();

    let report = analyze_shared_store_race(&kernel);
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].legality,
        SharedStoreLegality::RaceFreeDistinctIndices
    );
}

#[test]
fn multi_invocation_unguarded_constant_store_is_illegal() {
    let kernel = descriptor("multi_invocation_unguarded_bad")
        .slot(shared_rw(0, DataType::U32, 1024, "sh_mem"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(42)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreShared, [0, 0, 1])),
        )
        .build();

    let report = analyze_shared_store_race(&kernel);
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].legality,
        SharedStoreLegality::IllegalMultiInvocationConstantStore
    );
}

#[test]
fn multi_invocation_atomic_store_is_legal() {
    let kernel = descriptor("multi_invocation_atomic_ok")
        .slot(shared_rw(0, DataType::U32, 1024, "sh_mem"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(op(
                    KernelOpKind::Atomic {
                        op: AtomicOp::Add,
                        ordering: MemoryOrdering::SeqCst,
                    },
                    [0, 0, 1],
                    2,
                )),
        )
        .build();

    let report = analyze_shared_store_race(&kernel);
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].legality,
        SharedStoreLegality::RaceFreeAtomic
    );
}
