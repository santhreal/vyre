//! Test: atomics.
use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};

#[test]
fn atomic_add_emits_statement() {
    use vyre_foundation::ir::AtomicOp;
    let desc = descriptor("atomic_add")
        .slots([global_rw(0, DataType::U32, "counter")])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(
                        KernelOpKind::Atomic {
                            op: AtomicOp::Add,
                            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
                        },
                        [0, 0, 1],
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let module = emit(&desc).unwrap();
    assert!(!module.entry_points.is_empty());
    assert!(
        module.global_variables.iter().any(|(_, global)| {
            let ty = &module.types[global.ty].inner;
            matches!(
                ty,
                TypeInner::Array { base, .. }
                    if matches!(module.types[*base].inner, TypeInner::Atomic(_))
            )
        }),
        "Fix: descriptor buffers targeted by atomics must use atomic element types, otherwise Naga rejects the emitted atomic pointer."
    );
    // Also assert the atomic operation was actually emitted in the function
    // body, not just that the global variable type is correct. A regressor
    // that sets up the global correctly but drops the Statement::Atomic from
    // the body would pass the type-only check above while silently omitting
    // the atomic operation.
    assert!(
        block_has_atomic(&module.entry_points[0].function.body),
        "AtomicAdd must emit Statement::Atomic in the function body, not just declare \
         the global variable with an atomic element type"
    );
}

#[test]
fn atomic_fetch_nand_emits_compare_exchange_loop() {
    use vyre_foundation::ir::AtomicOp;
    let desc = descriptor("k")
        .slots([global_rw(0, DataType::U32, "b")])
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(
                        KernelOpKind::Atomic {
                            op: AtomicOp::FetchNand,
                            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
                        },
                        [0, 0, 1],
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let module = emit(&desc).expect("FetchNand must lower to a compare-exchange loop");
    let body = &module.entry_points[0].function.body;
    assert!(block_has_loop(body));
    assert!(block_has_atomic(body));
}

#[test]
fn atomic_compare_exchange_emits_statement() {
    use vyre_foundation::ir::AtomicOp;
    let desc = descriptor("atomic_cx")
        .slots([global_rw(0, DataType::U32, "counter")])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(
                        KernelOpKind::Atomic {
                            op: AtomicOp::CompareExchange,
                            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
                        },
                        [0, 0, 0, 1],
                        2,
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let module = emit(&desc).expect("compare-exchange must lower to Naga atomic exchange");
    assert!(block_has_atomic(&module.entry_points[0].function.body));
}
