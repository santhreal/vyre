use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, global_rw, lit, op, SlotCount};

#[test]
fn select_emits_selp_with_correct_dtype() {
    let kernel = descriptor("select")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    // cond bool
                    lit(1, 1),
                    // u32
                    lit(2, 2),
                    // u32
                    op(KernelOpKind::Select, [0, 1, 2], 3),
                ])
                .literals([
                    LiteralValue::Bool(true),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                ]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("selp.u32"));
}

#[test]
fn atomic_compare_exchange_emits_atom_global_cas_b32() {
    use vyre_foundation::ir::{AtomicOp, MemoryOrdering};
    let kernel = descriptor("cas")
        .slot(global_rw(0, DataType::U32, "buf").with_count(4))
        .body(
            body()
                .ops([
                    lit(0, 0),
                    // index
                    lit(1, 1),
                    // cmp
                    lit(2, 2),
                    // new
                    op(
                        KernelOpKind::Atomic {
                            op: AtomicOp::CompareExchange,
                            ordering: MemoryOrdering::SeqCst,
                        },
                        [0, 0, 1, 2],
                        3,
                    ),
                ])
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(7),
                    LiteralValue::U32(8),
                ]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("atom.global.cas.b32"),
        "must emit atom.global.cas.b32:\n{s}"
    );
}

#[test]
fn select_on_predicates_does_not_emit_selp_pred() {
    // PTX `selp` does not support `.pred` operands. ptxas rejects
    // `selp.pred` with "Unexpected instruction types specified for 'selp'".
    // When both arms are bool, lower as not/and/and/or.
    let kernel = descriptor("select_pred")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    // cond bool
                    lit(1, 1),
                    // bool true
                    lit(2, 2),
                    // bool false
                    op(KernelOpKind::Select, [0, 1, 2], 3),
                ])
                .literals([
                    LiteralValue::Bool(true),
                    LiteralValue::Bool(true),
                    LiteralValue::Bool(false),
                ]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(
        !s.contains("selp.pred"),
        "must not emit invalid selp.pred:\n{s}"
    );
    assert!(
        s.contains("not.pred") && s.contains("and.pred") && s.contains("or.pred"),
        "predicate select must lower to not/and/or:\n{s}"
    );
}

#[test]
fn fma_emits_fma_rn_with_dtype() {
    let kernel = descriptor("fma")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    op(KernelOpKind::Fma, [0, 1, 2], 3),
                ])
                .literals([
                    LiteralValue::F32(1.0),
                    LiteralValue::F32(2.0),
                    LiteralValue::F32(3.0),
                ]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("fma.rn.f32"));
}
