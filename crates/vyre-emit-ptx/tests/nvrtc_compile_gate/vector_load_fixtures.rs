use super::*;

/// Both load-fusion fixtures read one buffer and write another.
fn fusion_slots() -> [vyre_lower::BindingSlot; 2] {
    [
        global_ro(0, DataType::U32, "input"),
        global_wo(1, DataType::U32, "output"),
    ]
}

/// Four unit-stride loads at compile-time indices, which the fusion pass must
/// coalesce into one vector load.
pub(crate) fn ptx_for_vector_load_fusion() -> String {
    let desc = descriptor("vector_load_fusion")
        .slots(fusion_slots())
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(load_global(0, 0, 2))
                .op(binop(BinOp::Add, 0, 1, 3))
                .op(load_global(0, 3, 4))
                .op(binop(BinOp::Add, 3, 1, 5))
                .op(load_global(0, 5, 6))
                .op(binop(BinOp::Add, 5, 1, 7))
                .op(load_global(0, 7, 8))
                .op(binop(BinOp::Add, 2, 4, 9))
                .op(binop(BinOp::Add, 9, 6, 10))
                .op(binop(BinOp::Add, 10, 8, 11))
                .op(store_global(1, 0, 11)),
        )
        .build();
    emit_ptx(&desc)
}

/// The same four-load chain based at a runtime invocation-derived index, so
/// fusion has to prove the stride without constant indices.
pub(crate) fn ptx_for_dynamic_vector_load_fusion() -> String {
    let desc = descriptor("dynamic_vector_load_fusion")
        .slots(fusion_slots())
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(4), LiteralValue::U32(1)])
                .op(invocation_id(0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(lit(1, 3))
                .op(load_global(0, 2, 4))
                .op(binop(BinOp::Add, 2, 3, 5))
                .op(load_global(0, 5, 6))
                .op(binop(BinOp::Add, 5, 3, 7))
                .op(load_global(0, 7, 8))
                .op(binop(BinOp::Add, 7, 3, 9))
                .op(load_global(0, 9, 10))
                .op(binop(BinOp::Add, 4, 6, 11))
                .op(binop(BinOp::Add, 11, 8, 12))
                .op(binop(BinOp::Add, 12, 10, 13))
                .op(store_global(1, 0, 13)),
        )
        .build();
    emit_ptx(&desc)
}
