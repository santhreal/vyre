use super::*;

fn output_slot() -> vyre_lower::BindingSlot {
    global_wo(0, DataType::U32, "output")
}

/// Four unit-stride stores at compile-time indices, which the fusion pass must
/// coalesce into one vector store.
pub(crate) fn ptx_for_vector_store_fusion() -> String {
    let desc = descriptor("vector_store_fusion")
        .slot(output_slot())
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(10),
                    LiteralValue::U32(11),
                    LiteralValue::U32(12),
                    LiteralValue::U32(13),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                ])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(lit(2, 2))
                .op(lit(3, 3))
                .op(lit(4, 4))
                .op(store_global(0, 0, 1))
                .op(lit(5, 5))
                .op(store_global(0, 5, 2))
                .op(lit(6, 6))
                .op(store_global(0, 6, 3))
                .op(lit(7, 7))
                .op(store_global(0, 7, 4)),
        )
        .build();
    emit_ptx(&desc)
}

/// The same four-store chain based at a runtime invocation-derived index, so
/// fusion has to prove the stride without constant indices.
pub(crate) fn ptx_for_dynamic_vector_store_fusion() -> String {
    let desc = descriptor("dynamic_vector_store_fusion")
        .slot(output_slot())
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(4),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                    LiteralValue::U32(1000),
                    LiteralValue::U32(1001),
                    LiteralValue::U32(1002),
                    LiteralValue::U32(1003),
                ])
                .op(invocation_id(0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(lit(1, 3))
                .op(lit(2, 4))
                .op(lit(3, 5))
                .op(lit(4, 6))
                .op(binop(BinOp::Add, 0, 6, 7))
                .op(lit(5, 8))
                .op(binop(BinOp::Add, 0, 8, 9))
                .op(lit(6, 10))
                .op(binop(BinOp::Add, 0, 10, 11))
                .op(lit(7, 12))
                .op(binop(BinOp::Add, 0, 12, 13))
                .op(store_global(0, 2, 7))
                .op(binop(BinOp::Add, 2, 3, 14))
                .op(store_global(0, 14, 9))
                .op(binop(BinOp::Add, 2, 4, 15))
                .op(store_global(0, 15, 11))
                .op(binop(BinOp::Add, 2, 5, 16))
                .op(store_global(0, 16, 13)),
        )
        .build();
    emit_ptx(&desc)
}
