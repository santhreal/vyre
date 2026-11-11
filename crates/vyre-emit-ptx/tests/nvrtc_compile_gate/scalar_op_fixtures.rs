use super::*;

/// Result and index ids are fixed across the three shapes so the trailing
/// store is written once.
const RESULT_ID: u32 = 3;
const IDX_ID: u32 = 2;

pub(crate) fn ptx_for_op(op_kind: KernelOpKind) -> String {
    let (ops, literals, binding) = match op_kind {
        KernelOpKind::Fma => (
            vec![
                invocation_id(0),
                op(
                    KernelOpKind::Cast {
                        target: DataType::F32,
                    },
                    [0],
                    1,
                ),
                lit(0, 4),
                lit(1, IDX_ID),
                lit(2, 5),
                op(KernelOpKind::Fma, [1, 4, 5], RESULT_ID),
            ],
            vec![
                LiteralValue::F32(2.0),
                LiteralValue::U32(0),
                LiteralValue::F32(3.0),
            ],
            global_rw(0, DataType::F32, "out"),
        ),
        KernelOpKind::BinOpKind(BinOp::Mul) => (
            vec![
                invocation_id(0),
                invocation_id(1),
                lit(0, IDX_ID),
                binop(BinOp::Mul, 0, 1, RESULT_ID),
            ],
            vec![LiteralValue::U32(0)],
            global_rw(0, DataType::U32, "out"),
        ),
        other => (
            vec![
                invocation_id(0),
                lit(0, 1),
                lit(1, IDX_ID),
                op(other, [0, 1], RESULT_ID),
            ],
            vec![LiteralValue::U32(7), LiteralValue::U32(0)],
            global_rw(0, DataType::U32, "out"),
        ),
    };

    let desc = descriptor("test")
        .slot(binding)
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals(literals)
                .ops(ops)
                .op(store_global(0, IDX_ID, RESULT_ID)),
        )
        .build();
    emit_ptx(&desc)
}
