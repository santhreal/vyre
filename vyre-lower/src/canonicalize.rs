pub(crate) use crate::rewrites::canonicalize_for_emit;
// Inline: covers the crate-private `canonicalize_for_emit`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{body, descriptor, lit, op};
    use crate::{KernelOpKind, LiteralValue};
    use vyre_foundation::ir::BinOp;

    #[test]
    fn one_walk_orders_pure_forward_dependency() {
        let descriptor = descriptor("emit-order")
            .dispatch(1, 1, 1)
            .body(
                body()
                    .literals([LiteralValue::U32(1), LiteralValue::U32(2)])
                    .ops([
                        lit(0, 0),
                        op(KernelOpKind::BinOpKind(BinOp::Add), [0, 2], 3),
                        lit(1, 2),
                    ]),
            )
            .build();

        let output = canonicalize_for_emit(&descriptor);

        assert_eq!(output.body.ops[1].result, Some(2));
        assert_eq!(output.body.ops[2].result, Some(3));
        assert_eq!(canonicalize_for_emit(&output), output);
    }

    #[test]
    fn one_walk_does_not_move_memory_operations() {
        let descriptor = descriptor("side-effect-order")
            .dispatch(1, 1, 1)
            .body(body().ops([
                op(KernelOpKind::BinOpKind(BinOp::Add), [1, 2], 3),
                op(KernelOpKind::LoadGlobal, [0, 1], 2),
            ]))
            .build();

        let output = canonicalize_for_emit(&descriptor);

        assert!(matches!(
            output.body.ops[0].kind,
            KernelOpKind::BinOpKind(BinOp::Add)
        ));
        assert!(matches!(output.body.ops[1].kind, KernelOpKind::LoadGlobal));
    }
}
