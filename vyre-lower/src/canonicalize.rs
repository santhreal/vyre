pub(crate) use crate::rewrites::canonicalize::canonicalize_for_emit;
// Inline: covers the crate-private `canonicalize_for_emit`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
    };
    use vyre_foundation::ir::BinOp;

    #[test]
    fn one_walk_orders_pure_forward_dependency() {
        let descriptor = KernelDescriptor {
            id: "emit-order".into(),
            bindings: BindingLayout { slots: Vec::new() },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![0, 2],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(2),
                    },
                ],
                child_bodies: Vec::new(),
                literals: vec![LiteralValue::U32(1), LiteralValue::U32(2)],
            },
        };

        let output = canonicalize_for_emit(&descriptor);

        assert_eq!(output.body.ops[1].result, Some(2));
        assert_eq!(output.body.ops[2].result, Some(3));
        assert_eq!(canonicalize_for_emit(&output), output);
    }

    #[test]
    fn one_walk_does_not_move_memory_operations() {
        let descriptor = KernelDescriptor {
            id: "side-effect-order".into(),
            bindings: BindingLayout { slots: Vec::new() },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![1, 2],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 1],
                        result: Some(2),
                    },
                ],
                child_bodies: Vec::new(),
                literals: Vec::new(),
            },
        };

        let output = canonicalize_for_emit(&descriptor);

        assert!(matches!(
            output.body.ops[0].kind,
            KernelOpKind::BinOpKind(BinOp::Add)
        ));
        assert!(matches!(output.body.ops[1].kind, KernelOpKind::LoadGlobal));
    }
}
