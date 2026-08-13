//! Test: subgroup.
use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, lit, op};

fn block_has_subgroup_collective(block: &naga::Block) -> bool {
    block.iter().any(|statement| match statement {
        Statement::SubgroupCollectiveOperation { .. } => true,
        Statement::Block(child) => block_has_subgroup_collective(child),
        Statement::If { accept, reject, .. } => {
            block_has_subgroup_collective(accept) || block_has_subgroup_collective(reject)
        }
        Statement::Loop {
            body, continuing, ..
        } => block_has_subgroup_collective(body) || block_has_subgroup_collective(continuing),
        _ => false,
    })
}

fn first_subgroup_collective_op(block: &naga::Block) -> Option<naga::SubgroupOperation> {
    block.iter().find_map(|statement| match statement {
        Statement::SubgroupCollectiveOperation { op, .. } => Some(*op),
        Statement::Block(child) => first_subgroup_collective_op(child),
        Statement::If { accept, reject, .. } => {
            first_subgroup_collective_op(accept).or_else(|| first_subgroup_collective_op(reject))
        }
        Statement::Loop {
            body, continuing, ..
        } => {
            first_subgroup_collective_op(body).or_else(|| first_subgroup_collective_op(continuing))
        }
        _ => None,
    })
}

fn block_has_subgroup_ballot(block: &naga::Block) -> bool {
    block.iter().any(|statement| match statement {
        Statement::SubgroupBallot { .. } => true,
        Statement::Block(child) => block_has_subgroup_ballot(child),
        Statement::If { accept, reject, .. } => {
            block_has_subgroup_ballot(accept) || block_has_subgroup_ballot(reject)
        }
        Statement::Loop {
            body, continuing, ..
        } => block_has_subgroup_ballot(body) || block_has_subgroup_ballot(continuing),
        _ => false,
    })
}

#[test]
fn subgroup_add_emits_collective_operation() {
    let desc = descriptor("sub_add")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::SubgroupReduce {
                        op: vyre_lower::SubgroupReduceOp::Add,
                    }, [0], 1),
                ])
                .literal(LiteralValue::U32(7)),
        )
        .build();
    let module = emit(&desc).unwrap();
    // Assert the SubgroupCollectiveOperation statement is present in the body,
    // not just that the module compiled. A refactor that converts SubgroupAdd
    // into a no-op would still produce entry_points[0].name == "main" and
    // pass the old name-only assertion while silently dropping the operation.
    let body = &module.entry_points[0].function.body;
    assert!(
        block_has_subgroup_collective(body),
        "SubgroupReduce must emit Statement::SubgroupCollectiveOperation in the function body"
    );
}

#[test]
fn subgroup_reduce_maps_each_op_to_its_naga_operation() {
    use vyre_lower::SubgroupReduceOp;
    let cases = [
        (SubgroupReduceOp::Add, naga::SubgroupOperation::Add),
        (SubgroupReduceOp::Mul, naga::SubgroupOperation::Mul),
        (SubgroupReduceOp::Min, naga::SubgroupOperation::Min),
        (SubgroupReduceOp::Max, naga::SubgroupOperation::Max),
        (SubgroupReduceOp::And, naga::SubgroupOperation::And),
        (SubgroupReduceOp::Or, naga::SubgroupOperation::Or),
        (SubgroupReduceOp::Xor, naga::SubgroupOperation::Xor),
    ];
    for (reduce_op, expected) in cases {
        let desc = descriptor("sub_reduce")
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        op(KernelOpKind::SubgroupReduce { op: reduce_op }, [0], 1),
                    ])
                    .literal(LiteralValue::U32(7)),
            )
            .build();
        let module = emit(&desc).unwrap();
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            first_subgroup_collective_op(body),
            Some(expected),
            "SubgroupReduce({reduce_op:?}) must emit naga SubgroupOperation::{expected:?}, not Add"
        );
    }
}

#[test]
fn subgroup_ballot_emits_ballot_statement() {
    let desc = descriptor("ballot")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([lit(0, 0), op(KernelOpKind::SubgroupBallot, [0], 1)])
                .literal(LiteralValue::Bool(true)),
        )
        .build();
    let module = emit(&desc).unwrap();
    // Assert Statement::SubgroupBallot is present. A no-op refactor would
    // still pass the old name-only assertion.
    let body = &module.entry_points[0].function.body;
    assert!(
        block_has_subgroup_ballot(body),
        "SubgroupBallot must emit Statement::SubgroupBallot in the function body"
    );
}

#[test]
fn subgroup_scalar_builtins_are_emitted_only_when_used() {
    let mut desc = empty_desc();
    desc.body.ops.push(op(KernelOpKind::SubgroupLocalId, [], 0));
    desc.body.ops.push(op(KernelOpKind::SubgroupSize, [], 1));

    let module = emit(&desc).expect("descriptor subgroup scalar builtins must emit");
    let args = &module.entry_points[0].function.arguments;
    assert!(
        args.iter().any(|arg| matches!(
            arg.binding,
            Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId))
        )),
        "SubgroupLocalId must add the subgroup invocation builtin"
    );
    assert!(
        args.iter()
            .any(|arg| matches!(arg.binding, Some(Binding::BuiltIn(BuiltIn::SubgroupSize)))),
        "SubgroupSize must add the subgroup size builtin"
    );
}
