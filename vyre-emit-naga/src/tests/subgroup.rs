//! Test: subgroup.
use super::*;

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
    let desc = KernelDescriptor {
        id: "sub_add".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::SubgroupAdd,
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let module = emit(&desc).unwrap();
    // Assert the SubgroupCollectiveOperation statement is present in the body,
    // not just that the module compiled. A refactor that converts SubgroupAdd
    // into a no-op would still produce entry_points[0].name == "main" and
    // pass the old name-only assertion while silently dropping the operation.
    let body = &module.entry_points[0].function.body;
    assert!(
        block_has_subgroup_collective(body),
        "SubgroupAdd must emit Statement::SubgroupCollectiveOperation in the function body"
    );
}

#[test]
fn subgroup_ballot_emits_ballot_statement() {
    let desc = KernelDescriptor {
        id: "ballot".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::SubgroupBallot,
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::Bool(true)],
        },
    };
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
    desc.body.ops.push(KernelOp {
        kind: KernelOpKind::SubgroupLocalId,
        operands: vec![],
        result: Some(0),
    });
    desc.body.ops.push(KernelOp {
        kind: KernelOpKind::SubgroupSize,
        operands: vec![],
        result: Some(1),
    });

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
