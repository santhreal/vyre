//! A build without `subgroup-ops` has no subgroup, and answers none.
//!
//! `subgroup-ops` is a default feature, so every gate in the workspace runs the
//! interpreter with a real subgroup simulator behind it. Without the feature the
//! interpreter used to answer every subgroup expression anyway, from a subgroup
//! one lane wide: the size was 1, the local id 0, a ballot was its own
//! condition, a reduction its own value, and a shuffle answered `0` for every
//! lane but the first. A device runs 32 or 64 lanes, so each of those is a value
//! no execution produces, and the oracle issued them as the expected output a
//! backend is graded against.
//!
//! Two rules state the refusal. Validation owns the first: the oracle declares
//! `supports_subgroup_ops: false` to it in this configuration, so a program
//! carrying a subgroup expression is refused as V041 before a value is
//! evaluated. The evaluator owns the second, for a caller that validated the
//! program itself under capabilities of its own. This file is the only place
//! either is checked, because no other test in the crate compiles under the
//! feature-off configuration.
#![cfg(not(feature = "subgroup-ops"))]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, SubgroupReduceOp};
use vyre_reference::value::Value;
use vyre_reference::workgroup::InvocationIds;
use vyre_reference::{reference_eval_expr, ReferenceErrorClass, ReferenceMemory, ReferenceRequest};

/// A program that stores `expr` into a one-element output buffer.
fn storing(expr: Expr) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), expr)],
    )
}

/// Every subgroup expression, each paired with the name it is refused under.
///
/// The reductions come from `SubgroupReduceOp::ALL` rather than a list written
/// here, so adding an operator turns these red until it is accounted for.
fn every_subgroup_expression() -> Vec<(Expr, String)> {
    let mut exprs = vec![
        (Expr::SubgroupSize, "subgroup_size".to_string()),
        (Expr::SubgroupLocalId, "subgroup_local_id".to_string()),
        (
            Expr::SubgroupBallot {
                cond: Box::new(Expr::u32(1)),
            },
            "subgroup_ballot".to_string(),
        ),
        (
            Expr::SubgroupShuffle {
                value: Box::new(Expr::u32(7)),
                lane: Box::new(Expr::u32(0)),
            },
            "subgroup_shuffle".to_string(),
        ),
        (
            Expr::SubgroupShuffle {
                value: Box::new(Expr::u32(7)),
                lane: Box::new(Expr::u32(31)),
            },
            "subgroup_shuffle".to_string(),
        ),
    ];
    for op in SubgroupReduceOp::ALL {
        exprs.push((
            Expr::SubgroupReduce {
                op,
                value: Box::new(Expr::u32(9)),
            },
            "subgroup_reduce".to_string(),
        ));
    }
    exprs
}

/// WHY: the request path a conformance run uses. Each expression used to return
/// a one-lane answer through it; none may now produce an output at all. The case
/// asserts the class, because the caller routes on it to tell a build that
/// cannot run the program from a program that is wrong.
#[test]
fn a_subgroup_expression_produces_no_output_in_a_build_without_a_subgroup() {
    for (expr, _) in every_subgroup_expression() {
        let refusal = match ReferenceRequest::standard(&storing(expr.clone()), &[]).outputs() {
            Ok(outputs) => panic!(
                "Fix: {expr:?} must be refused, got {} outputs",
                outputs.len()
            ),
            Err(refusal) => refusal,
        };
        assert_eq!(
            refusal.error_class(),
            ReferenceErrorClass::IncompleteDispatchSemantics,
            "Fix: a subgroup this build cannot model is an unsupported capability, got: {refusal}"
        );
        assert!(
            refusal.to_string().contains("subgroup"),
            "Fix: the refusal must name what it could not evaluate, got: {refusal}"
        );
    }
}

/// WHY: validation refuses these programs first, so the evaluator's own arms are
/// reached only by a caller that validated under capabilities of its own. That
/// caller is exactly the one who would be graded against a fabricated lane
/// value, so the arms are driven directly here. Without this the one-lane
/// answers could come back and every other case in the crate would still pass.
#[test]
fn the_evaluator_refuses_a_subgroup_expression_it_is_handed_directly() {
    let program = storing(Expr::u32(0));
    for (expr, name) in every_subgroup_expression() {
        let mut memory = ReferenceMemory::empty();
        let refusal = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &expr)
            .expect_err("Fix: the evaluator must refuse a subgroup expression it cannot model");
        assert_eq!(
            refusal.error_class(),
            ReferenceErrorClass::IncompleteDispatchSemantics,
            "Fix: {name} must be refused as an unsupported capability, got: {refusal}"
        );
        assert!(
            refusal.to_string().contains(&name) && refusal.to_string().contains("subgroup-ops"),
            "Fix: the refusal must name {name} and the feature that models it, got: {refusal}"
        );
    }
}

/// WHY: the refusal must be the subgroup rule rather than a program that fails
/// for an unrelated reason, so the same shape without a subgroup expression runs
/// and returns its value.
#[test]
fn a_program_without_a_subgroup_expression_still_runs() {
    let outputs = ReferenceRequest::standard(&storing(Expr::u32(7)), &[])
        .outputs()
        .expect("Fix: only subgroup expressions are refused in this build");
    assert_eq!(
        outputs[0],
        Value::from(7u32.to_le_bytes().to_vec()),
        "Fix: a store of a constant is that constant"
    );
}
