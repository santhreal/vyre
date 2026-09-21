//! Node lowering unit tests.

use super::super::lower;
use crate::descriptor::{KernelBody, KernelOp, KernelOpKind};
use crate::lower::loop_site::find_loop;
use vyre_foundation::ir::{BufferAccess, DataType, Program};

#[test]
fn loop_variable_lowers_to_child_loop_index_result() {
    use vyre_foundation::ir::{BufferDecl, Expr, Node};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::buf_len("input"),
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::load("input", Expr::var("i")),
            )],
        )],
    );

    let desc = lower(&program).expect("Fix: loop variable must descriptor-lower");
    assert!(crate::verify::verify(&desc).is_ok());
    let (loop_body, loop_op) =
        find_loop(&desc.body).expect("Fix: structured loop op must be present");
    let child = &loop_body.child_bodies[loop_op.operands[2] as usize];
    assert!(
        matches!(
            child.ops.first().map(|op| &op.kind),
            Some(KernelOpKind::LoopIndex { loop_var }) if loop_var.as_ref() == "i"
        ),
        "loop body must materialize the induction value before lowering input[i]"
    );
}

#[test]
fn loop_variable_does_not_clobber_same_named_outer_binding() {
    use vyre_foundation::ir::{BufferDecl, Expr, Node};

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("i", Expr::u32(9)),
            Node::loop_for("i", Expr::u32(0), Expr::u32(1), vec![]),
            Node::store("out", Expr::u32(0), Expr::var("i")),
        ],
    );

    let desc = lower(&program).expect("Fix: shadowed loop variable must descriptor-lower");
    assert!(crate::verify::verify(&desc).is_ok());
    let store = find_store(&desc.body).expect("Fix: post-loop store must be present");
    assert_eq!(
        store.operands[2], 0,
        "post-loop read must use the outer i binding, not the loop induction result"
    );

    fn find_store(body: &KernelBody) -> Option<&KernelOp> {
        body.ops
            .iter()
            .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
            .or_else(|| body.child_bodies.iter().find_map(find_store))
    }
}

#[test]
fn if_else_branches_lower_from_the_same_incoming_scope() {
    use vyre_foundation::ir::{BufferDecl, Expr, Node};

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::u32(1)),
            Node::if_then_else(
                Expr::bool(true),
                vec![Node::assign("x", Expr::add(Expr::var("x"), Expr::u32(1)))],
                vec![Node::store("out", Expr::u32(0), Expr::var("x"))],
            ),
        ],
    );

    let desc = lower(&program).expect("Fix: if/else must descriptor-lower");
    assert!(crate::verify::verify(&desc).is_ok());
    let (_, if_op) = find_if_else(&desc.body).expect("Fix: if/else op must be present");
    let parent = find_parent_body_containing_op(&desc.body, if_op as *const KernelOp)
        .expect("Fix: if op parent body must be found");
    let else_body = &parent.child_bodies[if_op.operands[2] as usize];
    let else_store = else_body
        .ops
        .iter()
        .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
        .expect("Fix: else branch must contain the store");
    let else_carrier = else_body
        .ops
        .iter()
        .find(|op| matches!(&op.kind, KernelOpKind::LoopCarrier { name } if name.as_ref() == "x"))
        .expect("Fix: else branch must read x through the if carrier seeded from incoming scope");
    let else_carrier_id = else_carrier
        .result
        .expect("Fix: else carrier read must produce an SSA result");
    assert_eq!(
        else_store.operands[2], else_carrier_id,
        "else branch must read the incoming x through its carrier, not the result assigned only by then"
    );

    fn find_if_else(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
        for op in &body.ops {
            if matches!(op.kind, KernelOpKind::StructuredIfThenElse) {
                return Some((body, op));
            }
        }
        body.child_bodies.iter().find_map(find_if_else)
    }

    fn find_parent_body_containing_op(
        body: &KernelBody,
        target: *const KernelOp,
    ) -> Option<&KernelBody> {
        if body.ops.iter().any(|op| std::ptr::eq(op, target)) {
            return Some(body);
        }
        body.child_bodies
            .iter()
            .find_map(|child| find_parent_body_containing_op(child, target))
    }
}
