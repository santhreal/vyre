//! Every operand slot a statement owns must have its calls expanded.
//!
//! WHY: the caller-side inline walk used to enumerate `Node` itself, and its
//! arms for `AsyncLoad`, `AsyncStore` and `Trap` cloned the offset, the size and
//! the trap address verbatim. A call in one of those positions therefore
//! survived a pass whose whole contract is that no call reaches a backend. Under
//! `UnresolvedCalls::Reject` that is the exact case the pass exists to refuse,
//! and it refused nothing. The callee-side walk expanded those positions, so the
//! two sides disagreed about which positions hold an expression.
//!
//! The slot set comes from `vyre_test_support::ir_variants::node_operand_samples`
//! at run time, so a variant that gains an operand slot is covered here without
//! anyone editing this file. A hand-written list would report a clean pass for
//! the slot it forgot, which is the failure this test exists to catch.
//!
//! Not covered: whether the expansion is correct. The value a callee produces is
//! pinned by the inline value tests; this pins that the call is gone.

use std::ops::ControlFlow;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::transform::inline::inline_calls_with_resolver;
use vyre_foundation::transform::visit::try_for_each_expr;
use vyre_test_support::ir_variants::node_operand_samples;

/// `leaf_op(src) = src[0]`, the smallest inlinable composition.
fn leaf_callee() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::output("result", 1, DataType::U32).with_count(64),
        ],
        [1, 1, 1],
        vec![Node::store(
            "result",
            Expr::u32(0),
            Expr::load("src", Expr::u32(0)),
        )],
    )
}

fn resolver(op_id: &str) -> Option<Program> {
    (op_id == "leaf_op").then(leaf_callee)
}

/// Every op id an `Expr::Call` still names, anywhere in `program`.
///
/// Descent is `try_for_each_expr`, the exhaustive owner, so a call reached
/// through a position this test did not think of still counts.
fn calls_left(program: &Program) -> Vec<String> {
    let mut found = Vec::new();
    try_for_each_expr(program.entry(), |expr| {
        if let Expr::Call { op_id, .. } = expr {
            found.push(op_id.to_string());
        }
        ControlFlow::<()>::Continue(())
    });
    found
}

#[test]
fn a_call_in_any_operand_slot_is_expanded() {
    let call = Expr::call("leaf_op", vec![Expr::load("caller_src", Expr::u32(0))]);
    let samples = node_operand_samples(&call);
    assert!(
        samples.len() >= 8,
        "the operand slot enumeration went empty, so this test would pass without checking anything"
    );

    for sample in samples {
        let caller = Program::wrapped(
            vec![
                BufferDecl::storage("caller_src", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(64),
                BufferDecl::output("caller_out", 1, DataType::U32).with_count(64),
            ],
            [1, 1, 1],
            vec![sample.node.clone()],
        );

        let inlined = inline_calls_with_resolver(&caller, resolver)
            .unwrap_or_else(|error| panic!("{} refused inlining: {error}", sample.label()));

        assert_eq!(
            calls_left(&inlined),
            Vec::<String>::new(),
            "{} kept a call after inlining, so that operand slot is never expanded",
            sample.label()
        );
    }
}
