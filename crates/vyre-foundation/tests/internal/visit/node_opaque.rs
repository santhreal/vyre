//! Integration test crate for the containing Vyre package.

use super::*;

vyre_test_support::test_expr_extension!(
    TestOpaqueExpr,
    kind: "test.opaque_expr",
    identity: "test",
    result_type: None,
    cse_safe: false,
    fingerprint: 7,
);

struct CountingNodeVisitor {
    count: usize,
}

impl NodeVisitor for CountingNodeVisitor {
    type Break = Infallible;

    node_visitor_uniform_arms!(
        |this| {
            this.count += 1;
            Continue(())
        },
        visit_let,
        visit_assign,
        visit_store,
        visit_if,
        visit_loop,
        visit_indirect_dispatch,
        visit_async_load,
        visit_async_store,
        visit_async_wait,
        visit_trap,
        visit_resume,
        visit_return,
        visit_barrier,
        visit_logical_barrier,
        visit_collective,
        visit_tile,
        visit_block,
        visit_region,
        visit_opaque_node,
    );
}

#[test]
fn node_preorder_visits_nested_nodes() {
    let node = Node::if_then(
        Expr::bool(true),
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(2),
            vec![Node::return_()],
        )],
    );
    let mut visitor = CountingNodeVisitor { count: 0 };
    visit_node_preorder(&mut visitor, &node);
    assert_eq!(visitor.count, 3);
}

#[test]
fn node_visitors_use_inline_stack_for_shallow_trees_and_survive_deep_trees() {
    let mut node = Node::return_();
    for _ in 0..4096 {
        node = Node::Block(vec![node]);
    }
    let mut visitor = CountingNodeVisitor { count: 0 };

    visit_node_postorder(&mut visitor, &node);

    assert_eq!(visitor.count, 4097);
}

#[test]
fn expr_entry_point_handles_opaque_expr_explicitly() {
    let expr = Expr::Opaque(Arc::new(TestOpaqueExpr));
    let mut visitor = CountingExprVisitor { count: 0 };
    visit_expr(&mut visitor, &expr);
    assert_eq!(visitor.count, 1);
}

// ------------------------------------------------------------------
// Adversarial ControlFlow::Break tests for F-IR visitor exhaustiveness.
// ------------------------------------------------------------------
