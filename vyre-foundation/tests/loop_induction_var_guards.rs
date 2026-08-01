//! Loop passes must decline when the body can redefine the induction variable.
//!
//! `loop_peel` lifts iteration 0 by substituting `var := 0` into the guard's
//! `then` and the trailing body. `loop_var_range_fold` folds an `If` whose
//! condition is decided by the loop's constant range. Both are only sound while
//! `var` still denotes the loop counter everywhere the rewrite reaches: a `Let`
//! or `Assign` naming `var` inside the body makes a later `Var(var)` denote that
//! new binding, and the rewrite would substitute the wrong value into it.
//!
//! The two guards behind that, `body_writes_loop_var` and `body_rebinds_var`,
//! used to be covered by a `#[cfg(test)]` module inside
//! `src/optimizer/passes/loops/substitution.rs`, which the organization contract
//! forbids for new code in `vyre-foundation/src`. These tests cover the same
//! cases through the public pass entry points, so they check the behavior the
//! guards exist for rather than the helpers themselves: each one pairs a case
//! that must rewrite with the same shape carrying a redefinition, which must
//! not.
#![forbid(unsafe_code)]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_peel::LoopPeelPass;
use vyre_foundation::optimizer::passes::loops::loop_var_range_fold::LoopVarRangeFoldPass;

/// One `u32` output buffer, so a store has somewhere to land.
fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(8)],
        [1, 1, 1],
        entry,
    )
}

/// `for i in 0..4 { if i == 0 { <then> } <rest> }`, the shape `loop_peel` peels.
fn peelable(then: Vec<Node>, rest: Vec<Node>) -> Program {
    let mut body = vec![Node::If {
        cond: Expr::eq(Expr::var("i"), Expr::u32(0)),
        then,
        otherwise: Vec::new(),
    }];
    body.extend(rest);
    program(vec![Node::Loop {
        var: "i".into(),
        from: Expr::u32(0),
        to: Expr::u32(4),
        body,
    }])
}

/// The peel fires on the plain shape.
///
/// The control for every decline case below: without it, a guard that rejected
/// everything would make them all pass.
#[test]
fn loop_peel_lifts_iteration_zero_when_the_body_leaves_the_counter_alone() {
    let result = LoopPeelPass::transform(peelable(
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        vec![Node::store("out", Expr::u32(1), Expr::var("i"))],
    ));

    assert!(
        result.changed,
        "Fix: the peelable shape must peel, or the decline cases below prove nothing."
    );
}

/// A `Let` naming the counter inside the guard's `then` blocks the peel.
///
/// The peel substitutes `i := 0` into `then`. If `then` binds its own `i`, a
/// later `Var(i)` reads that binding, and substituting 0 into it silently
/// changes what the program computes.
#[test]
fn loop_peel_declines_when_the_guard_body_binds_the_counter() {
    let result = LoopPeelPass::transform(peelable(
        vec![
            Node::let_bind("i", Expr::u32(7)),
            Node::store("out", Expr::u32(0), Expr::var("i")),
        ],
        vec![Node::store("out", Expr::u32(1), Expr::var("i"))],
    ));

    assert!(
        !result.changed,
        "Fix: a `Let` naming the loop counter inside the guard body must block the peel."
    );
}

/// An `Assign` to the counter from inside a differently named inner loop is
/// still a write to the counter.
///
/// The inner loop introduces `j`, not `i`, so it does not scope `i`: the
/// assignment reaches the outer counter and the peel must decline. A guard that
/// stopped descending at the first nested loop would miss this.
#[test]
fn loop_peel_declines_when_an_inner_loop_assigns_the_outer_counter() {
    let result = LoopPeelPass::transform(peelable(
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        vec![Node::Loop {
            var: "j".into(),
            from: Expr::u32(0),
            to: Expr::u32(2),
            body: vec![Node::assign("i", Expr::var("j"))],
        }],
    ));

    assert!(
        !result.changed,
        "Fix: an assignment to the outer counter inside an inner loop over a different variable \
         must block the peel."
    );
}

/// A write nested under `If` and `Block` is still a write.
#[test]
fn loop_peel_declines_when_the_write_is_nested_under_if_and_block() {
    let result = LoopPeelPass::transform(peelable(
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        vec![Node::If {
            cond: Expr::eq(Expr::var("i"), Expr::u32(2)),
            then: vec![Node::Block(vec![Node::let_bind("i", Expr::u32(0))])],
            otherwise: Vec::new(),
        }],
    ));

    assert!(
        !result.changed,
        "Fix: the write scan must descend through If and Block."
    );
}

/// `for i in 0..4 { if <cond> { store } }`, the shape `loop_var_range_fold`
/// decides from the loop's constant range.
fn range_foldable(extra: Vec<Node>) -> Program {
    let mut body = vec![Node::If {
        cond: Expr::lt(Expr::var("i"), Expr::u32(4)),
        then: vec![Node::store("out", Expr::u32(0), Expr::var("i"))],
        otherwise: Vec::new(),
    }];
    body.extend(extra);
    program(vec![Node::Loop {
        var: "i".into(),
        from: Expr::u32(0),
        to: Expr::u32(4),
        body,
    }])
}

/// The fold fires when the counter keeps its meaning through the body.
#[test]
fn range_fold_decides_a_condition_from_the_constant_loop_range() {
    let result = LoopVarRangeFoldPass::transform(range_foldable(Vec::new()));

    assert!(
        result.changed,
        "Fix: `i < 4` inside `for i in 0..4` is decided by the range and must fold, or the \
         decline case below proves nothing."
    );
}

/// A nested loop reusing the counter's name blocks the fold.
///
/// The inner `Loop` scopes its own `i`, so it does not write the outer counter,
/// but it does reintroduce the name. Range-folding a condition on `i` inside
/// that inner loop would apply the OUTER range to a different variable, which is
/// why the fold uses a rebind check rather than the write check the peel uses.
#[test]
fn range_fold_declines_when_a_nested_loop_reuses_the_counter_name() {
    let result = LoopVarRangeFoldPass::transform(range_foldable(vec![Node::Loop {
        var: "i".into(),
        from: Expr::u32(0),
        to: Expr::u32(2),
        body: vec![Node::store("out", Expr::u32(1), Expr::var("i"))],
    }]));

    assert!(
        !result.changed,
        "Fix: a nested loop reusing the counter name must block the range fold."
    );
}

/// A rebind nested under `If` and `Block` blocks the fold.
#[test]
fn range_fold_declines_when_the_rebind_is_nested_under_if_and_block() {
    let result = LoopVarRangeFoldPass::transform(range_foldable(vec![Node::If {
        cond: Expr::lt(Expr::var("i"), Expr::u32(2)),
        then: vec![Node::Block(vec![Node::let_bind("i", Expr::u32(0))])],
        otherwise: Vec::new(),
    }]));

    assert!(
        !result.changed,
        "Fix: the rebind scan must descend through If and Block."
    );
}
