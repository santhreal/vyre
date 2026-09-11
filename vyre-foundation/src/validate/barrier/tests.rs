use super::*;
use crate::ir_inner::model::expr::Expr;

fn check_loop_back_edge(body: &[Node], errors: &mut Vec<ValidationError>) {
    super::check_loop_back_edge(body, &FxHashMap::default(), errors);
}

#[test]
fn divergent_barrier_emits_v010() {
    let mut errors = Vec::new();
    check_barrier(true, MemoryOrdering::SeqCst, &mut errors);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].code().as_str() == "V010");
}

#[test]
fn uniform_barrier_is_valid() {
    let mut errors = Vec::new();
    check_barrier(false, MemoryOrdering::SeqCst, &mut errors);
    assert!(errors.is_empty());
}

#[test]
fn relaxed_barrier_is_rejected() {
    let mut errors = Vec::new();
    check_barrier(false, MemoryOrdering::Relaxed, &mut errors);
    assert!(errors.iter().any(|error| error.code().as_str() == "V043"));
}

fn barrier() -> Node {
    Node::Barrier {
        ordering: MemoryOrdering::SeqCst,
    }
}

/// The exact unsafe twin of `fixpoint::persistent_fixpoint`: a synchronizing
/// loop whose last node is a lane-dependent early exit.
#[test]
fn exit_after_the_last_barrier_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                vec![Node::Return],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a lane-dependent early exit after the last barrier must be refused"
    );
}

/// A barrier after the exit orders it against the back edge.
#[test]
fn a_barrier_after_the_exit_is_accepted() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::if_then(Expr::bool(true), vec![Node::Return]),
            barrier(),
        ],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "a barrier on the back edge discharges the obligation: {errors:?}"
    );
}

/// Logical barriers trigger and discharge the same loop back-edge obligation.
#[test]
fn logical_barrier_orders_a_logical_lane_exit() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            Node::logical_barrier(MemoryOrdering::SeqCst),
            Node::if_then(
                Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
                vec![Node::Return],
            ),
            Node::logical_barrier(MemoryOrdering::SeqCst),
        ],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "a logical barrier on the back edge discharges the logical exit obligation: {errors:?}"
    );
}

/// A loop with no barrier has no cross-invocation communication to order.
#[test]
fn an_exit_in_a_loop_with_no_barrier_is_accepted() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[Node::if_then(Expr::bool(true), vec![Node::Return])],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "a loop with no barrier has no collective contract to break: {errors:?}"
    );
}

/// A nested return still ends participation in the outer loop's barriers.
#[test]
fn a_nested_exit_after_the_last_barrier_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::if_then(
                    Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    vec![Node::Return],
                )],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a nested lane-dependent exit still leaves the outer loop's barriers"
    );
}

/// A nested loop whose return guard becomes lane-dependent in a later iteration must emit V055.
#[test]
fn nested_loop_loop_carried_divergence_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(0)), vec![Node::Return]),
                    Node::assign("x", Expr::InvocationId { axis: 0 }),
                ],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "loop-carried lane-dependent state reaching return guard on later iteration must trigger V055: {errors:?}"
    );
}

/// A nested loop where the return guard stays uniform across all iterations is accepted.
#[test]
fn nested_loop_purely_uniform_exit_is_accepted() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(10)), vec![Node::Return]),
                    Node::assign("x", Expr::add(Expr::var("x"), Expr::u32(1))),
                ],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "purely uniform nested loop exit must be accepted without V055: {errors:?}"
    );
}

/// A nested loop with divergent bounds makes modified variables divergent after the loop.
#[test]
fn nested_loop_divergent_bounds_modifying_variable_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::InvocationId { axis: 0 },
                vec![Node::assign("x", Expr::u32(1))],
            ),
            Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(1)), vec![Node::Return]),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "nested loop with divergent bounds must taint modified variable as divergent: {errors:?}"
    );
}

/// A nested loop with divergent inner branch makes modified variable divergent after the loop.
#[test]
fn nested_loop_divergent_inner_branch_modifying_variable_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::if_then(
                    Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    vec![Node::assign("x", Expr::u32(1))],
                )],
            ),
            Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(1)), vec![Node::Return]),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "nested loop with divergent inner branch modifying variable must trigger V055: {errors:?}"
    );
}

/// A doubly-nested loop with inner divergent bounds triggers V055 for outer exit.
#[test]
fn doubly_nested_loop_divergent_inner_bounds_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "mid",
                Expr::u32(0),
                Expr::u32(2),
                vec![Node::loop_for(
                    "inner",
                    Expr::u32(0),
                    Expr::InvocationId { axis: 0 },
                    vec![Node::assign("x", Expr::u32(1))],
                )],
            ),
            Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(1)), vec![Node::Return]),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "doubly nested loop with inner divergent bounds must trigger V055: {errors:?}"
    );
}

/// A nested loop after the last barrier where the return guard only matches on a later iteration
/// after loop-carried lane-dependent state has been assigned must emit V055.
#[test]
fn nested_loop_later_iteration_divergent_return_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::if_then(Expr::eq(Expr::var("x"), Expr::u32(1)), vec![Node::Return]),
                    Node::assign("x", Expr::InvocationId { axis: 0 }),
                ],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "nested loop reaching divergent return on later iteration must trigger V055: {errors:?}"
    );
}

/// A doubly-nested loop after the last barrier where lane-dependent state assigned in the inner
/// loop causes an outer or inner return guard to diverge on a later iteration must emit V055.
#[test]
fn doubly_nested_loop_later_iteration_divergent_return_emits_v055() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            Node::let_bind("x", Expr::u32(0)),
            Node::loop_for(
                "outer",
                Expr::u32(0),
                Expr::u32(3),
                vec![
                    Node::loop_for(
                        "inner",
                        Expr::u32(0),
                        Expr::u32(3),
                        vec![Node::if_then(
                            Expr::eq(Expr::var("x"), Expr::u32(2)),
                            vec![Node::Return],
                        )],
                    ),
                    Node::assign(
                        "x",
                        Expr::add(Expr::var("x"), Expr::InvocationId { axis: 0 }),
                    ),
                ],
            ),
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "doubly nested loop reaching divergent return on later iteration must trigger V055: {errors:?}"
    );
}
