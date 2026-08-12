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
            Node::If {
                cond: Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                then: vec![Node::Return],
                otherwise: Vec::new(),
            },
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
            Node::If {
                cond: Expr::bool(true),
                then: vec![Node::Return],
                otherwise: Vec::new(),
            },
            barrier(),
        ],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "a barrier on the back edge discharges the obligation: {errors:?}"
    );
}

/// A loop with no barrier has no cross-invocation communication to order.
#[test]
fn an_exit_in_a_loop_with_no_barrier_is_accepted() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[Node::If {
            cond: Expr::bool(true),
            then: vec![Node::Return],
            otherwise: Vec::new(),
        }],
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
            Node::Loop {
                var: "inner".into(),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![Node::If {
                    cond: Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    then: vec![Node::Return],
                    otherwise: Vec::new(),
                }],
            },
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a nested lane-dependent exit still leaves the outer loop's barriers"
    );
}
