use super::*;
use crate::ir_inner::model::expr::Expr;

fn check_loop_back_edge(body: &[Node], errors: &mut Vec<ValidationError>) {
    super::check_loop_back_edge(body, &FxHashMap::default(), errors);
}

fn barrier() -> Node {
    Node::Barrier {
        ordering: MemoryOrdering::SeqCst,
    }
}

fn exit_guard() -> Node {
    Node::If {
        cond: Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        then: vec![Node::Return],
        otherwise: Vec::new(),
    }
}

/// A barrier inside a `Node::Block` still makes the loop collective.
///
/// Wrapping a phase in a block must not silence V055's trigger.
#[test]
fn a_barrier_inside_a_block_still_makes_the_loop_collective() {
    let mut errors = Vec::new();
    check_loop_back_edge(&[Node::Block(vec![barrier()]), exit_guard()], &mut errors);
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a barrier nested in a Block must still trigger the back-edge check"
    );
}

/// A guarding barrier inside an unconditional block orders the back edge.
#[test]
fn a_guarding_barrier_inside_a_block_is_credited() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[barrier(), exit_guard(), Node::Block(vec![barrier()])],
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "a Block executes unconditionally, so its barrier orders the back edge: {errors:?}"
    );
}

/// A conditional barrier after the exit cannot order lanes that skip its branch.
#[test]
fn a_guarding_barrier_inside_an_if_is_not_credited() {
    let mut errors = Vec::new();
    check_loop_back_edge(
        &[
            barrier(),
            exit_guard(),
            Node::If {
                cond: Expr::bool(true),
                then: vec![barrier()],
                otherwise: Vec::new(),
            },
        ],
        &mut errors,
    );
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a barrier only reached on one branch does not order the back edge"
    );
}

/// A nested loop barrier triggers collectiveness but cannot guard the back edge.
///
/// The nested loop may execute zero times, so its barrier is not unconditional.
#[test]
fn a_barrier_only_inside_a_nested_loop_triggers_but_never_guards() {
    let nested = Node::Loop {
        var: "inner".into(),
        from: Expr::u32(0),
        to: Expr::u32(0),
        body: vec![barrier()],
    };
    let mut errors = Vec::new();
    check_loop_back_edge(&[exit_guard(), nested], &mut errors);
    assert!(
        errors.iter().any(|error| error.code().as_str() == "V055"),
        "a nested-loop barrier makes the loop collective but cannot guard the back edge"
    );
}
