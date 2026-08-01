//! The DCE fixpoint program must not let an invocation leave a synchronizing
//! loop body after that body's last barrier.
//!
//! The program's iteration body ends with a barrier and then an early exit:
//! once a step adds no bit, lane 0 records convergence and the invocation
//! returns. For a while that exit was the LAST thing in the body, after the
//! final barrier. One invocation could then take the back edge and write while
//! a sibling had not yet reached the exit; the sibling left the kernel and
//! froze the data it owned partway through. Nothing hangs, because a barrier
//! does not count invocations that already returned, so the damage is to
//! ANSWERS rather than to liveness, and one workgroup is enough to produce it.
//! `V055` in `vyre-foundation` refuses exactly that shape.
//!
//! The bug stayed invisible for as long as it did because the program's
//! correctness argument rested on a nested `Node::Return` being emitted as
//! NOTHING: the loop ran its full iteration budget on device and a `converged`
//! flag, not the `Return`, made the early exit real. When a nested `Return`
//! started lowering to a real branch, that argument became false and the
//! program became illegal at the same moment, with no test anywhere between the
//! two crates to notice. So these tests deliberately do not test the emitter or
//! the lowering. They assert the property that has to hold no matter what a
//! `Return` lowers to: the built program VALIDATES, and its body's last
//! synchronizing node comes after its exit.
//!
//! The last four tests are negative controls. A suite that only asserts "the
//! program is clean" cannot tell a real fix from a validator that stopped
//! looking, so each control breaks the program in one specific way and proves
//! `V055` still fires: with the trailing barrier removed (the original bug),
//! with the barrier moved inside the convergence gate (the plausible WRONG fix,
//! which desynchronizes a workgroup whose lanes may read the flag stale), and
//! with the exit moved after the barrier again.
#![forbid(unsafe_code)]

use std::sync::Arc;

use vyre_foundation::ir::{validate, Node, Program};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_self_substrate::optimizer::dce_program::{
    build_dce_bfs_program, build_persistent_bfs_program,
};

/// Shape with enough nodes and edges that the CSR walk is really emitted.
fn shape() -> ProgramGraphShape {
    ProgramGraphShape::new(64, 256)
}

/// The DCE program (early exit, no sticky mirror).
fn dce_program() -> Program {
    build_dce_bfs_program(shape(), 8)
}

/// The sticky persistent-BFS program, which pushes an extra node into the same
/// body and so is the variant most likely to drift out of order.
fn sticky_program() -> Program {
    build_persistent_bfs_program(shape(), 8, u32::MAX)
}

/// Name of the persistent loop's induction variable.
///
/// The body is located by NAME rather than by position, for two reasons. The
/// program wraps its entry in a `Region`, so the loop is not a top-level node,
/// and the CSR step inside the body contains its own edge loop, so "the only
/// loop" is not unique. This picks the outer persistent loop, which is the one
/// the back-edge rule is about.
const PERSISTENT_LOOP_VAR: &str = "iter";

/// Body of the program's persistent loop.
fn loop_body(program: &Program) -> &[Node] {
    fn find<'a>(nodes: &'a [Node]) -> Option<&'a [Node]> {
        for node in nodes {
            let found = match node {
                Node::Loop { var, body, .. } if var.as_str() == PERSISTENT_LOOP_VAR => {
                    return Some(body.as_slice())
                }
                Node::Loop { body, .. } => find(body),
                Node::If {
                    then, otherwise, ..
                } => find(then).or_else(|| find(otherwise)),
                Node::Block(body) => find(body),
                Node::Region { body, .. } => find(body),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }

    find(program.entry()).unwrap_or_else(|| {
        panic!("no persistent loop named `{PERSISTENT_LOOP_VAR}` in the program entry")
    })
}

/// Run `edit` against the body of the persistent loop, in place.
fn edit_loop_body(program: &mut Program, edit: impl FnOnce(&mut Vec<Node>)) {
    fn walk(nodes: &mut [Node], edit: &mut Option<Box<dyn FnOnce(&mut Vec<Node>) + '_>>) -> bool {
        for node in nodes.iter_mut() {
            let done = match node {
                Node::Loop { var, body, .. } if var.as_str() == PERSISTENT_LOOP_VAR => {
                    let apply = edit.take().expect("the edit runs exactly once");
                    apply(body);
                    true
                }
                Node::Loop { body, .. } => walk(body, edit),
                Node::If {
                    then, otherwise, ..
                } => walk(then, edit) || walk(otherwise, edit),
                Node::Block(body) => walk(body, edit),
                Node::Region { body, .. } => {
                    let owned: &mut Vec<Node> = Arc::make_mut(body);
                    walk(owned, edit)
                }
                _ => false,
            };
            if done {
                return true;
            }
        }
        false
    }

    let mut slot: Option<Box<dyn FnOnce(&mut Vec<Node>) + '_>> = Some(Box::new(edit));
    assert!(
        walk(program.entry_mut(), &mut slot),
        "expected to edit the body of the persistent loop"
    );
}

/// Barriers at any nesting depth.
fn barriers_at_any_depth(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Barrier { .. } => 1,
            Node::If {
                then, otherwise, ..
            } => barriers_at_any_depth(then) + barriers_at_any_depth(otherwise),
            Node::Loop { body, .. } => barriers_at_any_depth(body),
            Node::Block(body) => barriers_at_any_depth(body),
            Node::Region { body, .. } => barriers_at_any_depth(body),
            _ => 0,
        })
        .sum()
}

/// Barriers that execute unconditionally, at the top level of `nodes`.
fn top_level_barriers(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .filter(|node| matches!(node, Node::Barrier { .. }))
        .count()
}

/// Name of a node kind, for failure messages only.
///
/// Local on purpose: the production namer is not public, and a test that
/// reports what it FOUND is much faster to diagnose than one reporting only
/// that a match failed.
fn node_name(node: &Node) -> &'static str {
    match node {
        Node::Barrier { .. } => "Barrier",
        Node::Return => "Return",
        Node::If { .. } => "If",
        Node::Loop { .. } => "Loop",
        Node::Store { .. } => "Store",
        Node::Let { .. } => "Let",
        Node::Block(_) => "Block",
        _ => "other",
    }
}

/// `Return` nodes at any nesting depth.
fn returns_at_any_depth(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Return => 1,
            Node::If {
                then, otherwise, ..
            } => returns_at_any_depth(then) + returns_at_any_depth(otherwise),
            Node::Loop { body, .. } => returns_at_any_depth(body),
            Node::Block(body) => returns_at_any_depth(body),
            Node::Region { body, .. } => returns_at_any_depth(body),
            _ => 0,
        })
        .sum()
}

/// Index of the top-level node that carries the early exit.
fn exit_index(body: &[Node]) -> usize {
    let found: Vec<usize> = body
        .iter()
        .enumerate()
        .filter(|(_, node)| returns_at_any_depth(std::slice::from_ref(*node)) > 0)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "expected exactly one top-level node to carry the early exit"
    );
    found[0]
}

/// Index of the last unconditional barrier in `body`.
fn last_top_level_barrier(body: &[Node]) -> usize {
    body.iter()
        .enumerate()
        .filter(|(_, node)| matches!(node, Node::Barrier { .. }))
        .map(|(index, _)| index)
        .next_back()
        .expect("the iteration body must contain at least one barrier")
}

fn messages(program: &Program) -> Vec<String> {
    validate(program)
        .iter()
        .map(|error| error.message().to_string())
        .collect()
}

fn v055_count(program: &Program) -> usize {
    messages(program)
        .iter()
        .filter(|message| message.contains("V055"))
        .count()
}

/// THE COUPLING GUARD. Locks out: a change to how `Return` or barriers lower
/// silently making this program illegal.
///
/// Read this one before changing anything about `Node::Return` lowering, loop
/// lowering, or barrier placement in any crate, because this is the test that
/// is supposed to stop you.
///
/// The program's early exit is legal ONLY because of the unconditional barrier
/// appended as the last node of its loop body. Its safety argument used to rest
/// on something else entirely: that a `Return` nested in a loop lowered to
/// NOTHING, so the exit was inert on device. That argument lived in a comment,
/// in one direction only, and nothing anywhere asserted it. When a nested
/// `Return` began lowering to a real branch, this program became illegal at that
/// instant and the news arrived as roughly 95 failures in a different crate,
/// pointing at a dispatch, naming neither the program nor the lowering change
/// that caused it.
///
/// So this test does not test lowering, and deliberately so. It asserts the
/// property that must hold whatever a `Return` compiles to: the built program
/// VALIDATES. A future change to `Return` or barrier lowering that invalidates
/// this program fails HERE, in the crate that owns the program, naming the rule
/// it broke, instead of downstream as unrelated dispatch failures.
///
/// It asserts the exact error list is empty rather than counting errors, so a
/// NEW rule this program violates also names itself here.
#[test]
fn dce_program_validates_with_no_errors() {
    let program = dce_program();
    assert_eq!(
        messages(&program),
        Vec::<String>::new(),
        "the DCE fixpoint program must validate clean"
    );
}

/// Locks out: the sticky variant regressing while the DCE variant stays clean.
///
/// Both public builders share one internal body builder, and the sticky path
/// pushes an extra node into that body. If a future edit appends the trailing
/// barrier on the DCE path only, or pushes the sticky mirror after the barrier,
/// this fails and the other test does not.
#[test]
fn sticky_persistent_program_validates_with_no_errors() {
    let program = sticky_program();
    assert_eq!(
        messages(&program),
        Vec::<String>::new(),
        "the sticky persistent-BFS program must validate clean"
    );
}

/// Locks out: the early exit becoming the last thing in the loop body again.
///
/// The structural form of the fix. The body's final node must be a barrier so
/// that an invocation cannot leave while a sibling is still inside the
/// iteration.
#[test]
fn iteration_body_ends_with_a_barrier_in_both_variants() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        let last = body.last().expect("the iteration body is not empty");
        assert!(
            matches!(last, Node::Barrier { .. }),
            "{label}: the iteration body must END with a barrier, found {}",
            node_name(last)
        );
    }
}

/// Locks out: reordering the exit after the final barrier.
///
/// Stronger than "the last node is a barrier": it pins the RELATIVE order that
/// makes the program legal, so inserting any further node after the exit but
/// before the barrier stays legal, while moving the exit down does not.
#[test]
fn the_early_exit_precedes_the_last_barrier() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        let exit = exit_index(body);
        let barrier = last_top_level_barrier(body);
        assert!(
            exit < barrier,
            "{label}: the exit at index {exit} must come BEFORE the body's last \
             barrier at index {barrier}"
        );
        assert_eq!(
            barrier,
            body.len() - 1,
            "{label}: the last barrier must be the final node of the body"
        );
    }
}

/// Locks out: "fixing" this by putting a barrier inside the convergence gate.
///
/// The gate is entered on a deliberately racy read of `converged`, so a lane
/// may see a stale value and take the other path. A barrier under that
/// condition would be reached by a subset of the workgroup, which desyncs it.
/// The barriers must all be unconditional.
#[test]
fn no_barrier_sits_inside_a_conditional_in_the_iteration_body() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        assert_eq!(
            barriers_at_any_depth(body),
            top_level_barriers(body),
            "{label}: every barrier in the iteration body must be unconditional \
             at body level; a nested one means some lanes can skip it"
        );
    }
}

/// Locks out: silently adding or dropping a barrier in the body.
///
/// Three, at real values: one after lane 0 zeroes the per-iteration flag, one
/// after the CSR step publishes it, and the trailing one that orders the exit
/// against the back edge. A count of two means the fix was reverted; four means
/// someone paid for an extra workgroup-wide sync without saying so.
#[test]
fn the_iteration_body_holds_exactly_three_barriers() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        assert_eq!(
            top_level_barriers(body),
            3,
            "{label}: expected exactly three unconditional barriers"
        );
    }
}

/// Locks out: making the program legal by deleting the early exit.
///
/// Removing the `Return` would also silence `V055`, and would silently restore
/// the full-budget loop the early exit exists to avoid: a measured 183x on a
/// 2000-node star. The exit must still be there, and still be under the
/// convergence condition rather than unconditional.
#[test]
fn the_early_exit_is_retained_and_stays_conditional() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        assert_eq!(
            returns_at_any_depth(body),
            1,
            "{label}: the body must keep exactly one early exit"
        );
        assert_eq!(
            body.iter()
                .filter(|node| matches!(node, Node::Return))
                .count(),
            0,
            "{label}: the exit must be nested under the convergence condition, \
             never a top-level unconditional return"
        );
    }
}

/// Locks out: the fix depending on the iteration budget.
///
/// The builder clamps the budget with `max(1)`, so 0 and 1 both produce a real
/// loop. Those are the boundary values a caller reaches with an empty or
/// single-node graph, and the ordering argument must hold there too.
#[test]
fn edge_iteration_budgets_still_validate() {
    for max_iters in [0_u32, 1, 2, 1024] {
        let program = build_dce_bfs_program(shape(), max_iters);
        assert_eq!(
            messages(&program),
            Vec::<String>::new(),
            "max_iters {max_iters} must validate clean"
        );
        let body = loop_body(&program);
        assert!(
            matches!(body.last(), Some(Node::Barrier { .. })),
            "max_iters {max_iters} must still end its body with a barrier"
        );
    }
}

/// Locks out: the fix holding only for one graph shape.
///
/// The body is built around `shape.node_count`, and the CSR walk is emitted
/// under a bounds condition derived from it. A degenerate shape must not shift
/// the trailing barrier out of place.
#[test]
fn assorted_graph_shapes_still_validate() {
    for (nodes, edges) in [(1_u32, 0_u32), (2, 1), (64, 256), (4096, 16384)] {
        let program = build_dce_bfs_program(ProgramGraphShape::new(nodes, edges), 8);
        assert_eq!(
            messages(&program),
            Vec::<String>::new(),
            "shape ({nodes}, {edges}) must validate clean"
        );
    }
}

/// NEGATIVE CONTROL. Locks out: a validator that stopped enforcing V055.
///
/// Removes the trailing barrier and nothing else, reproducing the exact shape
/// that failed roughly 95 tests, and requires the refusal to come back. If this
/// test ever passes silently, every other test in this file has become
/// decoration.
#[test]
fn removing_the_trailing_barrier_is_still_refused() {
    let mut program = dce_program();
    assert_eq!(v055_count(&program), 0, "the built program starts clean");
    edit_loop_body(&mut program, |body| {
        let last = body.pop().expect("body is not empty");
        assert!(
            matches!(last, Node::Barrier { .. }),
            "expected to remove the trailing barrier"
        );
    });
    assert_eq!(
        v055_count(&program),
        1,
        "with its trailing barrier gone the program must be refused; messages \
         were {:?}",
        messages(&program)
    );
}

/// NEGATIVE CONTROL. Locks out: the plausible wrong fix passing validation.
///
/// Moves the trailing barrier INSIDE the convergence gate, which is what
/// someone reaches for when they want the exit to be last. A barrier reached
/// only on the taken branch does not order the back edge, so the refusal must
/// stand. This is the check that makes the warning in the source comment
/// enforceable rather than advisory.
#[test]
fn a_barrier_moved_inside_the_convergence_gate_is_still_refused() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| {
        let barrier = body.pop().expect("body is not empty");
        assert!(matches!(barrier, Node::Barrier { .. }));
        let exit = body.pop().expect("body still holds the exit");
        match exit {
            Node::If {
                cond,
                mut then,
                otherwise,
            } => {
                then.insert(0, barrier);
                body.push(Node::If {
                    cond,
                    then,
                    otherwise,
                });
            }
            other => panic!("expected the exit to be an If, found {}", node_name(&other)),
        }
    });
    assert_eq!(
        v055_count(&program),
        1,
        "a barrier under the convergence condition must not satisfy the \
         back-edge rule; messages were {:?}",
        messages(&program)
    );
}

/// NEGATIVE CONTROL. Locks out: V055 accepting an exit that trails the barrier
/// by any distance.
///
/// Keeps every node, and only moves the exit to the very end. The program is
/// otherwise identical to the clean one, including barrier count, which proves
/// the rule is about ORDER and not about how many barriers a body contains.
#[test]
fn moving_the_exit_after_the_last_barrier_is_still_refused() {
    let mut program = dce_program();
    let clean_barriers = top_level_barriers(loop_body(&program));
    edit_loop_body(&mut program, |body| {
        let barrier = body.pop().expect("body is not empty");
        let exit = body.pop().expect("body still holds the exit");
        body.push(barrier);
        body.push(exit);
    });
    assert_eq!(
        top_level_barriers(loop_body(&program)),
        clean_barriers,
        "the reordering must not change the barrier count"
    );
    assert_eq!(
        v055_count(&program),
        1,
        "an exit after the last barrier must be refused however the body is \
         otherwise shaped; messages were {:?}",
        messages(&program)
    );
}

/// Pins the rule's CURRENT conservative reach, deliberately, and records the
/// one refinement this suite argues for.
///
/// Appends an unconditional `Return` after the trailing barrier. Every
/// invocation reaches that exit together, so no sibling can be left mid
/// iteration and the shape is harmless in fact. `V055` refuses it anyway,
/// because it asks only whether an invocation CAN return after the last
/// barrier, not whether the invocations agree about it. OBSERVED, and the
/// reason this test exists rather than the accepting version it started as.
///
/// That over-refusal is a real cost and is accepted for now: refusing a safe
/// program is recoverable in one node, while admitting an unsafe one corrupts
/// answers silently, and proving uniformity is a genuine analysis rather than a
/// predicate. The trailing barrier the DCE program carries is safe for exactly
/// the reason this test's shape is safe (the exit condition is settled by the
/// preceding barrier, so it is workgroup-uniform), which makes that program the
/// motivating example for a uniformity carve-out.
///
/// If that carve-out ever lands, this test is the one that must change, and it
/// should change to assert acceptance rather than being deleted, so the
/// boundary stays described either way.
#[test]
fn even_a_uniform_exit_after_the_barrier_is_refused_today() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| body.push(Node::Return));
    assert_eq!(
        v055_count(&program),
        1,
        "V055 is conservative by design: it refuses any exit after the last \
         barrier, including this uniform one; messages were {:?}",
        messages(&program)
    );
}
