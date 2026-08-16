//! V055 must distinguish collective fixpoint exits from lane-dependent exits.
//!
//! The DCE iteration ends with an early exit after a barrier. Its condition
//! loads `changed[0]`, the same address in every lane, after the barrier settles
//! all writes to that word. No write intervenes. Every lane therefore returns
//! together or every lane takes the back edge together, so a second trailing
//! barrier would add synchronization without adding safety.
//!
//! These tests couple the producer to the validator contract. Both public
//! builders must validate with the exit as the final body node and exactly two
//! barriers. Negative controls replace the settled condition with a
//! lane-dependent predicate, dirty the settled word before the exit, and place
//! a barrier only on a conditional path. V055 must reject every unsafe twin.
#![forbid(unsafe_code)]

use std::sync::Arc;

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::validate::validate;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_pass_engine::optimizer::dce_program::{
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

type LoopBodyEdit<'a> = Box<dyn FnOnce(&mut Vec<Node>) + 'a>;

/// Run `edit` against the body of the persistent loop, in place.
fn edit_loop_body(program: &mut Program, edit: impl FnOnce(&mut Vec<Node>)) {
    fn walk(nodes: &mut [Node], edit: &mut Option<LoopBodyEdit<'_>>) -> bool {
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

    let mut slot: Option<LoopBodyEdit<'_>> = Some(Box::new(edit));
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
    validate(program)
        .iter()
        .filter(|error| error.code().as_str() == "V055")
        .count()
}

/// THE COUPLING GUARD. Locks out: a change to how `Return` or barriers lower
/// silently making this program illegal.
///
/// Read this one before changing anything about `Node::Return` lowering, loop
/// lowering, or barrier placement in any crate, because this is the test that
/// is supposed to stop you.
///
/// The program's early exit is legal because the validator derives a
/// collective condition from the acquiring barrier, the same-address
/// `changed[0]` load, and the absence of an intervening write. Its safety
/// argument used to rest on a nested `Node::Return` emitting no device code.
/// When nested returns became real branches, a trailing barrier was added as a
/// conservative workaround.
///
/// This test asserts the stronger current contract: the built program validates
/// after that workaround is removed. A change to return lowering, barrier
/// ordering, or uniformity analysis now fails in the crate that owns the
/// program instead of surfacing as unrelated dispatch failures downstream.
///
/// It asserts the exact error list is empty so any new violated rule names
/// itself here.
#[test]
fn dce_program_validates_with_no_errors() {
    let program = dce_program();
    assert_eq!(
        messages(&program),
        Vec::<String>::new(),
        "the DCE fixpoint program must validate clean"
    );
}

/// The sticky variant must retain the same collective proof.
///
/// It inserts one extra atomic mirror before the settling barrier. If that
/// mirror drifts after the barrier, the settled-load proof must fail here.
#[test]
fn sticky_persistent_program_validates_with_no_errors() {
    let program = sticky_program();
    assert_eq!(
        messages(&program),
        Vec::<String>::new(),
        "the sticky persistent-BFS program must validate clean"
    );
}

/// The uniform exit is intentionally the final iteration-body node.
///
/// This proves the validator's uniformity carve-out subsumes the old trailing
/// barrier workaround rather than retaining it under a different comment.
#[test]
fn iteration_body_ends_with_the_collective_exit_in_both_variants() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        let exit = exit_index(body);
        assert_eq!(
            exit,
            body.len() - 1,
            "{label}: the collective exit must be the final body node"
        );
        assert!(
            returns_at_any_depth(std::slice::from_ref(&body[exit])) == 1,
            "{label}: final node must carry exactly one return"
        );
    }
}

/// The acquiring barrier must immediately precede the collective exit.
///
/// Any intervening write to `changed` invalidates the settled-load proof and
/// must reactivate V055.
#[test]
fn the_collective_exit_follows_the_last_barrier() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        let exit = exit_index(body);
        let barrier = last_top_level_barrier(body);
        assert_eq!(
            barrier + 1,
            exit,
            "{label}: no node may intervene between the settling barrier and exit"
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

/// The loop pays only the two barriers required by its data dependencies.
///
/// One follows lane 0 clearing the iteration flag. One settles all atomic
/// updates before the collective exit reads the flag.
#[test]
fn the_iteration_body_holds_exactly_two_barriers() {
    for (label, program) in [("dce", dce_program()), ("sticky", sticky_program())] {
        let body = loop_body(&program);
        assert_eq!(
            top_level_barriers(body),
            2,
            "{label}: expected exactly two unconditional barriers"
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

/// Boundary iteration budgets retain the collective-exit proof.
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
        assert_eq!(exit_index(body), body.len() - 1);
    }
}

/// Locks out: the fix holding only for one graph shape.
///
/// The body is built around `shape.node_count`, and the CSR walk is emitted
/// under a bounds condition derived from it. A degenerate shape must retain the
/// same barrier-settled exit proof.
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

/// A lane-dependent exit after the settling barrier must still trigger V055.
///
/// This is the direct negative twin of the production condition. A validator
/// that accepted it would allow only part of a workgroup to leave.
#[test]
fn lane_dependent_exit_is_refused() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| {
        let exit = exit_index(body);
        let Node::If { cond, .. } = &mut body[exit] else {
            panic!("collective exit must remain an If");
        };
        *cond = Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0));
    });
    assert_eq!(
        v055_count(&program),
        1,
        "lane-dependent exit must be refused: {:?}",
        messages(&program)
    );
}

/// A write after the barrier invalidates the settled-load proof.
///
/// Even an identical scalar store is conservatively rejected because it races
/// the exit load without another acquiring barrier.
#[test]
fn write_after_barrier_invalidates_uniform_exit() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| {
        let exit = exit_index(body);
        body.insert(exit, Node::store("changed", Expr::u32(0), Expr::u32(0)));
    });
    assert_eq!(
        v055_count(&program),
        1,
        "dirty settled word must reactivate V055: {:?}",
        messages(&program)
    );
}

/// A barrier on only one path cannot rescue a lane-dependent exit.
///
/// The back-edge guard credits only unconditional barriers, so this plausible
/// workaround remains rejected.
#[test]
fn conditional_barrier_does_not_rescue_lane_dependent_exit() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| {
        let exit = exit_index(body);
        let Node::If { cond, .. } = &mut body[exit] else {
            panic!("collective exit must remain an If");
        };
        *cond = Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0));
        body.push(Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![Node::barrier()],
        ));
    });
    assert_eq!(
        v055_count(&program),
        1,
        "conditional barrier must not satisfy V055: {:?}",
        messages(&program)
    );
}

/// An unconditional return after an acquiring barrier is collective by shape.
///
/// This locks the refined boundary: V055 rejects disagreement between lanes,
/// not harmless exits that every lane reaches together.
#[test]
fn unconditional_exit_after_barrier_is_accepted() {
    let mut program = dce_program();
    edit_loop_body(&mut program, |body| body.push(Node::Return));
    assert_eq!(
        v055_count(&program),
        0,
        "uniform unconditional exit must be accepted: {:?}",
        messages(&program)
    );
}

fn pack_words(words: &[u32]) -> vyre_reference::value::Value {
    vyre_reference::value::Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn decode_words(value: &vyre_reference::value::Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte u32 output")))
        .collect()
}

fn execute_single_node_dce(reversed: bool) -> Vec<Vec<u32>> {
    let program = build_dce_bfs_program(ProgramGraphShape::new(1, 0), 8);
    let inputs = vec![
        pack_words(&[0]),
        pack_words(&[0, 0]),
        pack_words(&[0]),
        pack_words(&[0]),
        pack_words(&[0]),
        pack_words(&[1]),
        pack_words(&[0]),
        pack_words(&[0]),
        pack_words(&[0]),
    ];
    let outputs = if reversed {
        vyre_reference::reference_eval_lane_reversed(&program, &inputs)
    } else {
        vyre_reference::reference_eval(&program, &inputs)
    }
    .expect("Fix: the reference interpreter must execute the DCE fixpoint");

    ["frontier_out", "changed", "converged"]
        .iter()
        .map(|name| {
            let index = vyre_reference::output_index(&program, name)
                .unwrap_or_else(|| panic!("Fix: DCE output `{name}` must remain declared"));
            decode_words(&outputs[index])
        })
        .collect()
}

/// Removing the trailing barrier must preserve the production fixpoint result.
///
/// A one-node closure converges on its first pass. Forward and reversed lane
/// schedules must both keep the seeded frontier, report no change, and publish
/// convergence. This executes the real emitted Program rather than only
/// inspecting its validation shape.
#[test]
fn barrier_elision_preserves_exact_dce_fixpoint_execution() {
    let expected = vec![vec![1], vec![0], vec![1]];
    assert_eq!(execute_single_node_dce(false), expected);
    assert_eq!(execute_single_node_dce(true), expected);
}
