//! Contract tests for `fixpoint::persistent_fixpoint::persistent_fixpoint`, the
//! in-kernel `Node::Loop` form, and specifically for the ordering obligation on
//! its LOOP BACK EDGE.
//!
//! # The defect these lock out
//!
//! The builder emits, per iteration: clear `changed[0]` (invocation 0 only),
//! barrier, caller transfer body, per-word compare that `atomic_or`s
//! `changed[0]`, barrier, then `if changed[0] == 0 { Return }`.
//!
//! For a long time nothing separated that final READ from the NEXT iteration's
//! CLEAR of the same word. The two barriers the builder emitted sat
//! consecutively BEFORE the read, so the second was a no-op and the back edge
//! was unguarded. The warp holding invocation 0 could take the back edge and
//! clear `changed[0]` while a sibling warp of the same workgroup had not yet
//! executed the read; that warp then read 0, took the `Return`, and left the
//! kernel while the rest kept iterating.
//!
//! That is a PARTIAL EXIT, and its consequences are the reason these tests are
//! structural rather than statistical:
//!
//! - The invocations that left stop running `transfer_body`, so the words they
//!   own freeze mid-transfer. The dispatch returns a partially-transferred
//!   state.
//! - Nothing hangs. `bar.sync` does not count invocations that have already
//!   returned, so the survivors sail through every later barrier. The defect
//!   costs ANSWERS and never liveness, which is exactly why it survived.
//! - It needs no second workgroup and no host concurrency. It is a race between
//!   warps of ONE workgroup, so `words <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`
//!   does not make it safe, though the builder's doc used to say it did.
//!
//! It reached a consumer. `exatok`'s BPE merge loop composes this builder at
//! one workgroup and saw nondeterministic partial merges: a pretoken left with
//! two unmerged tokens, reported honestly by its own per-segment flags, with the
//! device pass counter BELOW the budget because the loop had exited early rather
//! than run out of passes.
//!
//! # Why structural and not a convergence run
//!
//! A partial exit is a warp-scheduling race. Reproducing it end to end took
//! tens of full-suite runs to fail a handful of times, so a convergence test
//! would pass almost always while the defect was present. The ordering
//! obligation, in contrast, is visible in the emitted IR and holds or does not
//! hold every single time. These tests therefore assert the SHAPE that makes the
//! exit collective, which is the property the emitter's uniformity proof
//! silently depends on.
#![cfg(feature = "fixpoint")]

use vyre_foundation::ir::{AtomicOp, Expr, MemoryOrdering, Node, Program};

use vyre_libs::fixpoint::persistent_fixpoint::{
    persistent_fixpoint, persistent_fixpoint_grid, OP_ID, OP_ID_GRID,
    PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};

/// Buffer names every program here is built with.
const CURRENT: &str = "current";
const NEXT: &str = "next";
const CHANGED: &str = "changed";

/// A transfer body that contains no barrier and no `changed` access of its own,
/// so every barrier and every `changed` access in the emitted program is one the
/// builder put there. Counting is exact only with such a body.
fn inert_transfer_body() -> Vec<Node> {
    vec![Node::store(
        NEXT,
        Expr::LogicalIndex { axis: 0 },
        Expr::load(CURRENT, Expr::LogicalIndex { axis: 0 }),
    )]
}

/// The single generator `Region` body `Program::wrapped` puts at the entry.
///
/// `op_id` is asserted rather than ignored, so a test that means to inspect one
/// builder cannot silently be handed the other's program.
fn region_nodes<'a>(program: &'a Program, op_id: &str) -> &'a [Node] {
    match program.entry() {
        [Node::Region {
            generator, body, ..
        }] => {
            assert_eq!(
                generator.as_str(),
                op_id,
                "the builder under test must attribute its region to its own op id"
            );
            body
        }
        other => panic!("expected exactly one generator Region at the entry, got {other:?}"),
    }
}

/// The body of the one in-kernel `Node::Loop` the builder emits.
fn loop_body(program: &Program) -> &[Node] {
    let nodes = region_nodes(program, OP_ID);
    match nodes {
        [Node::Loop { body, .. }] => body,
        other => panic!("expected exactly one in-kernel Loop in the region, got {other:?}"),
    }
}

/// True when `node` is an `If` whose `then` arm is exactly the early exit.
///
/// Matched on the `Return` rather than on the condition expression, because the
/// condition's exact `Expr` tree is an implementation detail while "the node
/// that leaves the kernel" is the thing this file reasons about.
fn is_exit_guard(node: &Node) -> bool {
    matches!(node, Node::If { then, .. } if then.iter().any(|n| matches!(n, Node::Return)))
}

/// True when `node` guards a write to `changed` (the per-iteration clear).
fn is_changed_clear(node: &Node) -> bool {
    fn writes_changed(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| match node {
            Node::Let {
                value: Expr::Atomic { buffer, op, .. },
                ..
            } => buffer.as_str() == CHANGED && *op != AtomicOp::Or,
            Node::Store { buffer, .. } => buffer.as_str() == CHANGED,
            _ => false,
        })
    }
    match node {
        Node::If {
            then, otherwise, ..
        } => writes_changed(then) || writes_changed(otherwise),
        other => writes_changed(std::slice::from_ref(other)),
    }
}

fn is_barrier(node: &Node) -> bool {
    matches!(node, Node::LogicalBarrier { .. })
}

/// Locks out an unguarded loop back edge: the defect that let one warp clear
/// `changed[0]` while a sibling warp had not yet read it, producing a partial
/// exit and a partially-transferred state that reads as converged.
///
/// If this regresses, `persistent_fixpoint` returns wrong answers
/// nondeterministically under warp scheduling, at ANY workgroup count, and
/// nothing hangs or errors to say so.
#[test]
fn a_barrier_guards_the_back_edge_at_any_group_count_including_one() {
    let program = persistent_fixpoint(
        inert_transfer_body(),
        CURRENT,
        NEXT,
        CHANGED,
        PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0],
        8,
    );
    let body = loop_body(&program);

    let clear_at = body
        .iter()
        .position(is_changed_clear)
        .expect("the builder must clear `changed` once per iteration");
    let exit_at = body
        .iter()
        .position(is_exit_guard)
        .expect("the builder must emit the early exit inside the loop body");
    assert!(
        clear_at < exit_at,
        "the clear must open the iteration and the exit must close it, got clear at \
         {clear_at} and exit at {exit_at}"
    );

    // The back edge runs from `exit_at` to `clear_at` of the next iteration, so
    // the guard has to sit AFTER the exit. A barrier before it (there are two,
    // and they are necessary for other reasons) orders nothing on this edge.
    let guard_at = body.iter().skip(exit_at + 1).position(is_barrier);
    assert!(
        guard_at.is_some(),
        "a barrier MUST follow the exit read at index {exit_at} to order it against the \
         next iteration's clear of the same word. Without it the warp that takes the back \
         edge first clears `changed[0]` while a sibling warp has not yet read it; the \
         sibling reads 0, returns, and stops running the transfer body while the rest keep \
         iterating. That is a partial exit: the words the departed invocations own freeze \
         mid-transfer and the dispatch reports convergence on a partially-transferred \
         state. Nothing hangs, because `bar.sync` ignores invocations that already \
         returned. Loop body was: {body:?}"
    );
    assert_eq!(
        exit_at + 1 + guard_at.unwrap_or_default(),
        body.len() - 1,
        "the guarding barrier must be the LAST node of the loop body, so nothing can be \
         inserted between it and the back edge"
    );
}

/// Locks the ordering obligation as a general rule rather than as one node
/// index: between the exit read and every write to `changed`, in back-edge
/// order, there must be a barrier.
///
/// This is the CLASS test. The test above pins today's node layout; this one
/// keeps holding if the builder is restructured, because it states the property
/// the hardware actually requires instead of the shape that currently satisfies
/// it. A reintroduction that moved the clear, split the iteration differently,
/// or added a second `changed` write would slip past an index assertion and is
/// caught here.
#[test]
fn no_changed_write_is_reachable_from_the_exit_read_without_a_barrier() {
    for max_iterations in [1_u32, 2, 8, 64] {
        let program = persistent_fixpoint(
            inert_transfer_body(),
            CURRENT,
            NEXT,
            CHANGED,
            64,
            max_iterations,
        );
        let body = loop_body(&program);
        let exit_at = body
            .iter()
            .position(is_exit_guard)
            .expect("the builder must emit the early exit inside the loop body");

        // Walk the back edge: from just past the exit, around the wrap, up to
        // the exit again. The first thing encountered must be a barrier and not
        // a `changed` write.
        let wrapped = body
            .iter()
            .enumerate()
            .skip(exit_at + 1)
            .chain(body.iter().enumerate().take(exit_at));
        let mut saw_barrier = false;
        for (index, node) in wrapped {
            if is_barrier(node) {
                saw_barrier = true;
                break;
            }
            assert!(
                !is_changed_clear(node),
                "at max_iterations {max_iterations}: the write to `changed` at index {index} \
                 is reachable from the exit read at index {exit_at} along the loop back edge \
                 with no barrier in between, so a warp can clear the flag while a sibling \
                 warp is still reading it and leave the kernel partway through the transfer"
            );
        }
        assert!(
            saw_barrier,
            "at max_iterations {max_iterations}: the back edge from the exit read must cross \
             a barrier"
        );
    }
}

/// Locks out paying for the fix twice.
///
/// The guard was obtained by MOVING a redundant barrier, not by adding one: the
/// builder used to emit two consecutive `SeqCst` barriers before the exit read,
/// the second of which synchronized nothing. Three per iteration is therefore
/// the same cost as before the fix. A future edit that "restores symmetry" by
/// putting a barrier back before the read would make every iteration pay an
/// extra full-workgroup sync for nothing.
#[test]
fn the_iteration_pays_exactly_three_barriers() {
    let program = persistent_fixpoint(inert_transfer_body(), CURRENT, NEXT, CHANGED, 64, 4);
    let body = loop_body(&program);

    assert_eq!(
        body.iter().filter(|node| is_barrier(node)).count(),
        3,
        "one iteration must emit exactly three barriers: after the clear, after the \
         compare, and after the exit read. Loop body was: {body:?}"
    );
    // And none of them may be a grid barrier: this form is workgroup-scope by
    // construction, and a `GridSync` here would silently impose a cooperative
    // launch on every caller that chose this builder to avoid one.
    assert!(
        body.iter().all(|node| !matches!(
            node,
            Node::LogicalBarrier {
                ordering: MemoryOrdering::GridSync
            }
        )),
        "the in-kernel form must use workgroup-scope barriers only"
    );
}

/// Locks the reason the GRID form needs no such guard, so a future edit cannot
/// "unify" the two builders by giving the grid form a clear.
///
/// The grid form is safe for a different reason than the in-kernel form: it has
/// no clear at all and uses one never-cleared word per wave, so there is no
/// write to race the read and no back edge to guard. Adding a clear there would
/// reintroduce this whole class in a place where a partial exit strands other
/// CTAs at a cooperative grid barrier instead of merely corrupting state.
#[test]
fn the_grid_form_never_writes_changed_except_to_set_it() {
    let program = persistent_fixpoint_grid(inert_transfer_body(), CURRENT, NEXT, CHANGED, 64, 4);

    for node in region_nodes(&program, OP_ID_GRID) {
        assert!(
            !is_changed_clear(node),
            "the grid form must never clear or plain-store `changed`; its exit is collective \
             only because each wave owns a never-cleared word, got {node:?}"
        );
    }
}
