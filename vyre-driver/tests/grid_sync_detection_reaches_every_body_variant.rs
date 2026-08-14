//! `contains_grid_sync` must find a fence wherever the IR can nest one.
//!
//! # Why a miss here is a wrong answer, not a slow one
//!
//! `contains_grid_sync` is the routing gate. Every dispatch path in every driver
//! asks it first, and the answer decides between a host split, a cooperative
//! launch, and an outright refusal. Answer "no fence" for a program that has one
//! and the program takes the ORDINARY path: the fence lowers to a workgroup
//! barrier, blocks never synchronize with each other, and the kernel returns
//! plausible wrong numbers with no error anywhere.
//!
//! The gate used to run its own `match node` over `If`, `Loop`, `Block`, and
//! `Region` with `_ => false`. Those are every nesting variant that exists
//! today, so nothing escaped, and that is exactly the problem: the next variant
//! escapes, and it escapes into the failure mode above rather than into a
//! compile error. `Node` is `#[non_exhaustive]`, so this crate cannot be made to
//! fail at compile time; the detection now delegates to
//! `vyre_foundation::transform::visit::any_descendant`, whose child enumeration
//! IS exhaustive in the crate that owns the enum.
//!
//! # Which variants previously escaped
//!
//! None, at the time this was written. `If`, `Loop`, `Block`, and `Region` were
//! all listed. The mutation below is what makes the claim checkable rather than
//! a reading of the source: with a scratch `Node::Speculate { body: Vec<Node> }`
//! in the registry and the old private match restored,
//! `detects_a_fence_in_every_body_slot` reports
//! `Node::Speculate.body: contains_grid_sync must report a nested fence`.
//!
//! # What this test does NOT claim
//!
//! It does not claim every nested fence is hoisted to a launch boundary.
//! Hoisting is legal only out of unconditionally executed bodies, so a fence
//! under `If` or `Loop` stays where it is and the emitter refuses it. That
//! division is pinned by `grid_sync_nested_fence_survives_split.rs`. Here the
//! claim is narrower and load-bearing: the fence is never reported as absent.

use vyre_driver::grid_sync::{contains_grid_sync, split_on_grid_sync};
use vyre_foundation::ir::{BufferDecl, DataType, Node, Program};
use vyre_foundation::transform::visit::child_bodies;
use vyre_foundation::MemoryOrdering;
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, node_body_slot_samples, node_variant_samples,
};

fn program_with(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(256)],
        [256, 1, 1],
        entry,
    )
}

fn grid_fence() -> Node {
    Node::barrier_with_ordering(MemoryOrdering::GridSync)
}

fn grid_fence_count(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| {
            usize::from(matches!(
                node,
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                    ..
                }
            )) + child_bodies(node)
                .into_iter()
                .map(grid_fence_count)
                .sum::<usize>()
        })
        .sum()
}

/// A new `Node` variant fails here before it can reach the routing gate untested.
#[test]
fn every_declared_node_variant_has_a_fixture() {
    assert_covers_every_node_variant(&node_variant_samples());
}

/// The routing gate reports a fence nested in every body slot of every
/// body-carrying variant.
#[test]
fn detects_a_fence_in_every_body_slot() {
    let samples = node_body_slot_samples(&grid_fence());
    assert!(
        !samples.is_empty(),
        "the body-slot fixture set must not be empty, or this test asserts nothing"
    );

    for sample in &samples {
        let program = program_with(vec![sample.node.clone()]);
        assert_eq!(
            grid_fence_count(program.entry()),
            1,
            "{}: the fixture must contain exactly one fence",
            sample.label()
        );
        assert!(
            contains_grid_sync(&program),
            "{}: contains_grid_sync must report a nested fence. Reading it as absent routes \
             the program down the ordinary dispatch path, where the fence lowers to a \
             workgroup barrier and the kernel runs with no cross-block synchronization.",
            sample.label()
        );
    }
}

/// The same holds two levels down, so the gate cannot pass by peeking at
/// immediate children.
#[test]
fn detects_a_fence_nested_two_levels_deep() {
    for outer in node_body_slot_samples(&grid_fence()) {
        for inner in node_body_slot_samples(&outer.node) {
            let program = program_with(vec![inner.node.clone()]);
            assert!(
                contains_grid_sync(&program),
                "{} inside {}: the gate must recurse",
                outer.label(),
                inner.label()
            );
        }
    }
}

/// A detected fence is never silently dropped by the split.
///
/// Detection and hoisting have different depths on purpose. The split may leave a
/// fence nested where hoisting it would change which invocations reach it, but it
/// may never lose one: a program that emits clean and runs unsynchronized is the
/// outcome this pairing exists to prevent.
#[test]
fn a_fence_in_any_body_slot_survives_the_split() {
    for sample in node_body_slot_samples(&grid_fence()) {
        let program = program_with(vec![sample.node.clone()]);
        let segments = split_on_grid_sync(&program);
        assert!(
            !segments.is_empty(),
            "{}: the split must always produce at least one segment",
            sample.label()
        );
        let hoisted_to_boundary = segments.len() > 1;
        let surviving: usize = segments
            .iter()
            .map(|segment| grid_fence_count(segment.entry()))
            .sum();
        assert_eq!(
            surviving,
            usize::from(!hoisted_to_boundary),
            "{}: a fence must either become a launch boundary or survive in a segment, never \
             disappear",
            sample.label()
        );
    }
}

/// A program with no fence is not routed to the split, so the negative answer is
/// still a real measurement rather than a constant `true`.
#[test]
fn reports_no_fence_when_the_body_slots_hold_an_ordinary_barrier() {
    let workgroup_barrier = Node::barrier_with_ordering(MemoryOrdering::SeqCst);
    for sample in node_body_slot_samples(&workgroup_barrier) {
        let program = program_with(vec![sample.node.clone()]);
        assert!(
            !contains_grid_sync(&program),
            "{}: a workgroup barrier is not a grid fence, and routing it to the host split \
             would break every ordinary program that synchronizes",
            sample.label()
        );
    }
}

/// A variant that nests nothing cannot hide a fence.
#[test]
fn variants_without_bodies_report_no_fence() {
    let nesting: Vec<&'static str> = node_body_slot_samples(&grid_fence())
        .iter()
        .map(|sample| sample.variant)
        .collect();
    for sample in node_variant_samples() {
        if nesting.contains(&sample.variant) || sample.variant == "Barrier" {
            continue;
        }
        assert!(
            !contains_grid_sync(&program_with(vec![sample.node.clone()])),
            "{}: nothing is nested here, so no fence can be present",
            sample.label()
        );
    }
}
