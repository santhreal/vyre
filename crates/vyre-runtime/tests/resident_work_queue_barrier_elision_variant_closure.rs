//! The megakernel barrier planner must reach a barrier wherever the IR can hide
//! one.
//!
//! # The failure this locks out
//!
//! `elide_value_flow_barriers` is gated twice on "does this subtree contain a
//! barrier": once at the entry, once per `Region`. Both gates used to run a
//! private `match node` over `If`, `Loop`, `Block`, and `Region` ending in
//! `_ => false`, and `rewrite_node` still ends in a catch-all because `Node` is
//! `#[non_exhaustive]` and no match outside `vyre-foundation` can be exhaustive.
//!
//! A body-carrying variant that neither gate knows reads as a LEAF. The gate
//! answers "no barrier here", the pass returns the program untouched, and the
//! megakernel keeps every barrier it could have elided. Nothing is wrong with
//! the output, which is why this never surfaces as a bug report: it surfaces as
//! a megakernel that is slower than it should be, on a shape nobody profiles.
//!
//! # How this test cannot rot
//!
//! The fixtures are `vyre_test_support::ir_variants`, which is checked against
//! the variant list the registry macro emits from the `Node` declaration. A new
//! variant has no fixture, so `every_declared_node_variant_has_a_fixture` fails;
//! once a fixture exists, `elides_a_barrier_inside_every_body_slot` drives it and
//! fails until the planner descends into it.
//!
//! Verified to go red against the pre-fix planner: with `node_has_barrier`
//! restored to its own four-variant match and a scratch `Node::Speculate { body }`
//! added to the registry, `elides_a_barrier_inside_every_body_slot` reports
//! `Node::Speculate.body: expected 1 elided barrier, got 0`.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_runtime::resident_work_queue::planner::elide_value_flow_barriers;
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, node_body_slot_samples, node_variant_samples,
};

fn buffer(name: &str, binding: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32)
}

/// Two disjoint arms with one barrier between them: exactly one elidable barrier.
fn elidable_trio() -> Node {
    Node::Block(vec![
        Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]),
        Node::barrier(),
        Node::Block(vec![Node::store("b", Expr::u32(0), Expr::u32(2))]),
    ])
}

/// Two arms that both write `a`: a write/write dependency crosses the barrier,
/// so nothing may be elided.
fn conflicting_trio() -> Node {
    Node::Block(vec![
        Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]),
        Node::barrier(),
        Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(2))]),
    ])
}

fn program_with(entry: Vec<Node>) -> Program {
    Program::wrapped(vec![buffer("a", 0), buffer("b", 1)], [64, 1, 1], entry)
}

/// A new `Node` variant fails here before it can reach the planner untested.
#[test]
fn every_declared_node_variant_has_a_fixture() {
    assert_covers_every_node_variant(&node_variant_samples());
}

/// The planner elides a barrier nested inside every body slot of every
/// body-carrying variant.
///
/// One assertion per slot, not per variant: `Node::If` owns `then` and
/// `otherwise` separately, and a rewrite that descends into one and not the
/// other passes a per-variant test while losing half the tree.
#[test]
fn elides_a_barrier_inside_every_body_slot() {
    let samples = node_body_slot_samples(&elidable_trio());
    assert!(
        !samples.is_empty(),
        "the body-slot fixture set must not be empty, or this test asserts nothing"
    );

    for sample in &samples {
        let (rewritten, report) =
            elide_value_flow_barriers(program_with(vec![sample.node.clone()]));
        assert_eq!(
            report.removed,
            1,
            "{}: expected 1 elided barrier, got {}. The planner did not descend into this \
             slot, so every barrier under it is invisible to the pass.",
            sample.label(),
            report.removed
        );
        assert_eq!(
            barrier_count(rewritten.entry()),
            0,
            "{}: the elided barrier must be gone from the rewritten program, not merely \
             counted",
            sample.label()
        );
    }
}

/// The same walk refuses to elide when the arms conflict.
///
/// Without this, `elides_a_barrier_inside_every_body_slot` would still pass
/// against a planner that removed every barrier it found, which is precisely the
/// bug the dependency analysis exists to prevent.
#[test]
fn keeps_a_barrier_inside_every_body_slot_when_the_arms_conflict() {
    for sample in &node_body_slot_samples(&conflicting_trio()) {
        let (rewritten, report) =
            elide_value_flow_barriers(program_with(vec![sample.node.clone()]));
        assert_eq!(
            report.removed,
            0,
            "{}: a write/write dependency crosses this barrier, so it must survive",
            sample.label()
        );
        assert_eq!(
            barrier_count(rewritten.entry()),
            1,
            "{}: the surviving barrier must still be in the program",
            sample.label()
        );
    }
}

/// A variant that carries no body cannot hide a barrier, so the planner leaves
/// the program alone rather than reporting a phantom elision.
#[test]
fn variants_without_bodies_produce_no_elision() {
    for sample in node_variant_samples() {
        if node_body_slot_samples(&elidable_trio())
            .iter()
            .any(|body| body.variant == sample.variant)
        {
            continue;
        }
        let (_, report) = elide_value_flow_barriers(program_with(vec![sample.node.clone()]));
        assert_eq!(
            report.removed,
            0,
            "{}: nothing is nested here, so there is nothing to elide",
            sample.label()
        );
    }
}

/// Count barriers through the exhaustive child enumeration rather than a fourth
/// private copy of the nesting set.
fn barrier_count(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| {
            usize::from(matches!(node, Node::Barrier { .. }))
                + vyre_foundation::visit::child_bodies(node)
                    .into_iter()
                    .map(barrier_count)
                    .sum::<usize>()
        })
        .sum()
}
