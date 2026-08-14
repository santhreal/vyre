//! The class closed here: a new `Node` variant reaching a traversal that nobody
//! taught it to.
//!
//! # What used to stand here
//!
//! `scripts/check_unification_baselines.sh` carried a row named `P-DELETE-1`
//! that counted `match node {` occurrences under `validate/` and `transform/`
//! and required the count to stay at or below 18. That number is a proxy and it
//! is wrong in both directions. A count that falls proves nothing: 4 of the 22
//! blocks it counted ended in a catch-all arm, and deleting an exhaustive one
//! lowers the count while making the workspace less safe. A count that rises
//! blocks a legitimate new traversal that handles every variant. The row never
//! looked at whether any traversal descends into anything.
//!
//! # The property that replaced it
//!
//! A traversal that must reach every body-carrying or operand-carrying variant
//! cannot pass once a variant is added without a decision recorded for it. Two
//! halves, because one alone is not enough:
//!
//! - COMPILE TIME. `vyre_foundation::transform::visit::node_shape` matches every
//!   variant with no catch-all arm. Adding a variant fails to compile there,
//!   which forces the author to say whether it nests bodies, owns operands, or
//!   is opaque. `child_bodies` is exhaustive for the same reason.
//! - RUN TIME. `Node` is `#[non_exhaustive]`, so no crate other than this one
//!   can be made to fail at compile time. Every downstream traversal ends in a
//!   catch-all arm and always will. So the fixtures in
//!   `vyre_test_support::ir_variants` are checked against
//!   `NODE_VARIANT_NAMES`, which the registry macro emits from the enum body,
//!   and each traversal suite drives those fixtures. A new variant has no
//!   fixture, the coverage assertion fails, and every suite built on it fails
//!   with it.
//!
//! The mutation that proves it: adding `Node::Speculate { body: Vec<Node> }` to
//! the registry makes `node_shape` and `child_bodies` fail to compile, and once
//! those are satisfied `node_variant_samples_cover_every_declared_variant`
//! fails naming `["Speculate"]`.

use vyre_foundation::ir::{Expr, Node, NODE_VARIANT_NAMES};
use vyre_foundation::transform::visit::{
    any_descendant, child_bodies, node_shape, walk_exprs, NodeShape,
};
use vyre_foundation::MemoryOrdering;
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, assert_samples_match_declared_shape, node_body_slot_samples,
    node_operand_samples, node_variant_samples,
};

fn program_of(nodes: Vec<Node>) -> vyre_foundation::ir::Program {
    vyre_foundation::ir::Program::wrapped(Vec::new(), [1, 1, 1], nodes)
}

/// Every declared variant has a fixture.
///
/// This is the assertion that goes red when somebody adds a `Node` variant. It
/// does NOT catch a variant handled wrongly; it catches a variant nobody
/// considered, which is the failure that used to be silent.
#[test]
fn node_variant_samples_cover_every_declared_variant() {
    let samples = node_variant_samples();
    assert_covers_every_node_variant(&samples);
    assert_eq!(
        samples.len(),
        NODE_VARIANT_NAMES.len(),
        "one fixture per declared variant, no more: {:?} against {NODE_VARIANT_NAMES:?}",
        samples.iter().map(|s| s.variant).collect::<Vec<_>>()
    );
}

/// The declared shape of a variant matches what the IR actually holds.
///
/// `node_shape` is a hand-recorded decision, so it can disagree with the enum.
/// A variant declared to nest nodes whose body a traversal cannot reach, or one
/// declared inert that in fact holds a body, would make every test below pass
/// for the wrong reason.
#[test]
fn declared_shape_agrees_with_reachable_contents() {
    let marker = Node::barrier_with_ordering(MemoryOrdering::SeqCst);
    let body_samples = node_body_slot_samples(&marker);
    assert_samples_match_declared_shape(&body_samples, true);
    for sample in &body_samples {
        let reachable = child_bodies(&sample.node)
            .into_iter()
            .flatten()
            .any(|child| *child == marker);
        assert!(
            reachable,
            "{}: child_bodies must expose the planted marker; a slot it does not \
             return is a subtree no traversal built on it can reach",
            sample.label()
        );
    }

    let operand = Expr::var("vyre_fixture_marker_operand");
    let operand_samples = node_operand_samples(&operand);
    assert_samples_match_declared_shape(&operand_samples, false);
    for sample in &operand_samples {
        let mut seen = false;
        walk_exprs(&program_of(vec![sample.node.clone()]), |expr| {
            if *expr == operand {
                seen = true;
            }
        });
        assert!(
            seen,
            "{}: walk_exprs must reach the planted operand",
            sample.label()
        );
    }

    for sample in node_variant_samples() {
        let shape = node_shape(&sample.node);
        let bodies_declared = shape.nests_nodes;
        let has_slot = node_body_slot_samples(&marker)
            .iter()
            .any(|body| body.variant == sample.variant);
        assert_eq!(
            bodies_declared,
            has_slot,
            "{}: node_shape says nests_nodes={bodies_declared} but the fixture set \
             {} a body slot for it",
            sample.label(),
            if has_slot { "has" } else { "has no" }
        );
    }
}

/// A shape is never both inert and something else.
#[test]
fn opaque_variants_are_not_reported_as_leaves() {
    let opaque = node_variant_samples()
        .into_iter()
        .find(|sample| sample.variant == "Opaque")
        .expect("Fix: the Opaque fixture is required by assert_covers_every_node_variant");
    assert_eq!(
        node_shape(&opaque.node),
        NodeShape {
            nests_nodes: false,
            carries_operands: false,
            opaque_payload: true,
        },
        "an extension payload must read as opaque, so an analysis answers unknown \
         rather than treating it as an empty leaf"
    );
}

/// `any_descendant` reaches a marker planted in every body slot of every
/// body-carrying variant.
///
/// This is the contract every traversal in the workspace inherits by delegating
/// to it instead of restating the nesting set. A traversal that keeps its own
/// `match node` with `_ => false` gets no such guarantee, which is why
/// `validate::barrier`, the megakernel barrier planner, and the grid-sync
/// splitter all route through here.
#[test]
fn any_descendant_reaches_every_body_slot() {
    let marker = Node::barrier_with_ordering(MemoryOrdering::SeqCst);
    let samples = node_body_slot_samples(&marker);
    assert!(!samples.is_empty(), "the fixture set must not be empty");

    for sample in &samples {
        assert!(
            any_descendant(&sample.node, &mut |node| *node == marker),
            "{}: any_descendant missed a marker planted directly in this slot",
            sample.label()
        );
    }
}

/// The same guarantee holds one level deeper, so a traversal cannot pass by
/// checking only immediate children.
#[test]
fn any_descendant_reaches_a_marker_nested_two_levels_deep() {
    let marker = Node::barrier_with_ordering(MemoryOrdering::SeqCst);
    for outer in node_body_slot_samples(&marker) {
        for inner in node_body_slot_samples(&outer.node) {
            assert!(
                any_descendant(&inner.node, &mut |node| *node == marker),
                "{} inside {}: any_descendant must recurse, not peek",
                outer.label(),
                inner.label()
            );
        }
    }
}

/// A variant with no body slot really has none, so a traversal that stops at it
/// loses nothing.
#[test]
fn variants_without_bodies_expose_no_children() {
    for sample in node_variant_samples() {
        if node_shape(&sample.node).nests_nodes {
            continue;
        }
        let children: usize = child_bodies(&sample.node)
            .into_iter()
            .map(<[Node]>::len)
            .sum();
        assert_eq!(
            children,
            0,
            "{}: declared as carrying no bodies, so child_bodies must return none",
            sample.label()
        );
    }
}
