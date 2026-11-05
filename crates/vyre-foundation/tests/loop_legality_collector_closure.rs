//! Run-time closure gate for the shared loop-legality read sets.
//!
//! [`var_reads`] and [`touched_buffers`] are the two questions every loop
//! restructuring pass asks before it reorders statements: which scalars does
//! this statement list read, and which buffers does it touch. Both were once
//! answered by a hand-written walk over `Node` and `Expr` that ended in
//! `_ => {}`, so a variant the author had not named answered ABSENT rather than
//! unknown. A `Var` read in `Node::Trap.address`, or in an async copy's
//! `offset`, was invisible, and `loop_fusion` fused two loops across a scalar
//! that one of them assigns, which silently changes the values the program
//! computes.
//!
//! Routing both onto the exhaustive owners in `visit` closes that for
//! the variants that exist. It does not close it for the next one, because a
//! fixture list written today would not exercise a variant added tomorrow. So
//! the fixtures here come from `vyre_test_support::ir_variants`, which checks
//! itself against `NODE_VARIANT_NAMES` as emitted by the enum declaration. A new
//! `Node` variant has no fixture, the coverage assertion fails, and every suite
//! below fails with it until somebody records what each collector owes the new
//! variant.
//!
//! The mutation that proves it: restoring a `_ => {}` arm over node operands in
//! `collect_var_reads` makes `every_operand_slot_is_a_read` fail naming
//! `Node::Trap.address`.

use vyre_foundation::ir::{Expr, Ident, Node};
use vyre_foundation::optimizer::passes::loops::{bound_names, touched_buffers, var_reads};
use vyre_foundation::visit::node_bound_name;
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, assert_samples_match_declared_shape, node_body_slot_samples,
    node_operand_samples, node_variant_samples, NodeSample,
};

/// The scalar planted in every operand slot.
const READ_MARKER: &str = "closure_marker_scalar";
/// The buffer planted in every body and operand slot.
const BUFFER_MARKER: &str = "closure_marker_buffer";
/// The name bound in every body slot.
const BIND_MARKER: &str = "closure_marker_binding";

fn operand_samples(marker: &Expr) -> Vec<NodeSample> {
    let samples = node_operand_samples(marker);
    assert_samples_match_declared_shape(&samples, false);
    samples
}

fn body_samples(marker: &Node) -> Vec<NodeSample> {
    let samples = node_body_slot_samples(marker);
    assert_samples_match_declared_shape(&samples, true);
    samples
}

fn holds(names: &[Ident], wanted: &str) -> bool {
    names.iter().any(|name| name.as_str() == wanted)
}

/// Every declared `Node` variant is exercised by at least one fixture below.
///
/// This is the assertion that goes red when somebody adds a variant. It does not
/// catch a variant answered wrongly; it catches a variant nobody considered,
/// which is the failure the hand-written walks used to hide.
#[test]
fn every_declared_node_variant_has_a_collector_fixture() {
    let mut all = node_variant_samples();
    all.extend(operand_samples(&Expr::var(READ_MARKER)));
    all.extend(body_samples(&Node::assign(READ_MARKER, Expr::u32(0))));
    assert_covers_every_node_variant(&all);
}

/// A `Var` in any operand slot of any variant is a read.
#[test]
fn every_operand_slot_is_a_read() {
    for sample in operand_samples(&Expr::var(READ_MARKER)) {
        let reads = var_reads(std::slice::from_ref(&sample.node));
        assert!(
            holds(&reads, READ_MARKER),
            "Fix: var_reads must report the Var in {}; a read it cannot see lets \
             loop_fusion fuse across a real scalar dependency. Saw {reads:?}",
            sample.label()
        );
    }
}

/// A `Var` read nested in any body slot of any variant is a read.
#[test]
fn every_body_slot_is_descended_for_reads() {
    let planted = Node::assign("closure_marker_sink", Expr::var(READ_MARKER));
    for sample in body_samples(&planted) {
        let reads = var_reads(std::slice::from_ref(&sample.node));
        assert!(
            holds(&reads, READ_MARKER),
            "Fix: var_reads must descend into {} and report the Var it holds. Saw {reads:?}",
            sample.label()
        );
    }
}

/// A store nested in any body slot of any variant touches that buffer.
#[test]
fn every_body_slot_is_descended_for_buffers() {
    let planted = Node::store(BUFFER_MARKER, Expr::u32(0), Expr::u32(1));
    for sample in body_samples(&planted) {
        let buffers = touched_buffers(std::slice::from_ref(&sample.node));
        assert!(
            holds(&buffers, BUFFER_MARKER),
            "Fix: touched_buffers must descend into {} and report the buffer the store \
             names; a missed write reads as disjoint memory and lets the loop passes \
             reorder a real memory dependence. Saw {buffers:?}",
            sample.label()
        );
    }
}

/// A buffer load in any operand slot of any variant touches that buffer.
#[test]
fn every_operand_slot_is_searched_for_buffers() {
    let load = Expr::load(BUFFER_MARKER, Expr::u32(0));
    for sample in operand_samples(&load) {
        let buffers = touched_buffers(std::slice::from_ref(&sample.node));
        assert!(
            holds(&buffers, BUFFER_MARKER),
            "Fix: touched_buffers must report the buffer loaded in {}. Saw {buffers:?}",
            sample.label()
        );
    }
}

/// A binding nested in any body slot of any variant is a binding.
#[test]
fn every_body_slot_is_descended_for_bindings() {
    let planted = Node::let_bind(BIND_MARKER, Expr::u32(0));
    for sample in body_samples(&planted) {
        let bound = bound_names(std::slice::from_ref(&sample.node));
        assert!(
            holds(&bound, BIND_MARKER),
            "Fix: bound_names must descend into {} and report the name it binds; a \
             binding it cannot see reads as dead and lets the loop passes reorder \
             statements across a live definition. Saw {bound:?}",
            sample.label()
        );
    }
}

/// `node_bound_name` answers for every declared variant, and the three binding
/// forms are the only ones that answer `Some`.
#[test]
fn only_the_binding_forms_report_a_bound_name() {
    for sample in node_variant_samples() {
        let answered = node_bound_name(&sample.node).is_some();
        let expected = matches!(
            sample.node,
            Node::Let { .. } | Node::Assign { .. } | Node::Loop { .. }
        );
        assert_eq!(
            answered,
            expected,
            "Fix: node_bound_name disagrees with the binding forms for {}",
            sample.label()
        );
    }
}
