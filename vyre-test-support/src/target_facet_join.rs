//! The second opinion the target facet contracts judge the facet registry with.
//!
//! A target facet states that a backend lowers an operation. Two suites check
//! that claim: one against fixture registrations, one against the linked
//! concrete drivers. Both need the same three things, and both had written
//! them out, so the two suites could disagree on what a node walk is while each
//! stayed green and neither said which walk the registry owes.
//!
//! The walk is deliberately not the driver's own join. It answers the same
//! question by a second route, which is what makes the comparison cost
//! something. One owner keeps it one second route rather than two.

use std::collections::HashSet;

use vyre_foundation::ir::{node_op_id, Node, Program};
use vyre_foundation::transform::schedule_lowering::lower_logical_schedule_borrowed;
use vyre_foundation::visit::child_bodies;

/// Every language-level operation the program's nodes name, at every nesting
/// depth, after schedule lowering resolves the logical execution markers.
///
/// Markers are resolved first because `validate_program_contract` resolves them
/// before admission: no backend lowers `vyre.node.logical_barrier` and none
/// needs to.
#[must_use]
pub fn lowered_node_ops(program: &Program) -> HashSet<&'static str> {
    let lowered = lower_logical_schedule_borrowed(program);
    let physical = lowered.as_ref().unwrap_or(program);
    let mut ops = HashSet::new();
    let mut stack: Vec<&Node> = physical.entry().iter().collect();
    while let Some(node) = stack.pop() {
        ops.insert(node_op_id(node));
        for body in child_bodies(node) {
            stack.extend(body.iter());
        }
    }
    ops
}

/// The failure both facet contracts report for a row where the published facet
/// and the backend's arms disagree.
#[must_use]
pub fn facet_disagreement(target: &str, declared: bool, operation: &str) -> String {
    let claim = if declared {
        "declares a facet for"
    } else {
        "declares no facet for"
    };
    format!(
        "Fix: target `{target}` {claim} operation `{operation}`, and its registered \
         language-level operation set says otherwise. A target facet is the \
         intersection of the declared catalog with the arms the backend lowers, never \
         the declared catalog alone."
    )
}

/// Every `(operation, target)` pair the facet registry published.
///
/// Read once here so both contracts compare against the same set rather than
/// against two spellings of the same query.
#[cfg(feature = "driver-contracts")]
#[must_use]
pub fn published_facet_pairs() -> HashSet<(&'static str, String)> {
    vyre_driver::registered_target_operation_facets()
        .expect("Fix: the target facet registry must start")
        .iter()
        .map(|facet| (facet.operation_id, facet.target_id.as_str().to_string()))
        .collect()
}
