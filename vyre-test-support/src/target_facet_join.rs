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

#[cfg(feature = "driver-contracts")]
use vyre_driver::BackendRegistration;
use vyre_foundation::ir::{node_op_id, Node, Program};
#[cfg(feature = "driver-contracts")]
use vyre_foundation::operation::OperationRegistry;
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

/// One `(operation, target)` row: what the facet registry published, and what
/// the target's registered arms say.
#[cfg(feature = "driver-contracts")]
pub struct FacetPair {
    /// Registered operation the row is about.
    pub operation: &'static str,
    /// Target the row is about.
    pub target: String,
    /// The target's registration lists the operation in its semantic set.
    pub claimed: bool,
    /// The facet registry published this pair.
    pub declared: bool,
    /// The target compiles natively, claims the operation, and registers an
    /// arm for every node of the operation's canonical program.
    pub expected: bool,
}

/// Every `(operation, target)` row over `backends`, against the whole
/// operation registry.
///
/// Both facet contracts ask one question of a different backend set: the
/// fixture registrations in the driver suite, the linked concrete drivers in
/// the conformance suite. The row and the join are that question, so the
/// caller supplies only the backends and the operation axis is read here.
#[cfg(feature = "driver-contracts")]
#[must_use]
pub fn facet_pairs<'a>(
    backends: impl IntoIterator<Item = &'a BackendRegistration>,
) -> Vec<FacetPair> {
    let backends: Vec<&BackendRegistration> = backends.into_iter().collect();
    let published = published_facet_pairs();

    let mut rows = Vec::new();
    for operation in OperationRegistry::global().iter() {
        let node_ops = operation.program().as_ref().map(lowered_node_ops);
        for backend in &backends {
            let claimed = (backend.semantic_operations)().contains(operation.id);
            let supported = (backend.supported_ops)();
            let expected = backend.target_compiler.is_some()
                && claimed
                && node_ops
                    .as_ref()
                    .is_some_and(|node_ops| node_ops.iter().all(|op| supported.contains(*op)));
            let target = backend.target_id.as_str().to_string();
            rows.push(FacetPair {
                operation: operation.id,
                claimed,
                declared: published.contains(&(operation.id, target.clone())),
                target,
                expected,
            });
        }
    }
    rows
}

/// Assert every row's published facet equals what the target's arms say.
///
/// The comparison is the contract both facet suites exist to state, so it is
/// one loop here rather than one loop each. A suite that wrote its own could
/// compare a different field pair or report a different failure while still
/// claiming to check the same registry.
#[cfg(feature = "driver-contracts")]
pub fn assert_facets_agree(rows: &[FacetPair]) {
    for pair in rows {
        assert_eq!(
            pair.declared,
            pair.expected,
            "{}",
            facet_disagreement(&pair.target, pair.declared, pair.operation)
        );
    }
}
