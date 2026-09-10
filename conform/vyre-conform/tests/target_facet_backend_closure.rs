//! Every target facet the shipped registry publishes is checked against the
//! lowering arms of the concrete driver that declared it.
//!
//! # Why this suite exists
//!
//! The generated operation schema recorded the same target facet list for every
//! operation in the catalog, because `registered_target_operation_facets`
//! emitted one facet per declared semantic operation and every concrete driver
//! declares the whole catalog through `dialect_only_supported_ops`. Uniform
//! evidence is either trivially true or false, and it was false: a driver
//! registers a language-level operation set for dispatch that is smaller than
//! the catalog, and admission refuses a program naming an operation outside it.
//!
//! `vyre-driver`'s own suite proves the join against fixture registrations.
//! This one runs the same question against the linked concrete drivers and the
//! whole shipped catalog, which is where the uniform claim was produced.
//!
//! # Why the pairs are enumerated and not listed
//!
//! Both axes are read from source at run time: `OperationRegistry::global()`
//! for the operations and `registered_backends()` for the targets. Registering
//! an operation whose canonical program names a node a linked target does not
//! lower turns this suite red until the facet join accounts for it, and linking
//! a new concrete driver adds its whole column without an edit here.

use vyre_driver::BackendRegistration;
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::transform::schedule_lowering::lower_logical_schedule_borrowed;
// The node walk, the published pair set and the disagreement wording are the
// second opinion both facet contracts judge the registry with, so they have one
// owner rather than one copy per suite.
use vyre_test_support::target_facet_join::{
    facet_disagreement, lowered_node_ops, published_facet_pairs,
};

/// One `(operation, target)` row: what the facet registry published, and what
/// the target's registered arms say.
struct Pair {
    operation: &'static str,
    target: String,
    claimed: bool,
    declared: bool,
    expected: bool,
}

fn pairs() -> Vec<Pair> {
    let backends = vyre_driver::registered_backends()
        .expect("Fix: the linked backend registry must start before target facets are read");
    let published = published_facet_pairs();

    let compiling: Vec<&BackendRegistration> = backends
        .iter()
        .filter(|backend| backend.target_compiler.is_some())
        .collect();

    let mut rows = Vec::new();
    for operation in OperationRegistry::global().iter() {
        let program = operation.program();
        let node_ops = program.as_ref().map(lowered_node_ops);
        for backend in &compiling {
            let claimed = (backend.semantic_operations)().contains(operation.id);
            let supported = (backend.supported_ops)();
            let expected = claimed
                && node_ops
                    .as_ref()
                    .is_some_and(|node_ops| node_ops.iter().all(|op| supported.contains(*op)));
            let target = backend.target_id.as_str().to_string();
            rows.push(Pair {
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

/// The closing contract, over the shipped catalog and the linked drivers.
#[test]
fn every_shipped_operation_target_pair_agrees_with_the_backend_lowering_arms() {
    let rows = pairs();
    assert!(
        !rows.is_empty(),
        "Fix: no compiler-capable target is linked into this binary, so the pair contract asserts nothing. The `gpu` feature links the concrete drivers this suite judges."
    );
    for pair in &rows {
        assert_eq!(
            pair.declared,
            pair.expected,
            "{}",
            facet_disagreement(&pair.target, pair.declared, pair.operation)
        );
    }
}

/// Backend maturity is asymmetric, and the facet registry has to say so.
///
/// Every linked driver declares the whole catalog, so a facet set equal to the
/// declared catalog is the blanket answer this row removed. At least one pair
/// must be claimed and unsupported, or the join is back to reporting closure it
/// never checked.
#[test]
fn the_published_facets_are_narrower_than_the_declared_catalog() {
    let rows = pairs();
    let claimed = rows.iter().filter(|pair| pair.claimed).count();
    let published = rows.iter().filter(|pair| pair.declared).count();

    assert!(
        published > 0,
        "Fix: no target publishes a facet for any shipped operation, so the catalog reports no backend at all."
    );
    assert!(
        published < claimed,
        "Fix: every claimed pair is published, so the facet registry is reporting the declared catalog rather than the lowering arms. {published} published against {claimed} claimed."
    );
}

/// Non-vacuity for the comparison above: the narrowing comes from the arms, not
/// from a driver that quietly claims less than the catalog.
#[test]
fn every_linked_target_declares_the_whole_catalog() {
    let backends =
        vyre_driver::registered_backends().expect("Fix: the backend registry must start");
    let registry_size = OperationRegistry::global().iter().len();

    let mut compiling = 0usize;
    for backend in backends
        .iter()
        .filter(|backend| backend.target_compiler.is_some())
    {
        compiling += 1;
        assert_eq!(
            (backend.semantic_operations)().len(),
            registry_size,
            "Fix: target `{}` declares {} semantic operations against a catalog of {registry_size}. This suite compares published facets against a whole-catalog claim; a target that claims less needs its own comparison.",
            backend.target_id,
            (backend.semantic_operations)().len(),
        );
    }
    assert!(
        compiling > 0,
        "Fix: the `gpu` feature must link the concrete drivers this suite judges."
    );
}

/// A published facet is checked against the one Program-to-descriptor
/// boundary, not only against the operation set the target registered.
///
/// The contract above compares the facet registry with `supported_ops`, and
/// both read the same registration, so it catches a join that ignores the arms
/// and not an arm that stopped existing. `vyre_lower::lower` is the single
/// place a `Program` becomes something an emitter reads, so a node kind whose
/// lowering arm is removed fails here while the facet still claims the
/// operation. That is the second half of the claim: a facet asserts a real
/// lowering, and this is what makes the assertion cost something.
#[test]
fn every_published_facet_lowers_through_the_program_to_descriptor_boundary() {
    let facets = vyre_driver::registered_target_operation_facets()
        .expect("Fix: the target facet registry must start");
    assert!(
        !facets.is_empty(),
        "Fix: the facet registry published nothing, so this contract lowers nothing."
    );

    for facet in facets {
        let operation = OperationRegistry::global()
            .get(facet.operation_id)
            .expect("Fix: a facet must name a registered operation");
        let program = operation.program().unwrap_or_else(|| {
            panic!(
                "Fix: target `{}` publishes a facet for `{}`, which has no canonical program to lower.",
                facet.target_id, facet.operation_id
            )
        });
        let lowered = lower_logical_schedule_borrowed(&program);
        let physical = lowered.as_ref().unwrap_or(&program);
        if let Err(error) = vyre_lower::lower(physical) {
            panic!(
                "Fix: target `{}` publishes a facet for `{}`, and the shared Program-to-descriptor boundary refuses it: {error}. A facet states that a real lowering exists.",
                facet.target_id, facet.operation_id
            );
        }
    }
}
