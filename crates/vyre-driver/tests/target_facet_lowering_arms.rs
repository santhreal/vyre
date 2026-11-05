//! A target facet states that a backend lowers an operation, and every pair is
//! checked against the backend's own lowering arms.
//!
//! # Why this suite exists
//!
//! `registered_target_operation_facets` emitted one facet per declared semantic
//! operation, and every concrete driver declares the whole catalog through
//! `dialect_only_supported_ops`. Each target therefore reported the same 349
//! operations whatever its emitter set contained, and the generated operation
//! schema recorded that uniform answer as backend closure. A claim that holds
//! for every operation and every target carries no information, and here it was
//! also wrong: the language-level operation set a driver registers for dispatch
//! is smaller than the catalog, and a program the backend cannot lower is
//! refused by name at admission while its facet still claimed support.
//!
//! A facet is now the intersection: the backend declares a catalog, registers
//! the language-level operations it lowers, and contributes a facet only for a
//! canonical program whose every node it lowers.
//!
//! # Why the pairs are enumerated and not listed
//!
//! Both axes come from source at run time: `OperationRegistry::global()` for
//! the operations and `registered_backends()` for the targets. A written-out
//! operation list is the defect this suite closes, one level up: it agrees with
//! the catalog on the day it is written and stops agreeing in silence. An
//! operation registered with a node no linked backend lowers turns this suite
//! red until a facet decision exists for it.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::{BackendError, BackendRegistration, VyreBackend};
use vyre_foundation::ir::OpId;
use vyre_foundation::operation::{OperationRegistry, TargetId};
// The node walk, the published pair set, the pair row and the disagreement
// wording are the second opinion both facet contracts judge the registry with,
// so they have one owner rather than one copy per suite.
use vyre_test_support::target_facet_join::{
    assert_facets_agree, facet_pairs, lowered_node_ops, FacetPair,
};

/// Registers the same language-level set the two emitting production drivers
/// register, so the pairs this fixture produces are the pairs a shipped target
/// produces.
const EVERY_ARM: &str = "facet-fixture-every-arm";
/// The same registration one arm short. A target whose emitter is missing one
/// physical operation is the asymmetry the row is about, and without it every
/// pair in the binary would agree for the same reason and prove nothing.
const NO_STORE_ARM: &str = "facet-fixture-no-store-arm";

/// The node operation both fixtures differ on.
const WITHHELD_ARM: &str = "vyre.node.store";

/// Neither fixture reaches a device or a native toolchain on any host. Dispatch
/// and compilation both refuse by name, exactly as a shipped driver refuses on
/// a host without its runtime, and the facet join reads neither: it asks
/// whether the registration declares native target compilation at all.
fn unavailable_backend() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::new(
        "target facet fixture backends do not dispatch. Fix: use them only for target facet registry validation.",
    ))
}

/// Every language-level operation the emitting production drivers register.
fn every_arm_ops() -> &'static HashSet<OpId> {
    vyre_driver::default_supported_ops_with_trap()
}

/// The same set without one arm.
fn no_store_arm_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(|| {
        let mut ops = vyre_driver::default_supported_ops_with_trap().clone();
        ops.remove(WITHHELD_ARM);
        ops
    });
    &OPS
}

inventory::submit! {
    BackendRegistration {
        id: EVERY_ARM,
        target_id: TargetId::expect_valid(EVERY_ARM),
        payload_format: Some("facet-fixture-every-arm-payload"),
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: every_arm_ops,
        semantic_operations: vyre_driver::dialect_only_supported_ops,
        target_compiler: Some(|| {
            Err(BackendError::new(
                "target facet fixture backends do not compile. Fix: use them only for target facet registry validation.",
            ))
        }),
        materializer: None,
    }
}

inventory::submit! {
    BackendRegistration {
        id: NO_STORE_ARM,
        target_id: TargetId::expect_valid(NO_STORE_ARM),
        payload_format: Some("facet-fixture-no-store-arm-payload"),
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_store_arm_ops,
        semantic_operations: vyre_driver::dialect_only_supported_ops,
        target_compiler: Some(|| {
            Err(BackendError::new(
                "target facet fixture backends do not compile. Fix: use them only for target facet registry validation.",
            ))
        }),
        materializer: None,
    }
}

/// Every `(operation, target)` row over every registered backend.
fn pairs() -> Vec<FacetPair> {
    let backends = vyre_driver::registered_backends()
        .expect("Fix: the linked backend registry must start before target facets are read");
    facet_pairs(backends)
}

/// The closing contract. Every registered operation against every linked
/// target, with no member of either axis written down here.
#[test]
fn every_operation_target_pair_agrees_with_the_backend_lowering_arms() {
    assert_facets_agree(&pairs());
}

/// Without this the contract above passes against an empty pair space, an empty
/// facet registry, or a registry where every backend answers the same way.
#[test]
fn the_pair_space_carries_both_answers() {
    let rows = pairs();
    assert!(
        rows.iter().any(|pair| pair.expected),
        "Fix: no linked target lowers any registered operation, so the pair contract asserts nothing. Register a compiler-capable backend whose arms cover a canonical program."
    );
    assert!(
        rows.iter().any(|pair| !pair.expected),
        "Fix: every pair is supported, so the pair contract cannot distinguish an intersection from a blanket default."
    );
}

/// The asymmetry itself: two targets that declare the identical catalog and
/// differ only in one arm must not report the identical facet set.
///
/// This is the shape the row names. Both production emitters declare the whole
/// catalog through `dialect_only_supported_ops`, so a facet set that follows
/// the declaration cannot tell a complete emitter from an incomplete one.
#[test]
fn two_targets_with_the_same_catalog_and_different_arms_report_different_facets() {
    let facets = vyre_driver::registered_target_operation_facets()
        .expect("Fix: the target facet registry must start");
    let count = |target: &str| {
        facets
            .iter()
            .filter(|facet| facet.target_id.as_str() == target)
            .count()
    };

    let complete = count(EVERY_ARM);
    let partial = count(NO_STORE_ARM);

    assert!(
        complete > 0,
        "Fix: the complete-arm fixture must contribute facets, otherwise the comparison below is between two empty sets."
    );
    assert!(
        partial < complete,
        "Fix: `{NO_STORE_ARM}` registers no `{WITHHELD_ARM}` arm and declares the same catalog as `{EVERY_ARM}`, so it must report fewer facets. It reported {partial} against {complete}."
    );

    for facet in facets
        .iter()
        .filter(|facet| facet.target_id.as_str() == NO_STORE_ARM)
    {
        let operation = OperationRegistry::global()
            .get(facet.operation_id)
            .expect("Fix: a facet must name a registered operation");
        let program = operation
            .program()
            .expect("Fix: a facet must name an operation with a canonical program");
        assert!(
            !lowered_node_ops(&program).contains(WITHHELD_ARM),
            "Fix: `{NO_STORE_ARM}` declares a facet for `{}`, whose canonical program names `{WITHHELD_ARM}`, an arm it does not register.",
            facet.operation_id
        );
    }
}

/// Non-vacuity for the fixture pair: the two backends differ in arms alone, not
/// in what they claim. A fixture that narrowed its declared catalog instead
/// would pass the contract above while proving nothing about the join.
#[test]
fn both_fixture_targets_declare_the_whole_catalog() {
    let backends =
        vyre_driver::registered_backends().expect("Fix: the backend registry must start");
    let catalog = vyre_driver::dialect_only_supported_ops();
    let registry_size = OperationRegistry::global().iter().len();

    assert_eq!(
        catalog.len(),
        registry_size,
        "Fix: the declared catalog must be the operation registry, otherwise the fixtures under-claim and the join is never asked a hard question."
    );

    for id in [EVERY_ARM, NO_STORE_ARM] {
        let backend = backends
            .iter()
            .find(|backend| backend.id == id)
            .unwrap_or_else(|| panic!("Fix: fixture backend `{id}` must be registered"));
        assert_eq!(
            (backend.semantic_operations)().len(),
            registry_size,
            "Fix: fixture backend `{id}` must declare the whole catalog, exactly as every concrete driver does."
        );
    }

    let complete = every_arm_ops();
    let partial = no_store_arm_ops();
    assert!(
        complete.contains(WITHHELD_ARM) && !partial.contains(WITHHELD_ARM),
        "Fix: the two fixtures must differ in exactly the `{WITHHELD_ARM}` arm."
    );
}
