//! Canonical semantic operation registry regression contracts.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{
    target_operation_facets, OperationRegistration, OperationRegistry, OperationTier,
    TargetOperationFacet, TolerancePolicy,
};

const OP_ID: &str = "fixture.operation_registry.identity";

fn fixture_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("output", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("output", Expr::u32(0), Expr::u32(7))],
    )
}

inventory::submit! {
    OperationRegistration::new(
        OP_ID,
        OperationTier::Primitive,
        Some(fixture_program),
        None,
        None,
    )
    .with_category("test")
    .with_tolerance(TolerancePolicy::f32_ulp(2))
}

inventory::submit! {
    TargetOperationFacet {
        operation_id: OP_ID,
        target_id: "fixture-target",
        version: 1,
    }
}

/// WHY: one semantic identity must resolve its program, derived effects,
/// capabilities, fixture policy, tolerance, and independently registered target
/// facets from one process-wide catalog.
#[test]
fn operation_identity_resolves_semantics_and_target_facets() {
    let registration = OperationRegistry::global()
        .get(OP_ID)
        .expect("fixture operation must be discoverable");

    assert_eq!(registration.semantic_version, 1);
    assert_eq!(registration.category, Some("test"));
    assert_eq!(registration.tolerance.f32_ulp, 2);
    let program = registration.program().expect("neutral program must build");
    assert_eq!(program.entry_op_id(), Some(OP_ID));
    let effects = registration.effects().expect("effects must derive");
    assert!(effects.reads);
    assert!(effects.writes);
    assert!(!effects.atomics);
    let capabilities = registration
        .required_capabilities()
        .expect("capabilities must derive");
    assert_eq!(capabilities.max_workgroup_size, [1, 1, 1]);

    let facets = target_operation_facets()
        .filter(|facet| facet.operation_id == OP_ID)
        .collect::<Vec<_>>();
    assert_eq!(facets.len(), 1);
    assert_eq!(facets[0].target_id, "fixture-target");
    assert_eq!(facets[0].version, 1);
}
