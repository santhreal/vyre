//! Canonical semantic operation registry regression contracts.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::operation::{
    OperationRegistration, OperationRegistry, OperationTier, TolerancePolicy,
};

const OP_ID: &str = "external_fixture::operation_registry::identity";

fn fixture_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("output", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("output", Expr::u32(0), Expr::u32(7))],
    )
}

inventory::submit! {
    OperationRegistration::new_unconstrained(
        OP_ID,
        OperationTier::External,
        Some(fixture_program),
        None,
        None,
    )
    .with_category("test")
    .with_tolerance(TolerancePolicy::f32_ulp(2))
}

/// WHY: one semantic identity must resolve its program, derived effects,
/// capabilities, fixture policy, and tolerance from one process-wide catalog.
#[test]
fn operation_identity_resolves_semantics() {
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
}

/// WHY: target IDs are opaque owner data, but malformed identities must fail
/// before they become cache, artifact, or routing keys.
#[test]
fn target_identity_rejects_empty_and_padded_values() {
    use vyre_foundation::operation::TargetId;

    assert!(TargetId::new("").is_err());
    assert!(TargetId::new(" target").is_err());
    assert!(TargetId::new("target ").is_err());
    assert_eq!(
        TargetId::new("external-target")
            .expect("canonical target identity")
            .as_str(),
        "external-target"
    );
}
