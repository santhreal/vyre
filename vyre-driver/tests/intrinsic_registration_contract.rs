//! Shared-driver adaptation contracts for canonical intrinsic descriptors.

use vyre_driver::{
    validate_intrinsic_lowering, DialectRegistry, IntrinsicRegistrationError, LoweringCtx, OpDef,
};
use vyre_foundation::dialect_lookup::PrimaryTextBuilder;
use vyre_intrinsics::harness::{all_entries, F32_UNARY_SIGNATURE};
use vyre_reference::value::Value;

const INTRINSIC_ID: &str = "vyre-intrinsics::hardware::bit_reverse_u32";

fn canonical_entry() -> &'static vyre_intrinsics::harness::OpEntry {
    all_entries()
        .find(|entry| entry.id == INTRINSIC_ID)
        .expect("canonical bit-reverse intrinsic")
}

fn registered_definition() -> &'static OpDef {
    let registry = DialectRegistry::global();
    let id = registry.intern_op(INTRINSIC_ID);
    registry
        .lookup(id)
        .expect("driver registry must consume the neutral intrinsic catalog")
}

fn concrete_text_lowering(_: &LoweringCtx<'_>) -> Result<(), String> {
    Ok(())
}

/// Positive: one canonical id and signature reaches the driver registry, and
/// its neutral builder executes against the same deterministic reference
/// fixture owned by `vyre-intrinsics`.
#[test]
fn canonical_identity_signature_and_fixture_reach_reference_registry() {
    let entry = canonical_entry();
    let definition = registered_definition();
    assert_eq!(definition.id, entry.id);
    assert_eq!(definition.signature, entry.signature);

    let program = definition.program().expect("canonical neutral builder");
    let inputs = entry.test_inputs.expect("canonical input fixture")();
    let expected = entry.expected_output.expect("canonical output fixture")();
    for (inputs, expected) in inputs.into_iter().zip(expected) {
        let values = inputs
            .into_iter()
            .map(|bytes| Value::Bytes(bytes.into()))
            .collect::<Vec<_>>();
        let actual = vyre_reference::reference_eval(&program, &values)
            .expect("reference interpreter must execute canonical intrinsic builder")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

/// Positive concrete path: a backend attaches its lowering implementation to
/// the shared definition without restating intrinsic identity or signature.
#[test]
fn concrete_lowering_accepts_canonical_definition() {
    let mut lowering = registered_definition().clone();
    let builder: PrimaryTextBuilder = concrete_text_lowering;
    lowering.lowerings.primary_text = Some(builder);
    let owner = validate_intrinsic_lowering(&lowering).expect("canonical lowering registration");
    assert_eq!(owner.id, INTRINSIC_ID);
    assert_eq!(owner.signature, lowering.signature);
}

/// Negative: a lowering cannot invent a driver-owned intrinsic identity.
#[test]
fn unknown_concrete_lowering_identity_is_rejected() {
    let mut lowering = registered_definition().clone();
    lowering.id = "driver.private.intrinsic";
    let error = match validate_intrinsic_lowering(&lowering) {
        Ok(_) => panic!("driver-private intrinsic id must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        IntrinsicRegistrationError::UnknownId {
            id: "driver.private.intrinsic"
        }
    );
}

/// Boundary: the canonical definition remains valid when a backend has not
/// attached any concrete lowering slot yet.
#[test]
fn reference_only_definition_is_valid_at_registration_boundary() {
    let definition = registered_definition();
    assert!(definition.lowerings.primary_text.is_none());
    assert_eq!(
        validate_intrinsic_lowering(definition)
            .expect("reference-only canonical definition")
            .id,
        INTRINSIC_ID
    );
}

/// Adversarial: a backend cannot reuse a known id with an ABI-compatible shape
/// but a different scalar signature.
#[test]
fn concrete_lowering_with_mismatched_signature_is_rejected() {
    let mut lowering = registered_definition().clone();
    lowering.signature = F32_UNARY_SIGNATURE;
    let error = match validate_intrinsic_lowering(&lowering) {
        Ok(_) => panic!("mismatched intrinsic signature must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        IntrinsicRegistrationError::SignatureMismatch { id: INTRINSIC_ID }
    );
}

/// Adversarial duplicate ownership is rejected before the frozen lookup can
/// silently replace either definition.
#[test]
fn duplicate_driver_definition_is_rejected() {
    let first = registered_definition().clone();
    let second = registered_definition().clone();
    let error = DialectRegistry::validate_no_duplicates([&first, &second])
        .expect_err("duplicate stable intrinsic id must fail");
    assert_eq!(error.op_id(), INTRINSIC_ID);
}
