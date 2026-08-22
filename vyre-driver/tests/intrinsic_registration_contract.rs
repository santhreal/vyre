//! Target-facet identity contracts for canonical intrinsic operations.

use vyre_driver::{validate_intrinsic_lowering, IntrinsicRegistrationError};
use vyre_foundation::operation::OperationRegistry;
use vyre_primitives::hardware::all_entries;
use vyre_reference::value::Value;

const INTRINSIC_ID: &str = "vyre-primitives::hardware::bit_reverse_u32";

fn canonical_entry() -> vyre_foundation::operation::SemanticOperation {
    all_entries()
        .find(|entry| entry.id == INTRINSIC_ID)
        .expect("canonical bit-reverse intrinsic")
}

#[test]
fn canonical_identity_signature_and_fixture_reach_reference_interpreter() {
    let entry = canonical_entry();
    let operation = OperationRegistry::global()
        .get(INTRINSIC_ID)
        .expect("canonical semantic operation");
    assert_eq!(operation.id, entry.id);
    assert_eq!(operation.signature, entry.signature);

    let program = operation.program().expect("canonical neutral builder");
    let inputs = entry.test_inputs.expect("canonical input fixture")();
    let expected = entry.expected_output.expect("canonical output fixture")();
    for (inputs, expected) in inputs.into_iter().zip(expected) {
        let values = inputs
            .into_iter()
            .map(|bytes| Value::Bytes(bytes.into()))
            .collect::<Vec<_>>();
        let actual = vyre_reference::reference_eval(&program, &values)
            .expect("reference interpreter executes canonical intrinsic")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[test]
fn target_facet_attaches_by_identity_without_repeating_signature() {
    let owner = validate_intrinsic_lowering(INTRINSIC_ID).expect("canonical target facet");
    assert_eq!(owner.id, INTRINSIC_ID);
    assert!(owner.signature.is_some());
}

#[test]
fn target_facet_cannot_invent_intrinsic_identity() {
    let error = validate_intrinsic_lowering("driver.private.intrinsic")
        .expect_err("driver-private intrinsic id must be rejected");
    assert_eq!(
        error,
        IntrinsicRegistrationError::UnknownId {
            id: "driver.private.intrinsic"
        }
    );
}
