//! Integration test verifying global operation registration completeness.

use vyre_test_support::operation_registration_universe::{
    assert_operation_registry_complete, registered_operation_ids, OPERATION_REGISTRATION_FLOOR,
};

#[test]
fn operation_registration_universe_is_complete_and_well_formed() {
    assert_operation_registry_complete();
    let ids = registered_operation_ids();
    assert!(ids.len() >= OPERATION_REGISTRATION_FLOOR);
}
