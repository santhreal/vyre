//! Canonical IEEE 754 contracts.

use vyre_reference::ieee754::{canonical_f32, canonical_ulp_distance};

#[test]
fn canonical_f32_collapses_nan_payloads() {
    let quiet_payload = f32::from_bits(0x7FC1_2345);
    let signaling_payload = f32::from_bits(0x7F81_2345);
    assert_eq!(canonical_f32(quiet_payload).to_bits(), 0x7FC0_0000);
    assert_eq!(canonical_f32(signaling_payload).to_bits(), 0x7FC0_0000);
    assert_eq!(canonical_ulp_distance(quiet_payload, signaling_payload), 0);
}

#[test]
fn canonical_ulp_distance_handles_zero_and_neighbors() {
    assert_eq!(canonical_ulp_distance(0.0, -0.0), 0);
    assert_eq!(
        canonical_ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)),
        1
    );
}
