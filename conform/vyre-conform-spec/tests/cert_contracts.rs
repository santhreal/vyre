//! Contracts for `vyre_conform_spec::cert`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_conform_spec::cert::fingerprint::{BackendFingerprint, ProbeObservation};

#[test]
fn fingerprint_stable_across_runs() {
    let observation = ProbeObservation::new("wgpu", "nvidia", 32, 0, 1);

    let first = BackendFingerprint::from_observation(&observation);
    let second = BackendFingerprint::from_observation(&observation);

    assert_eq!(first, second);
}

#[test]
fn fingerprint_diverges_on_simulated_driver_change() {
    let old = ProbeObservation::new("wgpu", "nvidia", 32, 0, 1);
    let new = ProbeObservation::new("wgpu", "nvidia", 32, 1, 1);

    assert_ne!(
        BackendFingerprint::from_observation(&old),
        BackendFingerprint::from_observation(&new)
    );
}
