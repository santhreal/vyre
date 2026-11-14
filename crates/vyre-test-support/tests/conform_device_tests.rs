//! Conformance live-device parity integration tests.

#[path = "conform_suite/backend_acquisition_runtime_sharing.rs"]
pub mod backend_acquisition_runtime_sharing;

#[path = "conform_suite/countless_readwrite_output_parity/mod.rs"]
pub mod countless_readwrite_output_parity;

#[path = "conform_suite/device_profile_derived_facts.rs"]
pub mod device_profile_derived_facts;

#[path = "conform_suite/lens_parity_device.rs"]
pub mod lens_parity_device;

#[path = "conform_suite/parity_matrix.rs"]
pub mod parity_matrix;

#[path = "conform_suite/production_route_device.rs"]
pub mod production_route_device;

#[path = "conform_suite/reference_parity_classes/mod.rs"]
pub mod reference_parity_classes;

#[path = "conform_suite/ulp_audit.rs"]
pub mod ulp_audit;

#[test]
fn vyre_conform_worker_entry() {
    vyre_test_support::conform::run_worker_from_env_or_exit();
}
