//! Integration test crate for the containing Vyre package.

use vyre_driver::backend::{
    acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
};
use vyre_driver_cuda::CudaBackendRegistration;

#[test]
fn cuda_registry_metadata_is_release_preferred_dispatch() {
    let _link_cuda_inventory_registration = std::any::TypeId::of::<CudaBackendRegistration>();

    assert!(
        backend_dispatches("cuda"),
        "Fix: CUDA backend must register dispatches=true for release dispatch selection"
    );
    assert_eq!(
        backend_precedence("cuda"),
        5,
        "Fix: CUDA must keep canonical release precedence rank 5"
    );
}

#[test]
fn preferred_dispatch_backend_selects_cuda_not_reference_oracle() {
    let _link_cuda_inventory_registration = std::any::TypeId::of::<CudaBackendRegistration>();
    let backend = acquire_preferred_dispatch_backend().expect(
        "Fix: preferred dispatch acquisition must initialize the linked CUDA backend on the local GPU host.",
    );
    let id = backend.id();

    assert_eq!(
        id, "cuda",
        "Fix: CUDA has release precedence when linked; preferred dispatch got `{id}`"
    );
    assert_ne!(id, "reference");
    assert_ne!(id, "cpu-ref");
}
