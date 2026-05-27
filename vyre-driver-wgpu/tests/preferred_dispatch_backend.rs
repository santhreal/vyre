//! Integration test crate for the containing Vyre package.

#![allow(deprecated)]
use vyre_driver::backend::{
    acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
};
use vyre_driver_wgpu::WgpuBackend;

#[test]
fn wgpu_registry_metadata_is_dispatch_capable_gpu() {
    let _link_wgpu_inventory_registration = std::any::TypeId::of::<WgpuBackend>();

    assert!(
        backend_dispatches("wgpu"),
        "Fix: WGPU backend must register dispatches=true for preferred GPU dispatch selection"
    );
    assert_eq!(
        backend_precedence("wgpu"),
        30,
        "Fix: WGPU precedence rank must stay below CUDA but above reference/oracle backends"
    );
}

#[test]
fn preferred_dispatch_backend_selects_wgpu_not_reference_oracle() {
    let _link_wgpu_inventory_registration = std::any::TypeId::of::<WgpuBackend>();
    let backend = acquire_preferred_dispatch_backend().expect(
        "Fix: preferred dispatch acquisition must initialize the linked WGPU backend on the local GPU host.",
    );
    let id = backend.id();

    assert_eq!(
        id, "wgpu",
        "Fix: preferred dispatch must select the linked GPU backend, got `{id}`"
    );
    assert_ne!(id, "reference");
    assert_ne!(id, "cpu-ref");
}
