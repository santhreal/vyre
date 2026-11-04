//! Smoke test for vyre-libs-device.

use vyre_libs_builder::plumbing::registration::operation_catalog;

#[test]
fn device_link_anchor_retains_library_registrations() {
    // The device crate anchors through the builder crate. Calling its anchor
    // has to keep the library tier reachable in a binary that links the device
    // crate alone, which is the only reason the anchor exists.
    vyre_libs_device::link_anchor();

    assert!(
        operation_catalog::link_anchor() > 0,
        "linking the device crate must retain the library operation registry"
    );
}
