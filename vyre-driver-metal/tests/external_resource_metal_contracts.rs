//! Metal's answer to the backend-neutral external-import contract.
//!
//! The assertions belong to [`vyre_driver::external_import_contracts`], which
//! is the single statement of what pre-allocation refusal, zero-copy
//! admission, exact schedule execution and device-loss invalidation mean. What
//! is Metal here is the handles and the combination the Metal policy refuses.

use vyre_driver::external_import_contracts::{
    assert_external_import_contract, ExternalImportContractCase,
};
use vyre_driver::{
    ExternalMemoryKind, ImageFormat, ResourcePermittedUsages, TimelineSyncProtocol,
};
use vyre_driver_metal::{MetalExternalImportPolicy, MetalExternalMemoryHandle};

#[test]
fn metal_answers_the_external_import_contract() {
    assert_external_import_contract::<MetalExternalImportPolicy>(ExternalImportContractCase {
        // A Metal shared texture has no multi-planar YUV form.
        refused_handle: MetalExternalMemoryHandle::IOSurface(0x1000),
        refused_format: ImageFormat::Yuv420Planar,
        refused_usages: ResourcePermittedUsages::SAMPLED,
        refused_memory_kind: ExternalMemoryKind::MetalSharedResource,
        admitted_handle: MetalExternalMemoryHandle::SharedBuffer {
            ptr: 0x2000,
            byte_size: 1920 * 4 * 1080,
        },
        device_loss_handle: MetalExternalMemoryHandle::SharedTexture(0x4000),
        sync_protocol: TimelineSyncProtocol::MetalSharedEvent {
            event_id: 99,
            signal_value: 1,
        },
    });
}
