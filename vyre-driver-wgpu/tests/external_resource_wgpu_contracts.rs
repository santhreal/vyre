//! WGPU's answer to the backend-neutral external-import contract.
//!
//! The assertions belong to [`vyre_driver::external_import_contracts`], which
//! is the single statement of what pre-allocation refusal, zero-copy
//! admission, exact schedule execution and device-loss invalidation mean. What
//! is WGPU here is the handles and the combination the WGPU policy refuses.

use vyre_driver::external_import_contracts::{
    assert_external_import_contract, ExternalImportContractCase,
};
use vyre_driver::{
    ExternalMemoryKind, ImageFormat, ResourcePermittedUsages, TimelineSyncProtocol,
};
use vyre_driver_wgpu::{WgpuExternalImportPolicy, WgpuExternalMemoryHandle};

#[test]
fn wgpu_answers_the_external_import_contract() {
    assert_external_import_contract::<WgpuExternalImportPolicy>(ExternalImportContractCase {
        // A depth/stencil surface has no storage-write binding.
        refused_handle: WgpuExternalMemoryHandle::HalTexture(0x1000),
        refused_format: ImageFormat::Depth32Float,
        refused_usages: ResourcePermittedUsages::STORAGE_WRITE,
        refused_memory_kind: ExternalMemoryKind::OpaqueFd,
        admitted_handle: WgpuExternalMemoryHandle::DmaBuf(5),
        device_loss_handle: WgpuExternalMemoryHandle::HostBuffer {
            ptr: 0x5000,
            byte_size: 1920 * 8 * 1080,
        },
        sync_protocol: TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 77,
            wait_value: 10,
            signal_value: 11,
        },
    });
}
