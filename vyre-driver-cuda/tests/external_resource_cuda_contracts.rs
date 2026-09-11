//! CUDA's answer to the backend-neutral external-import contract.
//!
//! The assertions belong to [`vyre_driver::external_import_contracts`], which
//! is the single statement of what pre-allocation refusal, zero-copy
//! admission, exact schedule execution and device-loss invalidation mean. What
//! is CUDA here is the handles and the combination the CUDA policy refuses.

use vyre_driver::external_import_contracts::{
    assert_external_import_contract, ExternalImportContractCase,
};
use vyre_driver::{ExternalMemoryKind, ImageFormat, ResourcePermittedUsages, TimelineSyncProtocol};
use vyre_driver_cuda::{CudaExternalImportPolicy, CudaExternalMemoryHandle};

#[test]
fn cuda_answers_the_external_import_contract() {
    assert_external_import_contract::<CudaExternalImportPolicy>(ExternalImportContractCase {
        // A DMA-BUF import has no depth/stencil form.
        refused_handle: CudaExternalMemoryHandle::DmaBufFd(4),
        refused_format: ImageFormat::Depth32Float,
        refused_usages: ResourcePermittedUsages::DEPTH_STENCIL_ATTACHMENT,
        refused_memory_kind: ExternalMemoryKind::DmaBuf,
        admitted_handle: CudaExternalMemoryHandle::DmaBufFd(5),
        device_loss_handle: CudaExternalMemoryHandle::HostPointer {
            ptr: 0x7FFF_0000,
            byte_size: 1920 * 8 * 1080,
        },
        sync_protocol: TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 42,
            wait_value: 100,
            signal_value: 101,
        },
    });
}

#[test]
fn cuda_win32_import_refuses_three_plane_planar_video() {
    assert_external_import_contract::<CudaExternalImportPolicy>(ExternalImportContractCase {
        // A Windows NT shared handle carries no 3-plane planar surface.
        refused_handle: CudaExternalMemoryHandle::Win32Nt(0x2000),
        refused_format: ImageFormat::Yuv420Planar,
        refused_usages: ResourcePermittedUsages::SAMPLED,
        refused_memory_kind: ExternalMemoryKind::Win32Nt,
        admitted_handle: CudaExternalMemoryHandle::Win32Nt(0x3000),
        device_loss_handle: CudaExternalMemoryHandle::Win32Nt(0x4000),
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    });
}
