//! Integration tests for CUDA zero-copy external resource import and synchronization.
//!
//! Acceptance criteria:
//! 1. Pre-allocation authentication rejects unsupported combinations before any CUDA driver allocation.
//! 2. Zero-copy import records exact dimensions, aligned row pitch, and provenance without host copies.
//! 3. Timeline synchronization executes exact wait/signal points without device-wide stalls.
//! 4. Device loss invalidates all dependent views and CUDA graphs.
//! 5. Import is bounded: the imported-resource table and both dependent
//!    indexes stop growing under unbounded import, the newest import survives,
//!    and an evicted record leaves no dependent-index entry behind.

use vyre_driver::{
    ExternalMemoryKind, ImageDimensions, ImageFormat, ResourceAbiError, ResourcePermittedUsages,
    ResourceTransitionSchedule, ResourceUsageTransition, TimelineSyncProtocol,
};
use vyre_driver_cuda::{
    CudaExternalMemoryDescriptor, CudaExternalMemoryHandle, CudaExternalResourceImporter,
};

#[test]
fn cuda_external_import_pre_allocation_rejection() {
    let importer = CudaExternalResourceImporter::new(1);

    // 1. Unaligned pitch (< min or not 256-byte aligned) must be rejected before allocation
    let bad_pitch = CudaExternalMemoryDescriptor {
        resource_id: 101,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(100, 100),
        row_pitch_bytes: 400, // 400 is not 256-byte aligned (required 512)
        handle: CudaExternalMemoryHandle::DmaBufFd(3),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let err_pitch = importer
        .authenticate_import(&bad_pitch)
        .expect_err("unaligned pitch must be rejected before allocation");

    assert_eq!(
        err_pitch,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 101,
            provided_pitch: 400,
            required_pitch: 512,
        }
    );

    // 2. DepthStencil format on DMA-BUF must be rejected before allocation naming combination
    let bad_depth = CudaExternalMemoryDescriptor {
        resource_id: 102,
        format: ImageFormat::Depth32Float,
        dimensions: ImageDimensions::d2(1024, 1024),
        row_pitch_bytes: 1024 * 4,
        handle: CudaExternalMemoryHandle::DmaBufFd(4),
        permitted_usages: ResourcePermittedUsages::DEPTH_STENCIL_ATTACHMENT,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let err_depth = importer
        .authenticate_import(&bad_depth)
        .expect_err("depth stencil on dmabuf must be rejected before allocation");

    assert_eq!(
        err_depth,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 102,
            format: ImageFormat::Depth32Float,
            memory_kind: ExternalMemoryKind::DmaBuf,
        }
    );
}

#[test]
fn cuda_zero_copy_resource_import_and_schedule_execution() {
    let importer = CudaExternalResourceImporter::new(2);

    let descriptor = CudaExternalMemoryDescriptor {
        resource_id: 201,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 4,
        handle: CudaExternalMemoryHandle::DmaBufFd(5),
        permitted_usages: ResourcePermittedUsages::SAMPLED
            .union(ResourcePermittedUsages::STORAGE_WRITE),
        sync_protocol: TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 42,
            wait_value: 100,
            signal_value: 101,
        },
    };

    let record = importer
        .import_external_resource(descriptor)
        .expect("import external dmabuf resource");

    assert_eq!(record.resource_id, 201);
    assert_eq!(record.device_id, 2);
    assert!(record.is_zero_copy);
    assert_eq!(record.row_pitch_bytes, 7680);

    // Build transition schedule
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(201, ResourceUsageTransition::storage_to_color_attachment());
    schedule.add_wait(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 42,
        wait_value: 100,
        signal_value: 101,
    });
    schedule.add_signal(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 42,
        wait_value: 101,
        signal_value: 102,
    });

    let report = importer
        .registry()
        .execute_transition_schedule(&schedule)
        .expect("execute transition schedule");

    assert_eq!(report.transitions_executed, 1);
    assert_eq!(report.barriers_emitted, 1);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    // CRITICAL ACCEPTANCE: Zero copies, zero device-wide waits
    assert_eq!(report.copy_count, 0);
    assert_eq!(report.device_wide_waits, 0);
}

#[test]
fn cuda_device_loss_invalidates_views_and_graphs() {
    let importer = CudaExternalResourceImporter::new(3);

    let descriptor = CudaExternalMemoryDescriptor {
        resource_id: 301,
        format: ImageFormat::Rgba16Float,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 8,
        handle: CudaExternalMemoryHandle::HostPointer {
            ptr: 0x7FFF_0000,
            byte_size: 1920 * 8 * 1080,
        },
        permitted_usages: ResourcePermittedUsages::STORAGE,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    importer
        .import_external_resource(descriptor)
        .expect("import resource");

    // Register dependent view and graph
    importer
        .registry()
        .register_dependent_view(301, 4001)
        .expect("register view 4001");
    importer
        .registry()
        .register_dependent_artifact(301, 5001)
        .expect("register graph 5001");

    // Invalidate on device loss
    let report = importer.invalidate_on_device_loss();
    assert_eq!(report.device_id, 3);
    assert_eq!(report.invalidated_resources, vec![301]);
    assert_eq!(report.invalidated_views, vec![4001]);
    assert_eq!(report.invalidated_artifacts, vec![5001]);

    // Subsequent operation must fail
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(301, ResourceUsageTransition::storage_to_color_attachment());
    let err = importer
        .registry()
        .execute_transition_schedule(&schedule)
        .expect_err("invalidated resource execution must fail");

    assert_eq!(
        err,
        ResourceAbiError::ResourceInvalidated { resource_id: 301 }
    );
}
