//! Integration tests for WGPU zero-copy external resource import and synchronization.
//!
//! Acceptance criteria:
//! 1. Pre-allocation authentication rejects unsupported combinations (e.g. depth-stencil format with storage write usage).
//! 2. Zero-copy import records exact dimensions, aligned row pitch, and provenance without host copies.
//! 3. Timeline synchronization executes exact wait/signal points without device-wide stalls.
//! 4. Device loss invalidates all dependent views and pipelines.

use vyre_driver::{
    ExternalMemoryKind, ImageDimensions, ImageFormat, ResourceAbiError, ResourcePermittedUsages,
    ResourceTransitionSchedule, ResourceUsageTransition, TimelineSyncProtocol,
};
use vyre_driver_wgpu::{
    WgpuExternalMemoryDescriptor, WgpuExternalMemoryHandle, WgpuExternalResourceImporter,
};

#[test]
fn wgpu_external_import_pre_allocation_rejection() {
    let importer = WgpuExternalResourceImporter::new(1);

    // 1. DepthStencil format with STORAGE_WRITE usage must be rejected before allocation
    let bad_depth_storage = WgpuExternalMemoryDescriptor {
        resource_id: 101,
        format: ImageFormat::Depth32Float,
        dimensions: ImageDimensions::d2(1024, 1024),
        row_pitch_bytes: 1024 * 4,
        handle: WgpuExternalMemoryHandle::HalTexture(0x1000),
        permitted_usages: ResourcePermittedUsages::STORAGE_WRITE,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let err_depth = importer
        .authenticate_import(&bad_depth_storage)
        .expect_err("depth stencil with storage write must be rejected before allocation");

    assert_eq!(
        err_depth,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 101,
            format: ImageFormat::Depth32Float,
            memory_kind: ExternalMemoryKind::OpaqueFd,
        }
    );

    // 2. Unaligned pitch (< min or not 256-byte aligned) must be rejected before allocation
    let bad_pitch = WgpuExternalMemoryDescriptor {
        resource_id: 102,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(100, 100),
        row_pitch_bytes: 400, // 400 is not 256-byte aligned
        handle: WgpuExternalMemoryHandle::DmaBuf(3),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let err_pitch = importer
        .authenticate_import(&bad_pitch)
        .expect_err("unaligned pitch must be rejected");

    assert_eq!(
        err_pitch,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 102,
            provided_pitch: 400,
            required_pitch: 512,
        }
    );
}

#[test]
fn wgpu_zero_copy_resource_import_and_schedule_execution() {
    let importer = WgpuExternalResourceImporter::new(2);

    let descriptor = WgpuExternalMemoryDescriptor {
        resource_id: 201,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 4,
        handle: WgpuExternalMemoryHandle::DmaBuf(5),
        permitted_usages: ResourcePermittedUsages::SAMPLED
            .union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        sync_protocol: TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 77,
            wait_value: 10,
            signal_value: 11,
        },
    };

    let record = importer
        .import_external_resource(descriptor)
        .expect("import external wgpu resource");

    assert_eq!(record.resource_id, 201);
    assert_eq!(record.device_id, 2);
    assert!(record.is_zero_copy);
    assert_eq!(record.row_pitch_bytes, 7680);

    // Transition schedule
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(201, ResourceUsageTransition::storage_to_color_attachment());
    schedule.add_wait(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 77,
        wait_value: 10,
        signal_value: 11,
    });
    schedule.add_signal(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 77,
        wait_value: 11,
        signal_value: 12,
    });

    let report = importer
        .execute_transition_schedule(&schedule)
        .expect("execute transition schedule");

    assert_eq!(report.transitions_executed, 1);
    assert_eq!(report.barriers_emitted, 1);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    assert_eq!(report.copy_count, 0);
    assert_eq!(report.device_wide_waits, 0);
}

#[test]
fn wgpu_device_loss_invalidates_views_and_pipelines() {
    let importer = WgpuExternalResourceImporter::new(3);

    let descriptor = WgpuExternalMemoryDescriptor {
        resource_id: 301,
        format: ImageFormat::Rgba16Float,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 8,
        handle: WgpuExternalMemoryHandle::HostBuffer {
            ptr: 0x5000,
            byte_size: 1920 * 8 * 1080,
        },
        permitted_usages: ResourcePermittedUsages::STORAGE,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    importer
        .import_external_resource(descriptor)
        .expect("import resource");

    // Register dependent view and pipeline
    importer
        .register_dependent_view(301, 8001)
        .expect("register view 8001");
    importer
        .register_dependent_pipeline(301, 9001)
        .expect("register pipeline 9001");

    // Invalidate on device loss
    let report = importer.invalidate_on_device_loss();
    assert_eq!(report.device_id, 3);
    assert_eq!(report.invalidated_resources, vec![301]);
    assert_eq!(report.invalidated_views, vec![8001]);
    assert_eq!(report.invalidated_artifacts, vec![9001]);

    // Subsequent operation must fail
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(301, ResourceUsageTransition::storage_to_color_attachment());
    let err = importer
        .execute_transition_schedule(&schedule)
        .expect_err("operation on invalidated resource must fail");

    assert_eq!(
        err,
        ResourceAbiError::ResourceInvalidated { resource_id: 301 }
    );
}
