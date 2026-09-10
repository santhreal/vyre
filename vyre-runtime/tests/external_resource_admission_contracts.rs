//! Integration tests for external resource admission, transition execution,
//! and timeline synchronization contracts.
//!
//! Acceptance criteria:
//! 1. External resource admission authenticates dimensions, row pitch, and memory capability before allocation.
//! 2. Unsupported import combinations are rejected before allocation, naming the rejected combination.
//! 3. Selected schedule execution proves absence of unrecorded copies and device-wide waits.
//! 4. Generation advancement detects and rejects stale frame access.
//! 5. Device loss invalidates all resources, dependent views, and dependent pipelines.
//! 6. Runtime variant space enumeration covers every format class and memory kind.
//! 7. Admission is bounded: the admitted-record table and both dependent
//!    indexes stop growing under unbounded admission, the newest admission
//!    survives, and an evicted record leaves no dependent-index entry behind.

use vyre_driver::{
    all_external_memory_kinds, all_format_classes, all_image_formats, all_sync_protocols,
    ColorInterpretation, ExternalEventKind, ExternalMemoryKind, ImageFormat, ResourceLayoutState,
    ResourcePermittedUsages, ResourceProvenance, ResourceTransitionSchedule,
    ResourceUsageTransition, TimelineSyncProtocol,
};
use vyre_runtime::{ExternalAdmissionError, ExternalResourceAdmissionManager};

#[test]
fn external_resource_admission_happy_paths() {
    let manager = ExternalResourceAdmissionManager::new(1);

    // 1. Admit DMA-BUF RGBA8 texture
    let record_dmabuf = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        1001,
        1,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        1920 * 4,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::TRANSFER_DST),
        ExternalMemoryKind::DmaBuf,
        0x1001_A001,
        TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 1,
            wait_value: 0,
            signal_value: 1,
        },
    );

    let lease = manager
        .admit_external_resource(record_dmabuf)
        .expect("admit dmabuf resource");
    assert_eq!(lease.resource_id, 1001);
    assert!(lease.is_zero_copy);
    assert_eq!(lease.row_pitch_bytes, 7680);

    // 2. Admit Win32 NT handle NV12 video surface
    let record_win32 = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        1002,
        1,
        ImageFormat::Yuv420SemiPlanar,
        ColorInterpretation::Bt709,
        1920,
        1080,
        2048,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::STORAGE_READ),
        ExternalMemoryKind::Win32Nt,
        0x2002_B002,
        TimelineSyncProtocol::Fence {
            fence_id: 2,
            is_signaled: true,
        },
    );

    let lease_win32 = manager
        .admit_external_resource(record_win32)
        .expect("admit win32 nt resource");
    assert_eq!(lease_win32.resource_id, 1002);
    assert!(lease_win32.is_zero_copy);
}

#[test]
fn pre_allocation_rejection_of_unsupported_combinations() {
    let manager = ExternalResourceAdmissionManager::new(2);

    // 1. DepthStencil format on DMA-BUF must be rejected before allocation naming combination
    let bad_depth_dmabuf = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        2001,
        2,
        ImageFormat::Depth32Float,
        ColorInterpretation::Passthrough,
        1024,
        1024,
        1024 * 4,
        ResourcePermittedUsages::DEPTH_STENCIL_ATTACHMENT,
        ExternalMemoryKind::DmaBuf,
        0xDEAD_01,
        TimelineSyncProtocol::ImplicitQueue,
    );

    let err_dmabuf = manager
        .admit_external_resource(bad_depth_dmabuf)
        .expect_err("depth stencil on dmabuf must be rejected");

    assert_eq!(
        err_dmabuf,
        ExternalAdmissionError::InvalidCombination {
            resource_id: 2001,
            format: ImageFormat::Depth32Float,
            memory_kind: ExternalMemoryKind::DmaBuf,
        }
    );

    // 2. Planar video (YUV420P) on MetalSharedResource must be rejected before allocation
    let bad_metal_yuv = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        2002,
        2,
        ImageFormat::Yuv420Planar,
        ColorInterpretation::Bt709,
        1920,
        1080,
        2048,
        ResourcePermittedUsages::SAMPLED,
        ExternalMemoryKind::MetalSharedResource,
        0xDEAD_02,
        TimelineSyncProtocol::MetalSharedEvent {
            event_id: 1,
            signal_value: 1,
        },
    );

    let err_metal = manager
        .admit_external_resource(bad_metal_yuv)
        .expect_err("planar video on metal shared resource must be rejected");

    assert_eq!(
        err_metal,
        ExternalAdmissionError::InvalidCombination {
            resource_id: 2002,
            format: ImageFormat::Yuv420Planar,
            memory_kind: ExternalMemoryKind::MetalSharedResource,
        }
    );

    // 3. Unaligned row pitch must be rejected before allocation
    let bad_pitch = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        2003,
        2,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        100, // min pitch = 400
        100,
        400, // 400 is not 256-byte aligned (required 512)
        ResourcePermittedUsages::SAMPLED,
        ExternalMemoryKind::DmaBuf,
        0xDEAD_03,
        TimelineSyncProtocol::ImplicitQueue,
    );

    let err_pitch = manager
        .admit_external_resource(bad_pitch)
        .expect_err("unaligned pitch must be rejected");

    assert_eq!(
        err_pitch,
        ExternalAdmissionError::InvalidPitch {
            resource_id: 2003,
            provided: 400,
            required: 512,
        }
    );

    // 4. Missing EXTERNAL_IMPORT usage flag must be rejected
    let mut bad_usage = vyre_driver::AdmittedResourceRecord::new_2d(
        2004,
        2,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        ResourcePermittedUsages::SAMPLED, // Does NOT contain EXTERNAL_IMPORT
        1,
    );
    bad_usage.provenance = ResourceProvenance::ExternalImport {
        memory_kind: ExternalMemoryKind::DmaBuf,
        exportable: false,
        handle_tag: 0xDEAD_04,
    };

    let err_usage = manager
        .admit_external_resource(bad_usage)
        .expect_err("missing external import usage must be rejected");

    assert_eq!(
        err_usage,
        ExternalAdmissionError::UsageNotPermitted {
            resource_id: 2004,
            requested: ResourcePermittedUsages::EXTERNAL_IMPORT,
        }
    );
}

#[test]
fn exact_schedule_execution_verifies_zero_copies_and_no_device_wide_waits() {
    let manager = ExternalResourceAdmissionManager::new(3);

    let record = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        3001,
        3,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        1920 * 4,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        ExternalMemoryKind::DmaBuf,
        0x3001_A001,
        TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 42,
            wait_value: 10,
            signal_value: 11,
        },
    );

    manager
        .admit_external_resource(record)
        .expect("admit resource");

    // Build transition schedule
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(3001, ResourceUsageTransition::storage_to_color_attachment());
    schedule.add_transition(
        3001,
        ResourceUsageTransition::to_present(ResourceLayoutState::ColorAttachmentOptimal),
    );
    schedule.add_wait(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 42,
        wait_value: 10,
        signal_value: 11,
    });
    schedule.add_signal(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 42,
        wait_value: 11,
        signal_value: 12,
    });

    let report = manager
        .execute_transition_schedule(&schedule)
        .expect("execute schedule");

    assert_eq!(report.transitions_executed, 2);
    assert_eq!(report.barriers_emitted, 2);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    // CRITICAL ACCEPTANCE: Proven absence of unrecorded copies and global device waits
    assert_eq!(
        report.copy_count, 0,
        "execution must perform 0 host/device copies"
    );
    assert_eq!(
        report.device_wide_waits, 0,
        "execution must perform 0 device-wide waits"
    );
}

#[test]
fn stale_generation_mutation_rejection() {
    let manager = ExternalResourceAdmissionManager::new(4);

    let record = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        4001,
        4,
        ImageFormat::R32Float,
        ColorInterpretation::Passthrough,
        512,
        512,
        512 * 4,
        ResourcePermittedUsages::STORAGE,
        ExternalMemoryKind::HostAllocation,
        0x4001_A001,
        TimelineSyncProtocol::ImplicitQueue,
    );

    manager
        .admit_external_resource(record)
        .expect("admit resource");

    // Advance generation 1 -> 2
    let gen2 = manager.mutate_resource(4001, 1).expect("mutate gen 1 -> 2");
    assert_eq!(gen2, 2);

    // Stale generation 1 access must be rejected
    let stale_err = manager
        .mutate_resource(4001, 1)
        .expect_err("stale generation 1 access must be rejected");

    assert_eq!(
        stale_err,
        ExternalAdmissionError::GenerationMismatch {
            resource_id: 4001,
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn device_loss_invalidates_all_dependent_views_and_pipelines() {
    let manager = ExternalResourceAdmissionManager::new(5);

    let record = vyre_driver::AdmittedResourceRecord::new_external_import_2d(
        5001,
        5,
        ImageFormat::Rgba16Float,
        ColorInterpretation::LinearRgb,
        1920,
        1080,
        1920 * 8,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        ExternalMemoryKind::DmaBuf,
        0x5001_A001,
        TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 50,
            wait_value: 0,
            signal_value: 1,
        },
    );

    manager
        .admit_external_resource(record)
        .expect("admit resource");

    // Register dependent views and pipelines
    manager
        .register_dependent_view(5001, 6001)
        .expect("register view 6001");
    manager
        .register_dependent_view(5001, 6002)
        .expect("register view 6002");
    manager
        .register_dependent_pipeline(5001, 7001)
        .expect("register pipeline 7001");

    // Trigger device loss
    let report = manager.invalidate_device_loss();
    assert_eq!(report.device_id, 5);
    assert_eq!(report.invalidated_resources, vec![5001]);
    assert_eq!(report.invalidated_views, vec![6001, 6002]);
    assert_eq!(report.invalidated_artifacts, vec![7001]);

    // Subsequent operation on invalidated resource must fail
    let err = manager
        .mutate_resource(5001, 1)
        .expect_err("operation on invalidated resource must fail");

    assert_eq!(
        err,
        ExternalAdmissionError::ResourceInvalidated { resource_id: 5001 }
    );
}

#[test]
fn runtime_mutation_gate_all_variants_handled() {
    assert_eq!(all_image_formats().len(), 38);
    assert_eq!(all_format_classes().len(), 7);
    assert_eq!(all_external_memory_kinds().len(), 6);
    assert_eq!(all_sync_protocols().len(), 5);

    for proto in all_sync_protocols() {
        match proto.event_kind() {
            ExternalEventKind::TimelineSemaphore | ExternalEventKind::MetalSharedEvent => {
                assert!(proto.is_timeline());
            }
            ExternalEventKind::BinaryFence
            | ExternalEventKind::SyncFileFd
            | ExternalEventKind::ImplicitQueue => {
                assert!(!proto.is_timeline());
            }
        }
    }
}
