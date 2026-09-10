//! Integration tests for Metal zero-copy external resource import and synchronization.
//!
//! Acceptance criteria:
//! 1. Pre-allocation authentication rejects unsupported combinations (e.g. planar video formats on Metal shared textures).
//! 2. Zero-copy import records exact dimensions, aligned row pitch, and provenance without host copies.
//! 3. Timeline synchronization executes exact wait/signal points without device-wide stalls.
//! 4. Device loss invalidates all dependent views and pipelines.
//! 5. Import is bounded: the imported-resource table and both dependent
//!    indexes stop growing under unbounded import, the newest import survives,
//!    and an evicted record leaves no dependent-index entry behind.

use vyre_driver::{
    ExternalMemoryKind, ImageDimensions, ImageFormat, ResourceAbiError, ResourceLayoutState,
    ResourcePermittedUsages, ResourceTransitionSchedule, ResourceUsageTransition,
    TimelineSyncProtocol,
};
use vyre_driver_metal::{
    MetalExternalMemoryDescriptor, MetalExternalMemoryHandle, MetalExternalResourceImporter,
};

#[test]
fn metal_external_import_pre_allocation_rejection() {
    let importer = MetalExternalResourceImporter::new(1);

    // 1. Planar video format (Yuv420Planar) on Metal shared resource must be rejected before allocation
    let bad_planar = MetalExternalMemoryDescriptor {
        resource_id: 101,
        format: ImageFormat::Yuv420Planar,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920,
        handle: MetalExternalMemoryHandle::IOSurface(0x1000),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let err_planar = importer
        .authenticate_import(&bad_planar)
        .expect_err("planar video on metal shared texture must be rejected");

    assert_eq!(
        err_planar,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 101,
            format: ImageFormat::Yuv420Planar,
            memory_kind: ExternalMemoryKind::MetalSharedResource,
        }
    );

    // 2. Unaligned pitch (< min or not 256-byte aligned) must be rejected before allocation
    let bad_pitch = MetalExternalMemoryDescriptor {
        resource_id: 102,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(100, 100),
        row_pitch_bytes: 400, // 400 is not 256-byte aligned
        handle: MetalExternalMemoryHandle::SharedBuffer {
            ptr: 0x2000,
            byte_size: 400 * 100,
        },
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
fn metal_zero_copy_resource_import_and_schedule_execution() {
    let importer = MetalExternalResourceImporter::new(2);

    let descriptor = MetalExternalMemoryDescriptor {
        resource_id: 201,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 4,
        handle: MetalExternalMemoryHandle::IOSurface(0x3000),
        permitted_usages: ResourcePermittedUsages::SAMPLED
            .union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        sync_protocol: TimelineSyncProtocol::MetalSharedEvent {
            event_id: 99,
            signal_value: 1,
        },
    };

    let record = importer
        .import_external_resource(descriptor)
        .expect("import external metal resource");

    assert_eq!(record.resource_id, 201);
    assert_eq!(record.device_id, 2);
    assert!(record.is_zero_copy);
    assert_eq!(record.row_pitch_bytes, 7680);

    // Transition schedule
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(
        201,
        ResourceUsageTransition::to_sampled(ResourceLayoutState::ColorAttachmentOptimal),
    );
    schedule.add_wait(TimelineSyncProtocol::MetalSharedEvent {
        event_id: 99,
        signal_value: 1,
    });
    schedule.add_signal(TimelineSyncProtocol::MetalSharedEvent {
        event_id: 99,
        signal_value: 2,
    });

    let report = importer
        .execute_transition_schedule(&schedule)
        .expect("execute schedule");

    assert_eq!(report.transitions_executed, 1);
    assert_eq!(report.barriers_emitted, 1);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    assert_eq!(report.copy_count, 0);
    assert_eq!(report.device_wide_waits, 0);
}

#[test]
fn metal_device_loss_invalidates_views_and_pipelines() {
    let importer = MetalExternalResourceImporter::new(3);

    let descriptor = MetalExternalMemoryDescriptor {
        resource_id: 301,
        format: ImageFormat::Rgba16Float,
        dimensions: ImageDimensions::d2(1920, 1080),
        row_pitch_bytes: 1920 * 8,
        handle: MetalExternalMemoryHandle::SharedTexture(0x4000),
        permitted_usages: ResourcePermittedUsages::STORAGE,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    importer
        .import_external_resource(descriptor)
        .expect("import resource");

    // Register dependent view and pipeline
    importer
        .register_dependent_view(301, 6001)
        .expect("register view 6001");
    importer
        .register_dependent_pipeline(301, 7001)
        .expect("register pipeline 7001");

    // Invalidate on device loss
    let report = importer.invalidate_on_device_loss();
    assert_eq!(report.device_id, 3);
    assert_eq!(report.invalidated_resources, vec![301]);
    assert_eq!(report.invalidated_views, vec![6001]);
    assert_eq!(report.invalidated_artifacts, vec![7001]);

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

/// How many imports every bounded-import case drives past the ceiling.
///
/// The capacity constant is private to `vyre-driver-metal`, so no test can
/// read it. This number only has to exceed it; every assertion below compares
/// measured live counts against each other rather than against a pinned
/// ceiling.
const IMPORTS_PAST_THE_CEILING: u64 = 2048;

/// A descriptor that passes authentication, distinguished only by `resource_id`.
fn importable_descriptor(resource_id: u64) -> MetalExternalMemoryDescriptor {
    MetalExternalMemoryDescriptor {
        resource_id,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(64, 64),
        row_pitch_bytes: 256,
        handle: MetalExternalMemoryHandle::IOSurface(resource_id as usize),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    }
}

/// Import `count` distinct resources into a fresh importer and return how many
/// records it still holds, read from the device-loss report.
fn live_records_after(count: u64) -> usize {
    let importer = MetalExternalResourceImporter::new(7);
    for resource_id in 1..=count {
        importer
            .import_external_resource(importable_descriptor(resource_id))
            .expect("authenticated descriptor is imported");
    }
    importer
        .invalidate_on_device_loss()
        .invalidated_resources
        .len()
}

#[test]
fn metal_imported_resource_table_stops_growing_under_unbounded_import() {
    let at_ceiling = live_records_after(IMPORTS_PAST_THE_CEILING);
    assert!(
        at_ceiling < IMPORTS_PAST_THE_CEILING as usize,
        "the table held {at_ceiling} of {IMPORTS_PAST_THE_CEILING} imports, so it is unbounded"
    );
    assert_eq!(
        at_ceiling,
        live_records_after(IMPORTS_PAST_THE_CEILING + 512),
        "512 imports past the ceiling moved the live count"
    );
}

#[test]
fn metal_eviction_takes_the_oldest_import_and_spares_the_newest() {
    let importer = MetalExternalResourceImporter::new(7);
    for resource_id in 1..=IMPORTS_PAST_THE_CEILING {
        importer
            .import_external_resource(importable_descriptor(resource_id))
            .expect("authenticated descriptor is imported");
    }

    importer
        .register_dependent_view(IMPORTS_PAST_THE_CEILING, 1)
        .expect("the newest import is still held");

    let err = importer
        .register_dependent_view(1, 2)
        .expect_err("the oldest import was evicted");
    assert_eq!(
        err,
        ResourceAbiError::ResourceInvalidated { resource_id: 1 }
    );

    let surviving = importer.invalidate_on_device_loss().invalidated_resources;
    let oldest_survivor = *surviving.first().expect("the table is not empty");
    let expected: Vec<u64> = (oldest_survivor..=IMPORTS_PAST_THE_CEILING).collect();
    assert_eq!(
        surviving, expected,
        "the survivors are not the contiguous newest imports"
    );
}

#[test]
fn metal_evicting_a_record_drops_its_dependent_view_and_pipeline_entries() {
    let importer = MetalExternalResourceImporter::new(7);
    importer
        .import_external_resource(importable_descriptor(1))
        .expect("authenticated descriptor is imported");
    importer
        .register_dependent_view(1, 8001)
        .expect("register view on the first import");
    importer
        .register_dependent_pipeline(1, 9001)
        .expect("register pipeline on the first import");

    for resource_id in 2..=IMPORTS_PAST_THE_CEILING {
        importer
            .import_external_resource(importable_descriptor(resource_id))
            .expect("authenticated descriptor is imported");
    }

    let report = importer.invalidate_on_device_loss();
    assert!(
        !report.invalidated_resources.contains(&1),
        "resource 1 was not evicted, so this case proves nothing"
    );
    assert_eq!(
        report.invalidated_views,
        Vec::<u64>::new(),
        "the evicted record left a dependent view entry behind"
    );
    assert_eq!(
        report.invalidated_artifacts,
        Vec::<u64>::new(),
        "the evicted record left a dependent pipeline entry behind"
    );
}
