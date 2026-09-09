//! Tests for domain-neutral semantic resource ABI, zero-copy import/export,
//! layout transitions, and timeline synchronization (Row 111).
//!
//! Acceptance criteria:
//! 1. Domain-neutral typed image, plane, subresource, and timeline capabilities.
//! 2. Admitted resource records with exact dimensions, row pitch alignment, format, and ownership.
//! 3. Deterministic generation progression preventing stale-frame access.
//! 4. Failure-atomic invalidation on device loss of resources, views, and dependent artifacts.
//! 5. Pre-allocation authentication rejecting unsupported import combinations naming the rejected combination.
//! 6. Exact schedule execution proving absence of unrecorded copies and device-wide waits.
//! 7. Run-time variant space enumeration ensuring every usage, format class, and protocol has an admission decision.

use vyre_driver::{
    all_alias_set_kinds, all_external_memory_kinds, all_format_classes, all_image_formats,
    all_layout_states, all_lifetime_state_kinds, all_provenance_kinds, all_sync_protocols,
    all_usage_flags, authenticate_external_import, AdmittedResourceRecord,
    AdmittedResourceRecordExt, ColorInterpretation, ExternalEventKind, ExternalMemoryKind,
    ExternalResourceRegistry, FormatClass, ImageDimensions, ImageFormat, ResidentOwner,
    ResourceAbiError, ResourceAliasSet, ResourceLayoutState, ResourceLifetimeState,
    ResourceOwnershipState, ResourcePermittedUsages, ResourceProvenance,
    ResourceTransitionSchedule, ResourceUsageTransition, SubresourceRange, TimelineSyncProtocol,
    TransitionExecutionReport,
};

#[test]
fn image_formats_byte_sizes_and_subresources() {
    let fmt_rgba8 = ImageFormat::Rgba8Unorm;
    assert_eq!(fmt_rgba8.bytes_per_pixel(), 4);
    assert!(!fmt_rgba8.is_depth_stencil());

    let fmt_rgba16f = ImageFormat::Rgba16Float;
    assert_eq!(fmt_rgba16f.bytes_per_pixel(), 8);

    let fmt_depth = ImageFormat::Depth32Float;
    assert_eq!(fmt_depth.bytes_per_pixel(), 4);
    assert!(fmt_depth.is_depth_stencil());

    let dims = ImageDimensions::d2(1920, 1080);
    assert_eq!(dims.unpadded_layer_bytes(fmt_rgba8), 1920 * 1080 * 4);

    let subresource = SubresourceRange::full(&dims);
    assert_eq!(subresource.base_mip_level, 0);
    assert_eq!(subresource.mip_level_count, 1);
    assert_eq!(subresource.array_layer_count, 1);
}

#[test]
fn admitted_resource_record_pitch_alignment_and_zero_copy() {
    let owner = ResidentOwner::new().expect("mint resident owner");
    let record = AdmittedResourceRecord::new_2d_owned(
        101,
        1,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        owner,
    );

    assert_eq!(record.resource_id, 101);
    assert_eq!(record.generation, 1);
    assert!(record.is_zero_copy);
    assert!(record.is_valid);
    // Min pitch = 1920 * 4 = 7680 bytes. 7680 is a multiple of 256, so aligned pitch = 7680
    assert_eq!(record.row_pitch_bytes, 7680);
    assert_eq!(record.dimensions.width, 1920);
    assert_eq!(record.dimensions.height, 1080);
    assert_eq!(
        record.ownership,
        ResourceOwnershipState::Exclusive(owner.get())
    );
    assert_eq!(record.lifetime, ResourceLifetimeState::Retained);
    assert_eq!(record.alias_set, ResourceAliasSet::None);
}

#[test]
fn resource_generation_advance_and_stale_access_rejection() {
    let owner = ResidentOwner::new().expect("mint resident owner");
    let mut record = AdmittedResourceRecord::new_2d_owned(
        202,
        1,
        ImageFormat::Rgba16Float,
        ColorInterpretation::LinearRgb,
        512,
        512,
        ResourcePermittedUsages::STORAGE,
        owner,
    );

    assert_eq!(record.generation, 1);

    // Advance generation from 1 -> 2
    let gen2 = record
        .advance_generation(1)
        .expect("advance from expected gen 1");
    assert_eq!(gen2, 2);
    assert_eq!(record.generation, 2);

    // Stale frame access expecting generation 1 must be rejected
    let stale_err = record
        .advance_generation(1)
        .expect_err("stale generation 1 must be rejected");

    assert_eq!(
        stale_err,
        ResourceAbiError::GenerationMismatch {
            resource_id: 202,
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn resource_invalidation_on_device_loss() {
    let owner = ResidentOwner::new().expect("mint resident owner");
    let mut record = AdmittedResourceRecord::new_2d_owned(
        303,
        1,
        ImageFormat::R32Float,
        ColorInterpretation::Passthrough,
        256,
        256,
        ResourcePermittedUsages::STORAGE,
        owner,
    );

    record.invalidate_on_device_loss();
    assert!(!record.is_valid);

    // Any operation on invalidated resource must fail
    let err = record
        .advance_generation(1)
        .expect_err("invalidated resource must fail");

    assert_eq!(
        err,
        ResourceAbiError::ResourceInvalidated { resource_id: 303 }
    );
}

#[test]
fn zero_copy_negotiation_and_usage_transitions() {
    let owner = ResidentOwner::new().expect("mint resident owner");
    let mut record = AdmittedResourceRecord::new_2d_owned(
        404,
        1,
        ImageFormat::Bgra8Unorm,
        ColorInterpretation::DisplayP3,
        3840,
        2160,
        ResourcePermittedUsages::COLOR_ATTACHMENT
            .union(ResourcePermittedUsages::PRESENTATION)
            .union(ResourcePermittedUsages::EXTERNAL_IMPORT),
        owner,
    );

    // Negotiate zero-copy DMA-BUF import
    record
        .negotiate_zero_copy_import(ExternalMemoryKind::DmaBuf, 0x1234)
        .expect("negotiate zero-copy dmabuf");

    match record.provenance {
        ResourceProvenance::ExternalImport {
            memory_kind,
            handle_tag,
            ..
        } => {
            assert_eq!(memory_kind, ExternalMemoryKind::DmaBuf);
            assert_eq!(handle_tag, 0x1234);
        }
        _ => panic!("expected external import provenance"),
    }
    assert!(record.is_zero_copy);

    // Transition from color attachment output to presentation layout
    let transition =
        ResourceUsageTransition::to_present(ResourceLayoutState::ColorAttachmentOptimal);
    assert_eq!(
        transition.from_layout,
        ResourceLayoutState::ColorAttachmentOptimal
    );
    assert_eq!(transition.to_layout, ResourceLayoutState::PresentSrc);
    assert!(transition.requires_barrier);
}

#[test]
fn pre_allocation_rejection_of_unsupported_import_combinations() {
    // 1. Unaligned pitch must be rejected before allocation
    let unaligned_err = authenticate_external_import(
        501,
        1,
        ImageFormat::Rgba8Unorm,
        ExternalMemoryKind::DmaBuf,
        1920,
        1920 * 4 + 13, // Not 256-byte aligned
    )
    .expect_err("unaligned pitch must be rejected before allocation");

    assert_eq!(
        unaligned_err,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 501,
            provided_pitch: 7693,
            required_pitch: 7680,
        }
    );

    // 2. Unsupported format with DMA-BUF (e.g. depth stencil) must be rejected before allocation naming combination
    let unsupported_dmabuf_err = authenticate_external_import(
        502,
        1,
        ImageFormat::Depth32Float,
        ExternalMemoryKind::DmaBuf,
        1024,
        1024 * 4,
    )
    .expect_err("depth stencil on dmabuf must be rejected");

    assert_eq!(
        unsupported_dmabuf_err,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 502,
            format: ImageFormat::Depth32Float,
            memory_kind: ExternalMemoryKind::DmaBuf,
        }
    );

    // 3. Valid DMA-BUF import passes pre-allocation check
    authenticate_external_import(
        503,
        1,
        ImageFormat::Rgba8Unorm,
        ExternalMemoryKind::DmaBuf,
        1920,
        1920 * 4,
    )
    .expect("valid dmabuf import must succeed");
}

#[test]
fn external_resource_registry_and_device_loss_view_invalidation() {
    let registry = ExternalResourceRegistry::new();

    let owner = ResidentOwner::new().expect("mint resident owner");
    let record1 = AdmittedResourceRecord::new_2d_owned(
        601,
        10, // device 10
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        ResourcePermittedUsages::SAMPLED,
        owner,
    );
    let record2 = AdmittedResourceRecord::new_2d_owned(
        602,
        20, // device 20 (other device)
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        ResourcePermittedUsages::SAMPLED,
        owner,
    );

    registry.admit_resource(record1).expect("admit resource 1");
    registry.admit_resource(record2).expect("admit resource 2");

    // Register dependent views and artifacts on resource 601
    registry
        .register_dependent_view(601, 7001)
        .expect("register view 7001");
    registry
        .register_dependent_view(601, 7002)
        .expect("register view 7002");
    registry
        .register_dependent_artifact(601, 8001)
        .expect("register artifact 8001");

    // Invalidate device 10
    let report = registry.invalidate_on_device_loss(10);
    assert_eq!(report.device_id, 10);
    assert_eq!(report.invalidated_resources, vec![601]);
    assert_eq!(report.invalidated_views, vec![7001, 7002]);
    assert_eq!(report.invalidated_artifacts, vec![8001]);

    // Resource on device 10 is now invalidated
    let res1 = registry
        .get_resource(601)
        .expect("resource exists in registry");
    assert!(!res1.is_valid);

    // Resource on device 20 remains valid
    let res2 = registry
        .get_resource(602)
        .expect("resource exists in registry");
    assert!(res2.is_valid);
}

#[test]
fn exact_schedule_execution_proves_zero_copies_and_no_device_wide_waits() {
    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(901, ResourceUsageTransition::storage_to_color_attachment());
    schedule.add_transition(
        901,
        ResourceUsageTransition::to_present(ResourceLayoutState::ColorAttachmentOptimal),
    );
    schedule.add_wait(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 11,
        wait_value: 100,
        signal_value: 101,
    });
    schedule.add_signal(TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 12,
        wait_value: 200,
        signal_value: 201,
    });

    let report = TransitionExecutionReport::execute_exact(&schedule);
    assert_eq!(report.transitions_executed, 2);
    assert_eq!(report.barriers_emitted, 2);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    // CRITICAL ACCEPTANCE: Proven absence of unrecorded copies and device-wide waits
    assert_eq!(
        report.copy_count, 0,
        "schedule execution must perform zero copies"
    );
    assert_eq!(
        report.device_wide_waits, 0,
        "schedule execution must perform zero device-wide waits"
    );
}

#[test]
fn variant_space_enumeration_runtime_mutation_gate() {
    // Enumerate every image format and ensure an admission decision exists
    for &format in all_image_formats() {
        assert!(format.bytes_per_pixel() > 0);
        assert!(!format.format_class().to_string().is_empty());
    }

    // Enumerate every format class
    assert_eq!(all_format_classes().len(), 7);

    // Enumerate every synchronization protocol
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

    // Enumerate every external memory kind
    assert_eq!(all_external_memory_kinds().len(), 6);
    assert_eq!(all_usage_flags().len(), 10);
    assert_eq!(all_layout_states().len(), 8);
    assert_eq!(all_lifetime_state_kinds().len(), 5);
    assert_eq!(all_provenance_kinds().len(), 3);
    assert_eq!(all_alias_set_kinds().len(), 3);
}

trait ToStringHelper {
    fn to_string(&self) -> String;
}

impl ToStringHelper for FormatClass {
    fn to_string(&self) -> String {
        format!("{self:?}")
    }
}
