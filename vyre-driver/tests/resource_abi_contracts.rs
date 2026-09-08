//! Tests for domain-neutral semantic resource ABI, zero-copy import/export,
//! layout transitions, and timeline synchronization (Row 111).
//!
//! Acceptance criteria:
//! 1. Domain-neutral typed image, plane, subresource, and timeline capabilities.
//! 2. Admitted resource records with exact dimensions, row pitch alignment, format, and ownership.
//! 3. Deterministic generation progression preventing stale-frame access.
//! 4. Failure-atomic invalidation on device loss.
//! 5. Zero-copy capability negotiation.
use vyre_driver::{
    AdmittedResourceRecord, ColorInterpretation, ExternalMemoryCapability, ImageDimensions,
    ImageFormat, ResidentOwner, ResourceAbiError, ResourceLayoutState, ResourcePermittedUsages,
    ResourceUsageTransition, SubresourceRange,
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
    let record = AdmittedResourceRecord::new_2d(
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
}

#[test]
fn resource_generation_advance_and_stale_access_rejection() {
    let owner = ResidentOwner::new().expect("mint resident owner");
    let mut record = AdmittedResourceRecord::new_2d(
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
    let mut record = AdmittedResourceRecord::new_2d(
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
    let mut record = AdmittedResourceRecord::new_2d(
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
        .negotiate_zero_copy_import(ExternalMemoryCapability::DmaBuf)
        .expect("negotiate zero-copy dmabuf");

    assert_eq!(
        record.external_memory,
        Some(ExternalMemoryCapability::DmaBuf)
    );
    assert!(record.is_zero_copy);

    // Transition from color attachment output to presentation layout
    let transition = ResourceUsageTransition::to_present(ResourceLayoutState::ColorAttachmentOptimal);
    assert_eq!(transition.from_layout, ResourceLayoutState::ColorAttachmentOptimal);
    assert_eq!(transition.to_layout, ResourceLayoutState::PresentSrc);
    assert!(transition.requires_barrier);
}
