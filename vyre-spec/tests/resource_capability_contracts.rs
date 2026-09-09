//! Tests for domain-neutral resource, image, view, sampler, external memory,
//! and timeline synchronization capabilities (Row 111).
//!
//! Acceptance criteria:
//! 1. Every format class, usage flag, sync protocol, and external memory kind is enumerated.
//! 2. Admitted resource records preserve all 10 specification elements.
//! 3. Deterministic generation progression and failure-atomic device loss invalidation.
//! 4. Variant enumeration completeness covers the entire capability space.

use vyre_spec::{
    all_address_modes, all_alias_set_kinds, all_border_colors, all_color_interpretations,
    all_compare_functions, all_external_event_kinds, all_external_memory_kinds, all_filter_modes,
    all_format_classes, all_image_formats, all_image_view_kinds, all_layout_states,
    all_lifetime_state_kinds, all_mipmap_filter_modes, all_plane_kinds, all_provenance_kinds,
    all_swizzle_components, all_sync_protocols, all_usage_flags, AddressMode,
    AdmittedResourceRecord, BorderColor, ColorInterpretation, CompareFunction, ComponentSwizzle,
    ExternalEventKind, ExternalMemoryCapability, ExternalMemoryKind, FilterMode, FormatClass,
    ImageDimensions, ImageFormat, ImagePlane, ImageViewDescriptor, ImageViewKind, MipmapFilterMode,
    PlaneKind, ResourceAliasSet, ResourceLayoutState, ResourceLifetimeState,
    ResourceOwnershipState, ResourcePermittedUsages, ResourceProvenance, ResourceUsageTransition,
    SamplerDescriptor, SubresourceRange, SwizzleComponent, TimelineSyncProtocol,
};

#[test]
fn image_formats_classification_and_channel_properties() {
    for &format in all_image_formats() {
        assert!(
            format.bytes_per_pixel() >= 1,
            "format {format:?} must have positive byte width"
        );
        assert!(
            format.channel_count() >= 1 && format.channel_count() <= 4,
            "format {format:?} invalid channel count"
        );
        let class = format.format_class();
        match class {
            FormatClass::DepthStencil => assert!(format.is_depth_stencil()),
            FormatClass::PlanarVideo => assert!(format.is_planar_video()),
            FormatClass::Unorm
            | FormatClass::Srgb
            | FormatClass::Float
            | FormatClass::Uint
            | FormatClass::Sint => {
                assert!(!format.is_depth_stencil());
            }
        }
    }
}

#[test]
fn all_format_classes_are_represented() {
    let mut class_covered = [false; 7];
    for &format in all_image_formats() {
        match format.format_class() {
            FormatClass::Unorm => class_covered[0] = true,
            FormatClass::Srgb => class_covered[1] = true,
            FormatClass::Float => class_covered[2] = true,
            FormatClass::Uint => class_covered[3] = true,
            FormatClass::Sint => class_covered[4] = true,
            FormatClass::DepthStencil => class_covered[5] = true,
            FormatClass::PlanarVideo => class_covered[6] = true,
        }
    }
    assert!(
        class_covered.iter().all(|&c| c),
        "every format class must have at least one format variant"
    );
    assert_eq!(all_format_classes().len(), 7);
}

#[test]
fn image_dimensions_and_subresource_ranges() {
    let dims1d = ImageDimensions::d1(1024);
    assert!(dims1d.is_valid());
    assert_eq!(dims1d.height, 1);
    assert_eq!(dims1d.depth, 1);

    let dims2d = ImageDimensions::d2(1920, 1080);
    assert!(dims2d.is_valid());
    assert_eq!(
        dims2d.unpadded_layer_bytes(ImageFormat::Rgba8Unorm),
        1920 * 1080 * 4
    );

    let dims3d = ImageDimensions::d3(64, 64, 64);
    assert!(dims3d.is_valid());
    assert_eq!(
        dims3d.unpadded_layer_bytes(ImageFormat::R32Float),
        64 * 64 * 64 * 4
    );

    let full_range = SubresourceRange::full(&dims2d);
    assert_eq!(full_range.base_mip_level, 0);
    assert_eq!(full_range.mip_level_count, 1);
    assert_eq!(full_range.base_array_layer, 0);
    assert_eq!(full_range.array_layer_count, 1);
    assert_eq!(full_range.aspect_mask, SubresourceRange::ASPECT_COLOR);
}

#[test]
fn image_view_and_swizzle_descriptors() {
    let sub = SubresourceRange {
        base_mip_level: 0,
        mip_level_count: 1,
        base_array_layer: 0,
        array_layer_count: 6,
        aspect_mask: SubresourceRange::ASPECT_COLOR,
    };
    let view = ImageViewDescriptor {
        format: ImageFormat::Rgba8Srgb,
        view_kind: ImageViewKind::Cube,
        subresource_range: sub,
        swizzle: ComponentSwizzle::IDENTITY,
    };
    assert_eq!(view.view_kind, ImageViewKind::Cube);
    assert_eq!(view.swizzle.r, SwizzleComponent::Identity);
    assert_eq!(view.format, ImageFormat::Rgba8Srgb);
}

#[test]
fn image_planes_and_plane_kinds() {
    let plane_y = ImagePlane {
        plane_index: 0,
        plane_kind: PlaneKind::Y,
        offset_bytes: 0,
        row_pitch_bytes: 1920,
        plane_height: 1080,
        format: ImageFormat::R8Unorm,
    };
    assert_eq!(plane_y.plane_kind, PlaneKind::Y);
    assert_eq!(plane_y.row_pitch_bytes, 1920);

    let plane_uv = ImagePlane {
        plane_index: 1,
        plane_kind: PlaneKind::Uv,
        offset_bytes: 1920 * 1080,
        row_pitch_bytes: 1920,
        plane_height: 540,
        format: ImageFormat::Rg8Unorm,
    };
    assert_eq!(plane_uv.plane_kind, PlaneKind::Uv);
    assert_eq!(plane_uv.plane_height, 540);
}

#[test]
fn resource_usage_transitions_and_compare_samplers() {
    let transition = ResourceUsageTransition::storage_to_color_attachment();
    assert_eq!(transition.from_layout, ResourceLayoutState::General);
    assert_eq!(
        transition.to_layout,
        ResourceLayoutState::ColorAttachmentOptimal
    );
    assert!(transition.requires_barrier);

    let to_sampled =
        ResourceUsageTransition::to_sampled(ResourceLayoutState::ColorAttachmentOptimal);
    assert_eq!(to_sampled.to_layout, ResourceLayoutState::ShaderReadOnly);

    let to_present =
        ResourceUsageTransition::to_present(ResourceLayoutState::ColorAttachmentOptimal);
    assert_eq!(to_present.to_layout, ResourceLayoutState::PresentSrc);

    let mut shadow_sampler = SamplerDescriptor::linear_clamp();
    shadow_sampler.compare = Some(CompareFunction::LessEqual);
    assert_eq!(shadow_sampler.compare, Some(CompareFunction::LessEqual));
}
#[test]
fn sampler_descriptors_and_defaults() {
    let linear = SamplerDescriptor::linear_clamp();
    assert_eq!(linear.filter_min, FilterMode::Linear);
    assert_eq!(linear.filter_mag, FilterMode::Linear);
    assert_eq!(linear.mipmap_filter, MipmapFilterMode::Linear);
    assert_eq!(linear.address_mode_u, AddressMode::ClampToEdge);
    assert_eq!(linear.border_color, BorderColor::TransparentBlack);

    let nearest = SamplerDescriptor::nearest_clamp();
    assert_eq!(nearest.filter_min, FilterMode::Nearest);
    assert_eq!(nearest.filter_mag, FilterMode::Nearest);
    assert_eq!(nearest.mipmap_filter, MipmapFilterMode::Nearest);
}

#[test]
fn external_memory_and_event_capabilities() {
    let dma_buf_cap = ExternalMemoryCapability {
        kind: ExternalMemoryKind::DmaBuf,
        can_import: true,
        can_export: true,
        requires_dedicated_allocation: false,
        alignment_bytes: 4096,
    };
    assert_eq!(dma_buf_cap.kind, ExternalMemoryKind::DmaBuf);
    assert!(dma_buf_cap.can_import);
    assert_eq!(dma_buf_cap.alignment_bytes, 4096);

    let timeline_proto = TimelineSyncProtocol::TimelineSemaphore {
        timeline_id: 42,
        wait_value: 10,
        signal_value: 11,
    };
    assert_eq!(
        timeline_proto.event_kind(),
        ExternalEventKind::TimelineSemaphore
    );
    assert!(timeline_proto.is_timeline());

    let fence_proto = TimelineSyncProtocol::Fence {
        fence_id: 7,
        is_signaled: false,
    };
    assert_eq!(fence_proto.event_kind(), ExternalEventKind::BinaryFence);
    assert!(!fence_proto.is_timeline());
}

#[test]
fn resource_permitted_usages_algebra() {
    let usage = ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::COLOR_ATTACHMENT);
    assert!(usage.contains(ResourcePermittedUsages::SAMPLED));
    assert!(usage.contains(ResourcePermittedUsages::COLOR_ATTACHMENT));
    assert!(!usage.contains(ResourcePermittedUsages::TRANSFER_DST));

    let isect = usage.intersection(ResourcePermittedUsages::SAMPLED);
    assert_eq!(isect, ResourcePermittedUsages::SAMPLED);

    let storage = ResourcePermittedUsages::STORAGE;
    assert!(storage.contains(ResourcePermittedUsages::STORAGE_READ));
    assert!(storage.contains(ResourcePermittedUsages::STORAGE_WRITE));
}

#[test]
fn admitted_resource_record_encodes_all_ten_row111_elements() {
    let record = AdmittedResourceRecord::new_2d(
        1001,
        1,
        ImageFormat::Rgba8Unorm,
        ColorInterpretation::Srgb,
        1920,
        1080,
        ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        42,
    );

    // 1. Device identity
    assert_eq!(record.device_id, 1);
    // 2. Format and color semantics
    assert_eq!(record.format, ImageFormat::Rgba8Unorm);
    assert_eq!(record.color, ColorInterpretation::Srgb);
    // 3. Dimensions and pitch
    assert_eq!(record.dimensions.width, 1920);
    assert_eq!(record.dimensions.height, 1080);
    assert_eq!(record.row_pitch_bytes, 1920 * 4); // 7680 is 256-byte aligned
                                                  // 4. Subresource range
    assert_eq!(record.subresource.mip_level_count, 1);
    assert_eq!(record.subresource.array_layer_count, 1);
    // 5. Permitted usages
    assert!(record
        .permitted_usages
        .contains(ResourcePermittedUsages::SAMPLED));
    assert!(record
        .permitted_usages
        .contains(ResourcePermittedUsages::COLOR_ATTACHMENT));
    // 6. Ownership state
    assert_eq!(record.ownership, ResourceOwnershipState::Exclusive(42));
    // 7. Alias set
    assert_eq!(record.alias_set, ResourceAliasSet::None);
    // 8. Lifetime
    assert_eq!(record.lifetime, ResourceLifetimeState::Retained);
    // 9. Synchronization protocol
    assert_eq!(record.sync_protocol, TimelineSyncProtocol::ImplicitQueue);
    // 10. Provenance
    match record.provenance {
        ResourceProvenance::InternalAllocation {
            allocator_tag,
            byte_size,
        } => {
            assert_eq!(allocator_tag, 1);
            assert_eq!(byte_size, 7680 * 1080);
        }
        _ => panic!("expected internal allocation provenance"),
    }
}

#[test]
fn admitted_external_import_record_properties() {
    let record = AdmittedResourceRecord::new_external_import_2d(
        2002,
        2,
        ImageFormat::Rgba16Float,
        ColorInterpretation::LinearRgb,
        3840,
        2160,
        3840 * 8,
        ResourcePermittedUsages::STORAGE_READ.union(ResourcePermittedUsages::EXTERNAL_EXPORT),
        ExternalMemoryKind::DmaBuf,
        0xDEAD_BEEF,
        TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 99,
            wait_value: 5,
            signal_value: 6,
        },
    );

    assert_eq!(record.resource_id, 2002);
    assert_eq!(record.device_id, 2);
    assert_eq!(record.format, ImageFormat::Rgba16Float);
    assert_eq!(record.color, ColorInterpretation::LinearRgb);
    assert!(record.is_zero_copy);
    assert!(record.is_valid);
    assert_eq!(record.lifetime, ResourceLifetimeState::ExternalPinned);
    assert_eq!(record.ownership, ResourceOwnershipState::ExternalHost);

    match record.provenance {
        ResourceProvenance::ExternalImport {
            memory_kind,
            exportable,
            handle_tag,
        } => {
            assert_eq!(memory_kind, ExternalMemoryKind::DmaBuf);
            assert!(exportable);
            assert_eq!(handle_tag, 0xDEAD_BEEF);
        }
        _ => panic!("expected external import provenance"),
    }
}

#[test]
fn resource_generation_advance_and_invalidation() {
    let mut record = AdmittedResourceRecord::new_2d(
        3003,
        1,
        ImageFormat::R32Float,
        ColorInterpretation::Passthrough,
        512,
        512,
        ResourcePermittedUsages::STORAGE,
        1,
    );

    assert_eq!(record.generation, 1);
    let next_gen = record.advance_generation(1).expect("advance generation");
    assert_eq!(next_gen, 2);
    assert_eq!(record.generation, 2);

    // Mismatched expected generation must fail
    let err = record.advance_generation(1).expect_err("stale generation");
    assert_eq!(
        err,
        vyre_spec::ResourceAbiError::GenerationMismatch {
            resource_id: 3003,
            expected: 1,
            actual: 2,
        }
    );

    // Invalidation on device loss
    record.invalidate_on_device_loss();
    assert!(!record.is_valid);
    let err_invalid = record
        .advance_generation(2)
        .expect_err("invalidated resource");
    assert_eq!(
        err_invalid,
        vyre_spec::ResourceAbiError::ResourceInvalidated { resource_id: 3003 }
    );
}

#[test]
fn variant_enumeration_completeness() {
    assert_eq!(all_image_formats().len(), 38);
    assert_eq!(all_format_classes().len(), 7);
    assert_eq!(all_color_interpretations().len(), 7);
    assert_eq!(all_plane_kinds().len(), 7);
    assert_eq!(all_image_view_kinds().len(), 6);
    assert_eq!(all_swizzle_components().len(), 7);
    assert_eq!(all_filter_modes().len(), 2);
    assert_eq!(all_mipmap_filter_modes().len(), 2);
    assert_eq!(all_address_modes().len(), 4);
    assert_eq!(all_compare_functions().len(), 8);
    assert_eq!(all_border_colors().len(), 3);
    assert_eq!(all_external_memory_kinds().len(), 6);
    assert_eq!(all_external_event_kinds().len(), 5);
    assert_eq!(all_sync_protocols().len(), 5);
    assert_eq!(all_usage_flags().len(), 10);
    assert_eq!(all_layout_states().len(), 8);
    assert_eq!(all_lifetime_state_kinds().len(), 5);
    assert_eq!(all_provenance_kinds().len(), 3);
    assert_eq!(all_alias_set_kinds().len(), 3);
}
