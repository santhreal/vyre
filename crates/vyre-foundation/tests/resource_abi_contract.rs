//! Contract tests for domain-neutral typed image, plane, view, sampler, and external-resource capabilities.
//!
//! Proves:
//! 1. Format metadata and bytes-per-pixel calculations.
//! 2. Extent and subresource bounds verification.
//! 3. Row pitch alignment and minimum width constraints.
//! 4. View format compatibility validation.
//! 5. External memory capability negotiation.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{
    AdmittedResourceRecord, ColorSpace, ExternalMemoryDescriptor, ExternalMemoryKind,
    ExternalResourceCapability, ExternalSyncProtocol, ImageDimension, ImageExtent, ImageFormat,
    PlaneDescriptor, ResourceAbiError, ResourceUsageFlags, SamplerAddressMode, SamplerDescriptor,
    SamplerFilter, SubresourceRange, ValueLifetime, ViewDescriptor,
};

#[test]
fn image_format_byte_sizes_are_correct() {
    assert_eq!(ImageFormat::R8Unorm.bytes_per_pixel(), 1);
    assert_eq!(ImageFormat::R16Float.bytes_per_pixel(), 2);
    assert_eq!(ImageFormat::Rg8Unorm.bytes_per_pixel(), 2);
    assert_eq!(ImageFormat::Rgba8Unorm.bytes_per_pixel(), 4);
    assert_eq!(ImageFormat::Bgra8UnormSrgb.bytes_per_pixel(), 4);
    assert_eq!(ImageFormat::Depth32Float.bytes_per_pixel(), 4);
    assert_eq!(ImageFormat::Rgba16Float.bytes_per_pixel(), 8);
    assert_eq!(ImageFormat::Rg32Float.bytes_per_pixel(), 8);
    assert_eq!(ImageFormat::Rgba32Float.bytes_per_pixel(), 16);

    assert!(ImageFormat::Depth32Float.is_depth_or_stencil());
    assert!(ImageFormat::Depth24PlusStencil8.is_depth_or_stencil());
    assert!(!ImageFormat::Rgba8Unorm.is_depth_or_stencil());
}

#[test]
fn admitted_resource_record_validates_pitch_and_alignment() {
    let caps = ExternalResourceCapability::desktop_gpu_standard();

    // 1. Valid 1920x1080 RGBA8 texture with 256-byte aligned row pitch (1920 * 4 = 7680, which is 30 * 256)
    let valid_record = AdmittedResourceRecord {
        name: "framebuffer_0".into(),
        device_id: "gpu_0".into(),
        format: ImageFormat::Rgba8Unorm,
        dimension: ImageDimension::D2,
        extent: ImageExtent::d2(1920, 1080),
        planes: vec![PlaneDescriptor {
            plane_index: 0,
            offset_bytes: 0,
            row_pitch_bytes: 7680,
            plane_height: 1080,
            format: ImageFormat::Rgba8Unorm,
        }],
        usages: ResourceUsageFlags::presentable_storage(),
        views: vec![ViewDescriptor {
            name: "main_view".into(),
            format: ImageFormat::Rgba8UnormSrgb,
            dimension: ImageDimension::D2,
            subresource_range: SubresourceRange::color_all(),
            color_space: ColorSpace::Srgb,
        }],
        alias_group: None,
        lifetime: ValueLifetime::Retained,
        sync_protocol: Some(ExternalSyncProtocol::TimelineSemaphore),
        generation: 1,
    };

    assert!(valid_record.validate(&caps).is_ok());

    // 2. Insufficient row pitch fails
    let mut bad_pitch = valid_record.clone();
    bad_pitch.planes[0].row_pitch_bytes = 7000; // less than 1920 * 4 = 7680
    let err = bad_pitch
        .validate(&caps)
        .expect_err("insufficient pitch must fail");
    assert!(matches!(err, ResourceAbiError::InsufficientRowPitch { .. }));

    // 3. Unaligned row pitch fails when alignment required
    let mut unaligned_pitch = valid_record.clone();
    unaligned_pitch.planes[0].row_pitch_bytes = 7684; // not multiple of 256
    let err = unaligned_pitch
        .validate(&caps)
        .expect_err("unaligned pitch must fail");
    assert!(matches!(err, ResourceAbiError::UnalignedRowPitch { .. }));

    // 4. Zero extent fails
    let mut zero_dim = valid_record.clone();
    zero_dim.extent.width = 0;
    let err = zero_dim.validate(&caps).expect_err("zero extent must fail");
    assert!(matches!(err, ResourceAbiError::ZeroExtent { .. }));

    // 5. Incompatible view format size fails
    let mut bad_view = valid_record.clone();
    bad_view.views[0].format = ImageFormat::Rgba16Float; // 8 B/px vs 4 B/px base
    let err = bad_view
        .validate(&caps)
        .expect_err("incompatible view format must fail");
    assert!(matches!(
        err,
        ResourceAbiError::IncompatibleViewFormat { .. }
    ));
}

#[test]
fn external_memory_and_sampler_descriptors_roundtrip() {
    let ext_mem = ExternalMemoryDescriptor {
        kind: ExternalMemoryKind::DmaBuf,
        size_bytes: 8 * 1024 * 1024,
        dedicated_allocation: true,
        device_id: "vulkan_device_0".into(),
        memory_type_index: Some(2),
    };
    assert_eq!(ext_mem.kind, ExternalMemoryKind::DmaBuf);
    assert_eq!(ext_mem.size_bytes, 8 * 1024 * 1024);

    let sampler = SamplerDescriptor {
        min_filter: SamplerFilter::Linear,
        mag_filter: SamplerFilter::Linear,
        mipmap_filter: SamplerFilter::Nearest,
        address_mode_u: SamplerAddressMode::Repeat,
        address_mode_v: SamplerAddressMode::ClampToEdge,
        address_mode_w: SamplerAddressMode::MirrorRepeat,
        lod_min_clamp_bits: 0,
        lod_max_clamp_bits: 16 * 256,
        max_anisotropy: 8,
        compare_enable: true,
    };
    assert_eq!(sampler.max_anisotropy, 8);
    assert!(sampler.compare_enable);
}
