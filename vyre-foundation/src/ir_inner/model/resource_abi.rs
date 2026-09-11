//! Domain-neutral typed image, plane, view, sampler, and external-resource capabilities.
//!
//! Interactive graphics and multi-device pipelines require buffers and images to
//! cross compute, rendering, and presentation boundaries without host copies or
//! implicit global synchronization barriers.
//!
//! This module defines domain-neutral types for:
//! 1. Logical image formats, color spaces, dimensions, and extents.
//! 2. Multi-planar memory layouts with row pitch and plane alignment contracts.
//! 3. Subresource ranges and view descriptors.
//! 4. Sampler configuration and address modes.
//! 5. External memory import/export capabilities and timeline synchronization protocols.
//! 6. Admitted resource records with provenance, alias group, and lifetime contracts.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::program_graph::ValueLifetime;

/// Canonical domain-neutral image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    /// 8-bit unsigned normalized single channel.
    R8Unorm,
    /// 8-bit signed normalized single channel.
    R8Snorm,
    /// 8-bit unsigned integer single channel.
    R8Uint,
    /// 8-bit signed integer single channel.
    R8Sint,
    /// 16-bit unsigned integer single channel.
    R16Uint,
    /// 16-bit signed integer single channel.
    R16Sint,
    /// 16-bit float single channel.
    R16Float,
    /// 8-bit unsigned normalized dual channel.
    Rg8Unorm,
    /// 8-bit signed normalized dual channel.
    Rg8Snorm,
    /// 8-bit unsigned integer dual channel.
    Rg8Uint,
    /// 8-bit signed integer dual channel.
    Rg8Sint,
    /// 32-bit unsigned integer single channel.
    R32Uint,
    /// 32-bit signed integer single channel.
    R32Sint,
    /// 32-bit float single channel.
    R32Float,
    /// 16-bit unsigned integer dual channel.
    Rg16Uint,
    /// 16-bit signed integer dual channel.
    Rg16Sint,
    /// 16-bit float dual channel.
    Rg16Float,
    /// 8-bit unsigned normalized four channel.
    Rgba8Unorm,
    /// 8-bit unsigned normalized four channel sRGB.
    Rgba8UnormSrgb,
    /// 8-bit signed normalized four channel.
    Rgba8Snorm,
    /// 8-bit unsigned integer four channel.
    Rgba8Uint,
    /// 8-bit signed integer four channel.
    Rgba8Sint,
    /// 8-bit unsigned normalized four channel BGRA order.
    Bgra8Unorm,
    /// 8-bit unsigned normalized four channel BGRA order sRGB.
    Bgra8UnormSrgb,
    /// 32-bit unsigned integer dual channel.
    Rg32Uint,
    /// 32-bit signed integer dual channel.
    Rg32Sint,
    /// 32-bit float dual channel.
    Rg32Float,
    /// 16-bit unsigned integer four channel.
    Rgba16Uint,
    /// 16-bit signed integer four channel.
    Rgba16Sint,
    /// 16-bit float four channel.
    Rgba16Float,
    /// 32-bit unsigned integer four channel.
    Rgba32Uint,
    /// 32-bit signed integer four channel.
    Rgba32Sint,
    /// 32-bit float four channel.
    Rgba32Float,
    /// 32-bit float depth channel.
    Depth32Float,
    /// 24-bit depth and 8-bit stencil combined format.
    Depth24PlusStencil8,
}

impl ImageFormat {
    /// Return the exact number of bytes per pixel/block for this format.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => 1,
            Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint => 2,
            Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Depth32Float
            | Self::Depth24PlusStencil8 => 4,
            Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float => 8,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 16,
        }
    }

    /// True if the format represents depth or stencil data.
    #[must_use]
    pub const fn is_depth_or_stencil(self) -> bool {
        matches!(self, Self::Depth32Float | Self::Depth24PlusStencil8)
    }
}

/// Color interpretation and transfer function standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    /// Standard sRGB non-linear encoding.
    Srgb,
    /// Linear sRGB / Rec. 709 primaries with linear transfer.
    LinearSrgb,
    /// Apple Display P3 wide gamut.
    DisplayP3,
    /// ITU-R BT.709 HDTV standard.
    Bt709,
    /// ITU-R BT.2020 UHDTV wide gamut.
    Bt2020,
    /// Raw uninterpreted sensor or numerical array data.
    Raw,
}

/// Dimensionality of a texture or image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDimension {
    /// 1-dimensional texture.
    D1,
    /// 2-dimensional texture.
    D2,
    /// 2-dimensional array texture.
    D2Array,
    /// 3-dimensional volume texture.
    D3,
    /// Cube map texture (6 faces).
    Cube,
    /// Cube map array texture.
    CubeArray,
}

/// 3D spatial and layer extent for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageExtent {
    /// Width in texels.
    pub width: u32,
    /// Height in texels (1 for 1D).
    pub height: u32,
    /// Depth in texels or number of array layers.
    pub depth_or_array_layers: u32,
}

impl ImageExtent {
    /// Create a 2D extent.
    #[must_use]
    pub const fn d2(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }

    /// Create a 3D extent.
    #[must_use]
    pub const fn d3(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth_or_array_layers: depth,
        }
    }

    /// Total texel count across all dimensions.
    #[must_use]
    pub const fn total_texels(&self) -> u64 {
        (self.width as u64)
            .saturating_mul(self.height as u64)
            .saturating_mul(self.depth_or_array_layers as u64)
    }
}

/// Target aspect for multi-aspect formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageAspect {
    /// Color planes only.
    Color,
    /// Depth component only.
    Depth,
    /// Stencil component only.
    Stencil,
    /// Both depth and stencil components.
    DepthStencil,
}

/// Contiguous subresource range within an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubresourceRange {
    /// Target image aspect.
    pub aspect: ImageAspect,
    /// First mip level included.
    pub base_mip_level: u32,
    /// Number of mip levels (0 means all remaining).
    pub mip_level_count: u32,
    /// First array layer included.
    pub base_array_layer: u32,
    /// Number of array layers (0 means all remaining).
    pub array_layer_count: u32,
}

impl SubresourceRange {
    /// Full resource range for color image.
    #[must_use]
    pub const fn color_all() -> Self {
        Self {
            aspect: ImageAspect::Color,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        }
    }
}

/// Permitted usage capabilities for an admitted GPU resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceUsageFlags {
    /// May be sampled in compute or fragment shaders.
    pub sampled: bool,
    /// May be read as a storage buffer or texture.
    pub storage_read: bool,
    /// May be written as a storage buffer or texture.
    pub storage_write: bool,
    /// May be bound as a render target / color attachment.
    pub color_attachment: bool,
    /// May be bound as a depth/stencil attachment.
    pub depth_stencil_attachment: bool,
    /// May be used as a copy source.
    pub transfer_src: bool,
    /// May be used as a copy destination.
    pub transfer_dst: bool,
    /// May be handed off to presentation engine or display subsystem.
    pub present: bool,
    /// May cross process or API boundaries via external memory import/export.
    pub external_shared: bool,
}

impl ResourceUsageFlags {
    /// Read-only compute shader input texture.
    #[must_use]
    pub const fn compute_sampled() -> Self {
        Self {
            sampled: true,
            storage_read: true,
            storage_write: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transfer_src: false,
            transfer_dst: true,
            present: false,
            external_shared: false,
        }
    }

    /// Read-write compute storage texture.
    #[must_use]
    pub const fn compute_storage() -> Self {
        Self {
            sampled: true,
            storage_read: true,
            storage_write: true,
            color_attachment: false,
            depth_stencil_attachment: false,
            transfer_src: true,
            transfer_dst: true,
            present: false,
            external_shared: false,
        }
    }

    /// Presentation target surface with compute write capability.
    #[must_use]
    pub const fn presentable_storage() -> Self {
        Self {
            sampled: true,
            storage_read: true,
            storage_write: true,
            color_attachment: true,
            depth_stencil_attachment: false,
            transfer_src: true,
            transfer_dst: true,
            present: true,
            external_shared: true,
        }
    }
}

/// Layout specification for one memory plane in a multi-planar image.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlaneDescriptor {
    /// Zero-based plane index.
    pub plane_index: u32,
    /// Byte offset from the start of the underlying memory allocation.
    pub offset_bytes: u64,
    /// Row pitch in bytes, accounting for hardware alignment constraints.
    pub row_pitch_bytes: u32,
    /// Height of this plane in rows.
    pub plane_height: u32,
    /// Specific pixel format of this plane.
    pub format: ImageFormat,
}

impl PlaneDescriptor {
    /// Total bytes required by this plane.
    #[must_use]
    pub fn plane_byte_len(&self) -> u64 {
        (self.row_pitch_bytes as u64).saturating_mul(self.plane_height as u64)
    }
}

/// Logical view configuration referencing a base image resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ViewDescriptor {
    /// Human-readable or debug name for this view.
    pub name: Arc<str>,
    /// Format interpretation for this view (may be an sRGB alias of the base format).
    pub format: ImageFormat,
    /// Dimension through which this view perceives the underlying texture.
    pub dimension: ImageDimension,
    /// Subresource subset selected by this view.
    pub subresource_range: SubresourceRange,
    /// Color interpretation space.
    pub color_space: ColorSpace,
}

/// Texture filtering interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerFilter {
    /// Nearest texel sample.
    Nearest,
    /// Linear filtering between adjacent texels.
    Linear,
}

/// Addressing mode for coordinates outside [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerAddressMode {
    /// Clamps coordinate to edge texel.
    ClampToEdge,
    /// Repeats texture periodically.
    Repeat,
    /// Mirrors and repeats texture periodically.
    MirrorRepeat,
    /// Returns a fixed border color.
    ClampToBorder,
}

/// Domain-neutral sampler state descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplerDescriptor {
    /// Minification filter.
    pub min_filter: SamplerFilter,
    /// Magnification filter.
    pub mag_filter: SamplerFilter,
    /// Mipmap level filter.
    pub mipmap_filter: SamplerFilter,
    /// U coordinate addressing mode.
    pub address_mode_u: SamplerAddressMode,
    /// V coordinate addressing mode.
    pub address_mode_v: SamplerAddressMode,
    /// W coordinate addressing mode.
    pub address_mode_w: SamplerAddressMode,
    /// Minimum level of detail clamp (fixed-point * 256 for Eq/Hash).
    pub lod_min_clamp_bits: u32,
    /// Maximum level of detail clamp (fixed-point * 256 for Eq/Hash).
    pub lod_max_clamp_bits: u32,
    /// Maximum anisotropic filtering ratio.
    pub max_anisotropy: u16,
    /// Whether shadow comparison is enabled.
    pub compare_enable: bool,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            mipmap_filter: SamplerFilter::Linear,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            lod_min_clamp_bits: 0,
            lod_max_clamp_bits: 32 * 256,
            max_anisotropy: 1,
            compare_enable: false,
        }
    }
}

/// Platform mechanism for sharing memory handles with foreign processes or APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalMemoryKind {
    /// POSIX file descriptor, opaque to the importer.
    OpaqueFd,
    /// Linux DMA-BUF handle for direct display and V4L2 integration.
    DmaBuf,
    /// Windows NT handle (`HANDLE`).
    OpaqueWin32,
    /// Direct3D 11/12 shared resource handle.
    D3D11Texture,
    /// Apple Metal shared event and IOSurface handle.
    MetalSharedSurface,
    /// Pinned host memory pointer.
    HostPtr,
}

/// Cross-process external memory allocation contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalMemoryDescriptor {
    /// Memory handle mechanism.
    pub kind: ExternalMemoryKind,
    /// Total allocation size in bytes.
    pub size_bytes: u64,
    /// Whether this resource requires a dedicated driver allocation.
    pub dedicated_allocation: bool,
    /// Unique identifier of the originating device.
    pub device_id: String,
    /// Driver-specific memory type index, if specified.
    pub memory_type_index: Option<u32>,
}

/// Cross-queue or cross-process synchronization primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalSyncProtocol {
    /// Monotonically increasing timeline semaphore.
    TimelineSemaphore,
    /// Binary fence or sync object.
    BinaryFence,
    /// Metal shared event timeline counter.
    MetalSharedEvent,
    /// Windows sync event object.
    Win32Event,
}

/// Driver capability profile for external resources and zero-copy transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalResourceCapability {
    /// Device supports importing foreign external memory without host copies.
    pub zero_copy_import: bool,
    /// Device supports exporting local memory to foreign APIs without host copies.
    pub zero_copy_export: bool,
    /// Device supports monotonic timeline semaphores for cross-engine sync.
    pub timeline_sync: bool,
    /// Device supports cross-queue family concurrent sharing.
    pub cross_queue_sharing: bool,
    /// Device supports direct presentation from compute storage textures.
    pub direct_presentation: bool,
    /// Minimum row pitch alignment required for linear textures in bytes.
    pub pitch_alignment_bytes: u32,
}

impl ExternalResourceCapability {
    /// Capability set a current discrete desktop device offers.
    #[must_use]
    pub const fn desktop_gpu_standard() -> Self {
        Self {
            zero_copy_import: true,
            zero_copy_export: true,
            timeline_sync: true,
            cross_queue_sharing: true,
            direct_presentation: true,
            pitch_alignment_bytes: 256,
        }
    }
}

/// Complete admission and ownership record for an admitted image or external resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedResourceRecord {
    /// Stable logical resource name.
    pub name: String,
    /// Owning physical device identifier.
    pub device_id: String,
    /// Primary format semantics.
    pub format: ImageFormat,
    /// Dimensionality.
    pub dimension: ImageDimension,
    /// Extents in texels/layers.
    pub extent: ImageExtent,
    /// Plane layout for multi-planar or row-pitched resources.
    pub planes: Vec<PlaneDescriptor>,
    /// Permitted usage flags verified at admission.
    pub usages: ResourceUsageFlags,
    /// Subresource views attached to this resource.
    pub views: Vec<ViewDescriptor>,
    /// Alias group identifier (resources in same alias group share physical memory).
    pub alias_group: Option<u32>,
    /// Lifetime discipline.
    pub lifetime: ValueLifetime,
    /// Associated synchronization protocol.
    pub sync_protocol: Option<ExternalSyncProtocol>,
    /// Monotonic generation counter; incremented on resize or rebinding.
    pub generation: u64,
}

/// Errors occurring during resource ABI validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourceAbiError {
    /// Dimensions are zero.
    #[error("zero extent not permitted for image `{name}`: width {width}, height {height}")]
    ZeroExtent {
        /// Resource name.
        name: String,
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
    /// Row pitch is insufficient for image width.
    #[error("row pitch {row_pitch} is insufficient for width {width} * {bytes_per_pixel} B/px")]
    InsufficientRowPitch {
        /// Row pitch in bytes.
        row_pitch: u32,
        /// Width in texels.
        width: u32,
        /// Bytes per pixel.
        bytes_per_pixel: u32,
    },
    /// Row pitch does not satisfy driver alignment requirements.
    #[error("row pitch {row_pitch} B is not aligned to required boundary {alignment} B")]
    UnalignedRowPitch {
        /// Row pitch in bytes.
        row_pitch: u32,
        /// Alignment in bytes.
        alignment: u32,
    },
    /// View format is incompatible with base image format.
    #[error("view format {view_format:?} is incompatible with base image format {base_format:?}")]
    IncompatibleViewFormat {
        /// View format.
        view_format: ImageFormat,
        /// Base image format.
        base_format: ImageFormat,
    },
}

impl AdmittedResourceRecord {
    /// Validate all structural and alignment constraints of this record.
    pub fn validate(&self, caps: &ExternalResourceCapability) -> Result<(), ResourceAbiError> {
        if self.extent.width == 0
            || self.extent.height == 0
            || self.extent.depth_or_array_layers == 0
        {
            return Err(ResourceAbiError::ZeroExtent {
                name: self.name.clone(),
                width: self.extent.width,
                height: self.extent.height,
            });
        }

        let bpp = self.format.bytes_per_pixel();
        let min_pitch = self.extent.width.saturating_mul(bpp);

        for plane in &self.planes {
            if plane.row_pitch_bytes < min_pitch {
                return Err(ResourceAbiError::InsufficientRowPitch {
                    row_pitch: plane.row_pitch_bytes,
                    width: self.extent.width,
                    bytes_per_pixel: bpp,
                });
            }
            if caps.pitch_alignment_bytes > 0
                && plane.row_pitch_bytes % caps.pitch_alignment_bytes != 0
            {
                return Err(ResourceAbiError::UnalignedRowPitch {
                    row_pitch: plane.row_pitch_bytes,
                    alignment: caps.pitch_alignment_bytes,
                });
            }
        }

        for view in &self.views {
            if view.format.bytes_per_pixel() != self.format.bytes_per_pixel() {
                return Err(ResourceAbiError::IncompatibleViewFormat {
                    view_format: view.format,
                    base_format: self.format,
                });
            }
        }

        Ok(())
    }
}
