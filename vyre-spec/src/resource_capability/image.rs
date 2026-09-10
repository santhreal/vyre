//! Image formats, planes, subresources and views.
//!
//! Everything a caller needs to name a region of image memory and describe how
//! it is read. Separate from the sampler and interop capabilities because a
//! format table changes on its own schedule and neither side reads the other.

use serde::{Deserialize, Serialize};

/// Classification family for image formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatClass {
    /// Unsigned normalized integer channels.
    Unorm,
    /// sRGB non-linear normalized color channels.
    Srgb,
    /// IEEE-754 floating point channels.
    Float,
    /// Unsigned integer channels.
    Uint,
    /// Signed integer channels.
    Sint,
    /// Depth or stencil attachment formats.
    DepthStencil,
    /// Multi-planar video formats (e.g. YUV).
    PlanarVideo,
}

/// Pixel and element formats for domain-neutral image and plane resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    /// 8-bit single channel unsigned normalized.
    R8Unorm,
    /// 8-bit single channel unsigned integer.
    R8Uint,
    /// 8-bit single channel signed integer.
    R8Sint,
    /// 8-bit two-channel unsigned normalized.
    Rg8Unorm,
    /// 8-bit two-channel unsigned integer.
    Rg8Uint,
    /// 8-bit two-channel signed integer.
    Rg8Sint,
    /// 8-bit four-channel unsigned normalized.
    Rgba8Unorm,
    /// 8-bit four-channel sRGB normalized.
    Rgba8Srgb,
    /// 8-bit four-channel BGRA unsigned normalized (common display format).
    Bgra8Unorm,
    /// 8-bit four-channel BGRA sRGB normalized.
    Bgra8Srgb,
    /// 8-bit four-channel unsigned integer.
    Rgba8Uint,
    /// 8-bit four-channel signed integer.
    Rgba8Sint,
    /// 16-bit unsigned integer single channel.
    R16Uint,
    /// 16-bit signed integer single channel.
    R16Sint,
    /// 16-bit float single channel.
    R16Float,
    /// 16-bit unsigned integer two channel.
    Rg16Uint,
    /// 16-bit signed integer two channel.
    Rg16Sint,
    /// 16-bit float two channel.
    Rg16Float,
    /// 16-bit unsigned integer four channel.
    Rgba16Uint,
    /// 16-bit signed integer four channel.
    Rgba16Sint,
    /// 16-bit float four channel (HDR rendering / compute).
    Rgba16Float,
    /// 32-bit unsigned integer single channel.
    R32Uint,
    /// 32-bit signed integer single channel.
    R32Sint,
    /// 32-bit float single channel.
    R32Float,
    /// 32-bit unsigned integer two channel.
    Rg32Uint,
    /// 32-bit signed integer two channel.
    Rg32Sint,
    /// 32-bit float two channel.
    Rg32Float,
    /// 32-bit unsigned integer four channel.
    Rgba32Uint,
    /// 32-bit signed integer four channel.
    Rgba32Sint,
    /// 32-bit float four channel.
    Rgba32Float,
    /// Depth 24-bit plus normalized.
    Depth24Plus,
    /// Depth 32-bit float.
    Depth32Float,
    /// Depth 24-bit plus 8-bit stencil.
    Depth24PlusStencil8,
    /// Depth 32-bit float plus 8-bit stencil.
    Depth32FloatStencil8,
    /// Planar YUV 4:2:0.
    Yuv420Planar,
    /// Semi-planar YUV 4:2:0 (NV12).
    Yuv420SemiPlanar,
    /// Planar YUV 4:2:2.
    Yuv422Planar,
    /// Planar YUV 4:4:4.
    Yuv444Planar,
}

impl ImageFormat {
    /// Return the byte size of one pixel or primary texel block.
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::R8Unorm | Self::R8Uint | Self::R8Sint => 1,
            Self::Rg8Unorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Depth24Plus
            | Self::Depth32Float
            | Self::Depth24PlusStencil8 => 4,
            Self::Depth32FloatStencil8
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float => 8,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 16,
            Self::Yuv420Planar
            | Self::Yuv420SemiPlanar
            | Self::Yuv422Planar
            | Self::Yuv444Planar => 1,
        }
    }

    /// Return the format classification family.
    #[must_use]
    pub const fn format_class(&self) -> FormatClass {
        match self {
            Self::R8Unorm | Self::Rg8Unorm | Self::Rgba8Unorm | Self::Bgra8Unorm => {
                FormatClass::Unorm
            }
            Self::Rgba8Srgb | Self::Bgra8Srgb => FormatClass::Srgb,
            Self::R16Float
            | Self::Rg16Float
            | Self::Rgba16Float
            | Self::R32Float
            | Self::Rg32Float
            | Self::Rgba32Float => FormatClass::Float,
            Self::R8Uint
            | Self::Rg8Uint
            | Self::Rgba8Uint
            | Self::R16Uint
            | Self::Rg16Uint
            | Self::Rgba16Uint
            | Self::R32Uint
            | Self::Rg32Uint
            | Self::Rgba32Uint => FormatClass::Uint,
            Self::R8Sint
            | Self::Rg8Sint
            | Self::Rgba8Sint
            | Self::R16Sint
            | Self::Rg16Sint
            | Self::Rgba16Sint
            | Self::R32Sint
            | Self::Rg32Sint
            | Self::Rgba32Sint => FormatClass::Sint,
            Self::Depth24Plus
            | Self::Depth32Float
            | Self::Depth24PlusStencil8
            | Self::Depth32FloatStencil8 => FormatClass::DepthStencil,
            Self::Yuv420Planar
            | Self::Yuv420SemiPlanar
            | Self::Yuv422Planar
            | Self::Yuv444Planar => FormatClass::PlanarVideo,
        }
    }

    /// Return number of logical color / data channels.
    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        match self {
            Self::R8Unorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Depth24Plus
            | Self::Depth32Float => 1,
            Self::Rg8Unorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Depth24PlusStencil8
            | Self::Depth32FloatStencil8 => 2,
            Self::Yuv420Planar
            | Self::Yuv420SemiPlanar
            | Self::Yuv422Planar
            | Self::Yuv444Planar => 3,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float => 4,
        }
    }

    /// Whether this format is a depth or stencil attachment format.
    #[must_use]
    pub const fn is_depth_stencil(&self) -> bool {
        matches!(
            self,
            Self::Depth24Plus
                | Self::Depth32Float
                | Self::Depth24PlusStencil8
                | Self::Depth32FloatStencil8
        )
    }

    /// Whether this format represents multi-planar video content.
    #[must_use]
    pub const fn is_planar_video(&self) -> bool {
        matches!(
            self,
            Self::Yuv420Planar | Self::Yuv420SemiPlanar | Self::Yuv422Planar | Self::Yuv444Planar
        )
    }
}

/// Color space and transfer function semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorInterpretation {
    /// Linear RGB colorspace with linear transfer function.
    LinearRgb,
    /// Standard sRGB gamut with non-linear sRGB transfer function.
    Srgb,
    /// Wide gamut Display P3.
    DisplayP3,
    /// ITU-R BT.709 standard gamut.
    Bt709,
    /// ITU-R BT.2020 wide color gamut.
    Bt2020,
    /// HDR10 with SMPTE ST 2084 (PQ) transfer function.
    Hdr10,
    /// Passthrough / uninterpreted raw data.
    Passthrough,
}

/// Spatial and array dimensions of an image resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageDimensions {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels (1 for 1D images).
    pub height: u32,
    /// Depth in pixels (1 for 1D/2D images).
    pub depth: u32,
    /// Number of array layers (1 for non-array images).
    pub array_layers: u32,
    /// Number of mipmap levels.
    pub mip_levels: u32,
    /// Multisample anti-aliasing sample count (1 for non-multisampled).
    pub sample_count: u32,
}

impl ImageDimensions {
    /// Construct standard 1D dimensions.
    #[must_use]
    pub const fn d1(width: u32) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
            array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
        }
    }

    /// Construct standard 2D dimensions.
    #[must_use]
    pub const fn d2(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
            array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
        }
    }

    /// Construct standard 3D dimensions.
    #[must_use]
    pub const fn d3(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
            array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
        }
    }

    /// Check if dimensions are valid non-zero.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.depth > 0
            && self.array_layers > 0
            && self.mip_levels > 0
            && self.sample_count > 0
    }

    /// Compute the unpadded byte size for layer 0, mip 0 under `format`.
    #[must_use]
    pub const fn unpadded_layer_bytes(&self, format: ImageFormat) -> u64 {
        (self.width as u64)
            .saturating_mul(self.height as u64)
            .saturating_mul(self.depth as u64)
            .saturating_mul(format.bytes_per_pixel() as u64)
    }
}

/// Plane identification within single-planar or multi-planar resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaneKind {
    /// Single contiguous plane.
    Single,
    /// Luma (Y) plane in planar/semi-planar formats.
    Y,
    /// Chroma U (Cb) plane in planar formats.
    U,
    /// Chroma V (Cr) plane in planar formats.
    V,
    /// Interleaved Chroma UV (CbCr) plane in semi-planar formats (NV12).
    Uv,
    /// Depth plane.
    Depth,
    /// Stencil plane.
    Stencil,
}

/// Plane descriptor for multi-planar and disjoint image allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImagePlane {
    /// Zero-based plane index.
    pub plane_index: u32,
    /// Kind of content in this plane.
    pub plane_kind: PlaneKind,
    /// Byte offset of the plane from the base allocation.
    pub offset_bytes: u64,
    /// Row pitch in bytes for this plane.
    pub row_pitch_bytes: u32,
    /// Plane height in lines.
    pub plane_height: u32,
    /// Subsampled format of this specific plane.
    pub format: ImageFormat,
}

/// Specific subresource slice within a multi-layer or multi-mip image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubresourceRange {
    /// First mip level included.
    pub base_mip_level: u32,
    /// Number of mip levels included.
    pub mip_level_count: u32,
    /// First array layer included.
    pub base_array_layer: u32,
    /// Number of array layers included.
    pub array_layer_count: u32,
    /// Aspect mask (e.g. 1 = Color, 2 = Depth, 4 = Stencil).
    pub aspect_mask: u32,
}

impl SubresourceRange {
    /// Color aspect mask.
    pub const ASPECT_COLOR: u32 = 1 << 0;
    /// Depth aspect mask.
    pub const ASPECT_DEPTH: u32 = 1 << 1;
    /// Stencil aspect mask.
    pub const ASPECT_STENCIL: u32 = 1 << 2;

    /// Full subresource range covering the entire image.
    #[must_use]
    pub const fn full(dimensions: &ImageDimensions) -> Self {
        Self {
            base_mip_level: 0,
            mip_level_count: dimensions.mip_levels,
            base_array_layer: 0,
            array_layer_count: dimensions.array_layers,
            aspect_mask: Self::ASPECT_COLOR,
        }
    }
}

/// Dimension kind of an image view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageViewKind {
    /// 1D texture view.
    D1,
    /// 2D texture view.
    D2,
    /// 2D array texture view.
    D2Array,
    /// 3D volume texture view.
    D3,
    /// Cubemap texture view.
    Cube,
    /// Cubemap array texture view.
    CubeArray,
}

/// Swizzle mapping component for image views.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwizzleComponent {
    /// Identity channel mapping.
    Identity,
    /// Fixed zero constant.
    Zero,
    /// Fixed one constant.
    One,
    /// Map from Red source channel.
    R,
    /// Map from Green source channel.
    G,
    /// Map from Blue source channel.
    B,
    /// Map from Alpha source channel.
    A,
}

/// Component swizzle configuration for four channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentSwizzle {
    /// Red component swizzle.
    pub r: SwizzleComponent,
    /// Green component swizzle.
    pub g: SwizzleComponent,
    /// Blue component swizzle.
    pub b: SwizzleComponent,
    /// Alpha component swizzle.
    pub a: SwizzleComponent,
}

impl ComponentSwizzle {
    /// Identity swizzle (RGBA -> RGBA).
    pub const IDENTITY: Self = Self {
        r: SwizzleComponent::Identity,
        g: SwizzleComponent::Identity,
        b: SwizzleComponent::Identity,
        a: SwizzleComponent::Identity,
    };
}

/// Domain-neutral image view descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageViewDescriptor {
    /// Format reinterpretation for this view.
    pub format: ImageFormat,
    /// Dimension kind of the view.
    pub view_kind: ImageViewKind,
    /// Subresource range selected by this view.
    pub subresource_range: SubresourceRange,
    /// Component swizzle mapping.
    pub swizzle: ComponentSwizzle,
}
