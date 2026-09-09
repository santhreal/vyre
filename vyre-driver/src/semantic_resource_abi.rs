//! Domain-neutral semantic resource ABI, logical image/view types, zero-copy import/export,
//! and timeline synchronization (Row 111).
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module defines domain-neutral typed image, plane, view, sampler, external-memory,
//! and external-event capabilities separate from semantic algorithms and concrete API handles.
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ResidentOwner;
/// Pixel and element formats for domain-neutral image and plane resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    /// 8-bit single channel unsigned normalized.
    R8Unorm,
    /// 8-bit two-channel unsigned normalized.
    Rg8Unorm,
    /// 8-bit four-channel unsigned normalized.
    Rgba8Unorm,
    /// 8-bit four-channel sRGB normalized.
    Rgba8Srgb,
    /// 8-bit four-channel BGRA unsigned normalized (common display format).
    Bgra8Unorm,
    /// 8-bit four-channel BGRA sRGB normalized.
    Bgra8Srgb,
    /// 16-bit float single channel.
    R16Float,
    /// 16-bit float two channel.
    Rg16Float,
    /// 16-bit float four channel (HDR rendering / compute).
    Rgba16Float,
    /// 32-bit float single channel.
    R32Float,
    /// 32-bit float two channel.
    Rg32Float,
    /// 32-bit float four channel.
    Rgba32Float,
    /// 32-bit unsigned integer single channel.
    R32Uint,
    /// 32-bit signed integer single channel.
    R32Sint,
    /// Depth 24-bit plus normalized.
    Depth24Plus,
    /// Depth 32-bit float.
    Depth32Float,
    /// Depth 24-bit plus 8-bit stencil.
    Depth24PlusStencil8,
    /// Planar YUV 4:2:0.
    Yuv420Planar,
    /// Semi-planar YUV 4:2:0 (NV12).
    Yuv420SemiPlanar,
}

impl ImageFormat {
    /// Return the byte size of one pixel or block.
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::R16Float => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rg16Float
            | Self::R32Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::Depth24Plus
            | Self::Depth32Float
            | Self::Depth24PlusStencil8 => 4,
            Self::Rgba16Float | Self::Rg32Float => 8,
            Self::Rgba32Float => 16,
            Self::Yuv420Planar | Self::Yuv420SemiPlanar => 1, // Subsampled plane baseline
        }
    }

    /// Whether this format is a depth or stencil attachment format.
    #[must_use]
    pub const fn is_depth_stencil(&self) -> bool {
        matches!(
            self,
            Self::Depth24Plus | Self::Depth32Float | Self::Depth24PlusStencil8
        )
    }

    /// Whether this format represents multi-planar video content.
    #[must_use]
    pub const fn is_planar_video(&self) -> bool {
        matches!(self, Self::Yuv420Planar | Self::Yuv420SemiPlanar)
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

    /// Compute the unpadded byte size for layer 0, mip 0 under `format`.
    #[must_use]
    pub const fn unpadded_layer_bytes(&self, format: ImageFormat) -> u64 {
        (self.width as u64)
            .saturating_mul(self.height as u64)
            .saturating_mul(self.depth as u64)
            .saturating_mul(format.bytes_per_pixel() as u64)
    }
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
    /// Aspect mask (e.g. Color, Depth, Stencil).
    pub aspect_mask: u32,
}

impl SubresourceRange {
    /// Full subresource range covering the entire image.
    #[must_use]
    pub const fn full(dimensions: &ImageDimensions) -> Self {
        Self {
            base_mip_level: 0,
            mip_level_count: dimensions.mip_levels,
            base_array_layer: 0,
            array_layer_count: dimensions.array_layers,
            aspect_mask: 1, // Color / Primary aspect
        }
    }
}

/// Bitflag set of permitted usages for an admitted resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourcePermittedUsages(pub u32);

impl ResourcePermittedUsages {
    /// Read via texture sampler in shader.
    pub const SAMPLED: Self = Self(1 << 0);
    /// Read/write storage image in compute shader.
    pub const STORAGE: Self = Self(1 << 1);
    /// Render target color attachment.
    pub const COLOR_ATTACHMENT: Self = Self(1 << 2);
    /// Render target depth/stencil attachment.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 3);
    /// Source for copy operations.
    pub const TRANSFER_SRC: Self = Self(1 << 4);
    /// Destination for copy operations.
    pub const TRANSFER_DST: Self = Self(1 << 5);
    /// Imported from external OS/graphics API.
    pub const EXTERNAL_IMPORT: Self = Self(1 << 6);
    /// Exported to external OS/graphics API.
    pub const EXTERNAL_EXPORT: Self = Self(1 << 7);
    /// Presentable to display swapchain.
    pub const PRESENTATION: Self = Self(1 << 8);

    /// Check whether `self` contains all flags in `other`.
    #[must_use]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of two usage sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Ownership state of a resident resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceOwnershipState {
    /// Owned exclusively by one backend instance.
    Exclusive(ResidentOwner),
    /// Concurrently shared across multiple backend owners without queue transfers.
    SharedConcurrent(Vec<ResidentOwner>),
    /// In flight during cross-queue or cross-engine ownership transfer.
    QueueFamilyTransfer {
        /// Source queue index.
        source_queue: u32,
        /// Destination queue index.
        target_queue: u32,
    },
    /// Owned externally by presentation engine or OS compositor.
    ExternalHost,
}

/// Memory layout and pipeline stage state for transition tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLayoutState {
    /// Undefined initial content.
    Undefined,
    /// General read/write layout for compute storage.
    General,
    /// Optimized layout for read-only texture sampling.
    ShaderReadOnly,
    /// Optimal layout for color rendering output.
    ColorAttachmentOptimal,
    /// Optimal layout for depth/stencil read/write.
    DepthStencilOptimal,
    /// Source of blit/transfer operations.
    TransferSrcOptimal,
    /// Destination of blit/transfer operations.
    TransferDstOptimal,
    /// Display swapchain presentation layout.
    PresentSrc,
}

/// Explicit layout and usage transition descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceUsageTransition {
    /// Source layout.
    pub from_layout: ResourceLayoutState,
    /// Target layout.
    pub to_layout: ResourceLayoutState,
    /// Source access flags.
    pub source_usage: ResourcePermittedUsages,
    /// Target access flags.
    pub target_usage: ResourcePermittedUsages,
    /// Whether an explicit execution barrier is required.
    pub requires_barrier: bool,
}

impl ResourceUsageTransition {
    /// Create a transition to shader read-only sampling.
    #[must_use]
    pub const fn to_sampled(from: ResourceLayoutState) -> Self {
        Self {
            from_layout: from,
            to_layout: ResourceLayoutState::ShaderReadOnly,
            source_usage: ResourcePermittedUsages::STORAGE,
            target_usage: ResourcePermittedUsages::SAMPLED,
            requires_barrier: true,
        }
    }

    /// Create a transition to presentation swapchain format.
    #[must_use]
    pub const fn to_present(from: ResourceLayoutState) -> Self {
        Self {
            from_layout: from,
            to_layout: ResourceLayoutState::PresentSrc,
            source_usage: ResourcePermittedUsages::COLOR_ATTACHMENT,
            target_usage: ResourcePermittedUsages::PRESENTATION,
            requires_barrier: true,
        }
    }
}

/// Domain-neutral timeline synchronization primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelineSyncProtocol {
    /// Monotonically increasing timeline semaphore.
    TimelineSemaphore {
        /// Timeline semaphore ID.
        timeline_id: u64,
        /// Value to wait on before execution.
        wait_value: u64,
        /// Value to signal upon execution completion.
        signal_value: u64,
    },
    /// Binary GPU fence.
    Fence {
        /// Fence ID.
        fence_id: u64,
        /// Current signaled state.
        is_signaled: bool,
    },
    /// Metal shared event with 64-bit signaled value.
    MetalSharedEvent {
        /// Event ID.
        event_id: u64,
        /// Signal value.
        signal_value: u64,
    },
    /// POSIX sync file file descriptor.
    SyncFileFd {
        /// File descriptor number.
        fd: i32,
    },
    /// Direct submission with implicit in-order queue synchronization.
    ImplicitQueue,
}

/// External memory handle capabilities for zero-copy import/export.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalMemoryCapability {
    /// Linux DMA-BUF file descriptor.
    DmaBuf,
    /// Windows NT handle for external memory.
    Win32Nt,
    /// Apple Metal shared texture / buffer.
    MetalSharedResource,
    /// Host pinned virtual memory.
    HostAllocation,
}

/// Provenance and descriptor record for an admitted domain-neutral resource (Row 111).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedResourceRecord {
    /// Unique resource identifier.
    pub resource_id: u64,
    /// Associated device identifier.
    pub device_id: u64,
    /// Monotonically increasing resource mutation generation.
    pub generation: u64,
    /// Format and element layout.
    pub format: ImageFormat,
    /// Color interpretation and gamut.
    pub color: ColorInterpretation,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes (including hardware alignment padding).
    pub row_pitch_bytes: u32,
    /// Subresource range.
    pub subresource: SubresourceRange,
    /// Permitted usage flags.
    pub permitted_usages: ResourcePermittedUsages,
    /// Current ownership state.
    pub ownership: ResourceOwnershipState,
    /// Alias set ID if memory is shared with other resources.
    pub alias_set_id: Option<u64>,
    /// Synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
    /// External memory export capability if zero-copy enabled.
    pub external_memory: Option<ExternalMemoryCapability>,
    /// Whether this resource was admitted without host copies.
    pub is_zero_copy: bool,
    /// Whether this resource is valid (invalidated on device loss or generation mismatch).
    pub is_valid: bool,
}

impl AdmittedResourceRecord {
    /// Create an admitted 2D texture resource record.
    #[must_use]
    pub fn new_2d(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner: ResidentOwner,
    ) -> Self {
        let dimensions = ImageDimensions::d2(width, height);
        let min_pitch = width * format.bytes_per_pixel();
        let aligned_pitch = (min_pitch + 255) & !255; // 256-byte hardware row alignment

        Self {
            resource_id,
            device_id,
            generation: 1,
            format,
            color,
            dimensions,
            row_pitch_bytes: aligned_pitch,
            subresource: SubresourceRange::full(&dimensions),
            permitted_usages,
            ownership: ResourceOwnershipState::Exclusive(owner),
            alias_set_id: None,
            sync_protocol: TimelineSyncProtocol::ImplicitQueue,
            external_memory: None,
            is_zero_copy: true,
            is_valid: true,
        }
    }

    /// Invalidate the resource record due to device loss.
    pub fn invalidate_on_device_loss(&mut self) {
        self.is_valid = false;
    }

    /// Advance the generation counter upon mutation, validating that expected generation matches.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::GenerationMismatch`] if expected generation does not match.
    pub fn advance_generation(&mut self, expected_gen: u64) -> Result<u64, ResourceAbiError> {
        if !self.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated {
                resource_id: self.resource_id,
            });
        }
        if self.generation != expected_gen {
            return Err(ResourceAbiError::GenerationMismatch {
                resource_id: self.resource_id,
                expected: expected_gen,
                actual: self.generation,
            });
        }
        self.generation = self.generation.saturating_add(1);
        Ok(self.generation)
    }

    /// Negotiate zero-copy external memory import.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] if format or memory capability is unsupported.
    pub fn negotiate_zero_copy_import(
        &mut self,
        capability: ExternalMemoryCapability,
    ) -> Result<(), ResourceAbiError> {
        if !self.permitted_usages.contains(ResourcePermittedUsages::EXTERNAL_IMPORT) {
            return Err(ResourceAbiError::UsageNotPermitted {
                resource_id: self.resource_id,
                requested_usage: ResourcePermittedUsages::EXTERNAL_IMPORT,
            });
        }
        self.external_memory = Some(capability);
        self.is_zero_copy = true;
        Ok(())
    }
}

/// Errors occurring during semantic resource ABI validation and negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ResourceAbiError {
    /// Resource has been invalidated due to device loss or teardown.
    #[error("resource {resource_id} is invalidated due to device loss")]
    ResourceInvalidated {
        /// Resource ID.
        resource_id: u64,
    },
    /// Resource generation mismatch (stale frame access detected).
    #[error("stale resource {resource_id} generation access: expected {expected}, got {actual}")]
    GenerationMismatch {
        /// Resource ID.
        resource_id: u64,
        /// Expected generation.
        expected: u64,
        /// Actual current generation.
        actual: u64,
    },
    /// Requested usage is not permitted for this resource.
    #[error("usage {requested_usage:?} is not permitted for resource {resource_id}")]
    UsageNotPermitted {
        /// Resource ID.
        resource_id: u64,
        /// Requested usage flag.
        requested_usage: ResourcePermittedUsages,
    },
    /// Unsupported zero-copy import/export negotiation.
    #[error("unsupported zero-copy memory negotiation for resource {resource_id}")]
    UnsupportedZeroCopyNegotiation {
        /// Resource ID.
        resource_id: u64,
    },
}
