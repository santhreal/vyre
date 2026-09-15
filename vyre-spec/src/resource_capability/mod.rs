//! Domain-neutral resource, image, view, plane, sampler, external memory,
//! and timeline synchronization capabilities.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering,
//! and presentation boundaries without host copies or implicit global waits.
//!
//! This module defines domain-neutral typed image, plane, view, sampler, external-memory,
//! and external-event capabilities separate from semantic algorithms and concrete API handles.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};

mod image;
mod sampler;

pub use image::*;
pub use sampler::*;

/// Bitflag set of permitted usages for an admitted resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourcePermittedUsages(pub u32);

impl ResourcePermittedUsages {
    /// Read via texture sampler in shader.
    pub const SAMPLED: Self = Self(1 << 0);
    /// Read access in storage image / buffer.
    pub const STORAGE_READ: Self = Self(1 << 1);
    /// Write access in storage image / buffer.
    pub const STORAGE_WRITE: Self = Self(1 << 2);
    /// Render target color attachment.
    pub const COLOR_ATTACHMENT: Self = Self(1 << 3);
    /// Render target depth/stencil attachment.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 4);
    /// Source for copy operations.
    pub const TRANSFER_SRC: Self = Self(1 << 5);
    /// Destination for copy operations.
    pub const TRANSFER_DST: Self = Self(1 << 6);
    /// Imported from external OS/graphics API.
    pub const EXTERNAL_IMPORT: Self = Self(1 << 7);
    /// Exported to external OS/graphics API.
    pub const EXTERNAL_EXPORT: Self = Self(1 << 8);
    /// Presentable to display swapchain.
    pub const PRESENTATION: Self = Self(1 << 9);

    /// Combined read-write storage usage.
    pub const STORAGE: Self = Self((1 << 1) | (1 << 2));

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

    /// Intersection of two usage sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

/// Ownership state of a resident resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceOwnershipState {
    /// Owned exclusively by one backend instance.
    Exclusive(u64),
    /// Concurrently shared across multiple backend owners without queue transfers.
    SharedConcurrent(Vec<u64>),
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

/// Lifetime policy and lease management for an admitted resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLifetimeState {
    /// Transient allocation valid only within one kernel or command buffer execution.
    Transient,
    /// Retained resource managed across multiple executions within an execution session.
    Retained,
    /// Leased resource with an explicit lease identifier and generation bound.
    Leased {
        /// Unique lease identifier.
        lease_id: u64,
    },
    /// Pinned external resource backed by an OS-level shared allocation.
    ExternalPinned,
    /// Presentable swapchain image managed by presentation engine handoff.
    SwapchainPresented,
}

/// Provenance of an admitted resource indicating its origin and import parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceProvenance {
    /// Internally allocated by Vyre device allocator.
    InternalAllocation {
        /// Allocator tag identifying the allocator module.
        allocator_tag: u32,
        /// Byte size allocated.
        byte_size: u64,
    },
    /// Imported from an external graphics API or OS memory handle.
    ExternalImport {
        /// Kind of external memory handle imported.
        memory_kind: ExternalMemoryKind,
        /// Whether this resource can be re-exported.
        exportable: bool,
        /// Authenticated handle identity tag.
        handle_tag: u64,
    },
    /// Derived subresource view of an existing parent resource.
    ViewDerived {
        /// Parent resource identifier.
        parent_id: u64,
        /// Subresource range in the parent resource.
        subresource: SubresourceRange,
    },
}

/// Aliasing group membership for memory-sharing verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceAliasSet {
    /// No memory aliasing with any other resource.
    None,
    /// Belongs to an explicit alias group sharing physical device memory.
    Group {
        /// Unique alias group identifier.
        group_id: u64,
    },
    /// Disjoint subresource planes sharing physical backing with non-overlapping subresources.
    DisjointPlanes {
        /// Disjoint group identifier.
        group_id: u64,
    },
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

    /// Create a transition from general compute storage to color attachment.
    #[must_use]
    pub const fn storage_to_color_attachment() -> Self {
        Self {
            from_layout: ResourceLayoutState::General,
            to_layout: ResourceLayoutState::ColorAttachmentOptimal,
            source_usage: ResourcePermittedUsages::STORAGE,
            target_usage: ResourcePermittedUsages::COLOR_ATTACHMENT,
            requires_barrier: true,
        }
    }
}

/// Admitted resource descriptor record capturing complete specification state.
///
/// An admitted resource records:
/// 1. Device identity (`device_id`)
/// 2. Format and color semantics (`format`, `color`)
/// 3. Dimensions and pitch (`dimensions`, `row_pitch_bytes`)
/// 4. Subresource range (`subresource`)
/// 5. Permitted usages (`permitted_usages`)
/// 6. Ownership state (`ownership`)
/// 7. Alias set (`alias_set`)
/// 8. Lifetime (`lifetime`)
/// 9. Synchronization protocol (`sync_protocol`)
/// 10. Provenance (`provenance`)
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
    /// Alias set configuration.
    pub alias_set: ResourceAliasSet,
    /// Resource lifetime state.
    pub lifetime: ResourceLifetimeState,
    /// Synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
    /// Resource origin provenance.
    pub provenance: ResourceProvenance,
    /// Whether this resource was admitted without host copies.
    pub is_zero_copy: bool,
    /// Whether this resource is valid (invalidated on device loss or generation mismatch).
    pub is_valid: bool,
}

impl AdmittedResourceRecord {
    /// Construct an admitted 2D texture resource record.
    #[must_use]
    pub fn new_2d(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner_id: u64,
    ) -> Self {
        let dimensions = ImageDimensions::d2(width, height);
        let min_pitch = width * format.bytes_per_pixel();
        let aligned_pitch = (min_pitch + 255) & !255; // 256-byte hardware row alignment
        let byte_size = (aligned_pitch as u64) * (height as u64);

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
            ownership: ResourceOwnershipState::Exclusive(owner_id),
            alias_set: ResourceAliasSet::None,
            lifetime: ResourceLifetimeState::Retained,
            sync_protocol: TimelineSyncProtocol::ImplicitQueue,
            provenance: ResourceProvenance::InternalAllocation {
                allocator_tag: 1,
                byte_size,
            },
            is_zero_copy: true,
            is_valid: true,
        }
    }

    /// Construct an admitted external import 2D texture record.
    #[must_use]
    pub fn new_external_import_2d(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        row_pitch_bytes: u32,
        permitted_usages: ResourcePermittedUsages,
        memory_kind: ExternalMemoryKind,
        handle_tag: u64,
        sync_protocol: TimelineSyncProtocol,
    ) -> Self {
        let dimensions = ImageDimensions::d2(width, height);
        Self {
            resource_id,
            device_id,
            generation: 1,
            format,
            color,
            dimensions,
            row_pitch_bytes,
            subresource: SubresourceRange::full(&dimensions),
            permitted_usages: permitted_usages.union(ResourcePermittedUsages::EXTERNAL_IMPORT),
            ownership: ResourceOwnershipState::ExternalHost,
            alias_set: ResourceAliasSet::None,
            lifetime: ResourceLifetimeState::ExternalPinned,
            sync_protocol,
            provenance: ResourceProvenance::ExternalImport {
                memory_kind,
                exportable: permitted_usages.contains(ResourcePermittedUsages::EXTERNAL_EXPORT),
                handle_tag,
            },
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
    /// Returns [`ResourceAbiError::GenerationMismatch`] if expected generation does not match,
    /// or [`ResourceAbiError::ResourceInvalidated`] if invalidated.
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
    /// Returns [`ResourceAbiError::UsageNotPermitted`] if external import is not permitted,
    /// or [`ResourceAbiError::ResourceInvalidated`] if the resource is invalidated.
    pub fn negotiate_zero_copy_import(
        &mut self,
        memory_kind: ExternalMemoryKind,
        handle_tag: u64,
    ) -> Result<(), ResourceAbiError> {
        if !self.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated {
                resource_id: self.resource_id,
            });
        }
        if !self
            .permitted_usages
            .contains(ResourcePermittedUsages::EXTERNAL_IMPORT)
        {
            return Err(ResourceAbiError::UsageNotPermitted {
                resource_id: self.resource_id,
                requested_usage: ResourcePermittedUsages::EXTERNAL_IMPORT,
            });
        }
        self.provenance = ResourceProvenance::ExternalImport {
            memory_kind,
            exportable: self
                .permitted_usages
                .contains(ResourcePermittedUsages::EXTERNAL_EXPORT),
            handle_tag,
        };
        self.is_zero_copy = true;
        Ok(())
    }
}

/// Errors occurring during semantic resource ABI validation and negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAbiError {
    /// Resource has been invalidated due to device loss or teardown.
    ResourceInvalidated {
        /// Resource ID.
        resource_id: u64,
    },
    /// Resource generation mismatch (stale frame access detected).
    GenerationMismatch {
        /// Resource ID.
        resource_id: u64,
        /// Expected generation.
        expected: u64,
        /// Actual current generation.
        actual: u64,
    },
    /// Requested usage is not permitted for this resource.
    UsageNotPermitted {
        /// Resource ID.
        resource_id: u64,
        /// Requested usage flag.
        requested_usage: ResourcePermittedUsages,
    },
    /// Unsupported zero-copy import/export negotiation combination.
    UnsupportedZeroCopyNegotiation {
        /// Resource ID.
        resource_id: u64,
        /// Format that was rejected.
        format: ImageFormat,
        /// External memory kind that was rejected.
        memory_kind: ExternalMemoryKind,
    },
    /// Invalid resource dimensions or pitch alignment.
    InvalidDimensionsOrPitch {
        /// Resource ID.
        resource_id: u64,
        /// Provided pitch.
        provided_pitch: u32,
        /// Minimum required pitch.
        required_pitch: u32,
    },
    /// Device loss invalidated the resource.
    DeviceLoss {
        /// Device ID that was lost.
        device_id: u64,
    },
}

impl fmt::Display for ResourceAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceInvalidated { resource_id } => {
                write!(
                    f,
                    "resource {resource_id} is invalidated due to device loss"
                )
            }
            Self::GenerationMismatch {
                resource_id,
                expected,
                actual,
            } => {
                write!(f, "stale resource {resource_id} generation access: expected {expected}, got {actual}")
            }
            Self::UsageNotPermitted {
                resource_id,
                requested_usage,
            } => {
                write!(
                    f,
                    "usage {requested_usage:?} is not permitted for resource {resource_id}"
                )
            }
            Self::UnsupportedZeroCopyNegotiation {
                resource_id,
                format,
                memory_kind,
            } => {
                write!(f, "unsupported zero-copy memory negotiation for resource {resource_id}: format {format:?} with {memory_kind:?}")
            }
            Self::InvalidDimensionsOrPitch {
                resource_id,
                provided_pitch,
                required_pitch,
            } => {
                write!(f, "invalid pitch {provided_pitch} for resource {resource_id}, required {required_pitch}")
            }
            Self::DeviceLoss { device_id } => {
                write!(f, "device loss on device {device_id}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Run-time variant enumerators for mutation-gate test completeness (BINDING)
// ---------------------------------------------------------------------------

/// Exhaustive slice of all canonical [`ImageFormat`] variants.
#[must_use]
pub const fn all_image_formats() -> &'static [ImageFormat] {
    &[
        ImageFormat::R8Unorm,
        ImageFormat::R8Uint,
        ImageFormat::R8Sint,
        ImageFormat::Rg8Unorm,
        ImageFormat::Rg8Uint,
        ImageFormat::Rg8Sint,
        ImageFormat::Rgba8Unorm,
        ImageFormat::Rgba8Srgb,
        ImageFormat::Bgra8Unorm,
        ImageFormat::Bgra8Srgb,
        ImageFormat::Rgba8Uint,
        ImageFormat::Rgba8Sint,
        ImageFormat::R16Uint,
        ImageFormat::R16Sint,
        ImageFormat::R16Float,
        ImageFormat::Rg16Uint,
        ImageFormat::Rg16Sint,
        ImageFormat::Rg16Float,
        ImageFormat::Rgba16Uint,
        ImageFormat::Rgba16Sint,
        ImageFormat::Rgba16Float,
        ImageFormat::R32Uint,
        ImageFormat::R32Sint,
        ImageFormat::R32Float,
        ImageFormat::Rg32Uint,
        ImageFormat::Rg32Sint,
        ImageFormat::Rg32Float,
        ImageFormat::Rgba32Uint,
        ImageFormat::Rgba32Sint,
        ImageFormat::Rgba32Float,
        ImageFormat::Depth24Plus,
        ImageFormat::Depth32Float,
        ImageFormat::Depth24PlusStencil8,
        ImageFormat::Depth32FloatStencil8,
        ImageFormat::Yuv420Planar,
        ImageFormat::Yuv420SemiPlanar,
        ImageFormat::Yuv422Planar,
        ImageFormat::Yuv444Planar,
    ]
}

/// Exhaustive slice of all canonical [`FormatClass`] variants.
#[must_use]
pub const fn all_format_classes() -> &'static [FormatClass] {
    &[
        FormatClass::Unorm,
        FormatClass::Srgb,
        FormatClass::Float,
        FormatClass::Uint,
        FormatClass::Sint,
        FormatClass::DepthStencil,
        FormatClass::PlanarVideo,
    ]
}

/// Exhaustive slice of all canonical [`ColorInterpretation`] variants.
#[must_use]
pub const fn all_color_interpretations() -> &'static [ColorInterpretation] {
    &[
        ColorInterpretation::LinearRgb,
        ColorInterpretation::Srgb,
        ColorInterpretation::DisplayP3,
        ColorInterpretation::Bt709,
        ColorInterpretation::Bt2020,
        ColorInterpretation::Hdr10,
        ColorInterpretation::Passthrough,
    ]
}

/// Exhaustive slice of all canonical [`PlaneKind`] variants.
#[must_use]
pub const fn all_plane_kinds() -> &'static [PlaneKind] {
    &[
        PlaneKind::Single,
        PlaneKind::Y,
        PlaneKind::U,
        PlaneKind::V,
        PlaneKind::Uv,
        PlaneKind::Depth,
        PlaneKind::Stencil,
    ]
}

/// Exhaustive slice of all canonical [`ImageViewKind`] variants.
#[must_use]
pub const fn all_image_view_kinds() -> &'static [ImageViewKind] {
    &[
        ImageViewKind::D1,
        ImageViewKind::D2,
        ImageViewKind::D2Array,
        ImageViewKind::D3,
        ImageViewKind::Cube,
        ImageViewKind::CubeArray,
    ]
}

/// Exhaustive slice of all canonical [`SwizzleComponent`] variants.
#[must_use]
pub const fn all_swizzle_components() -> &'static [SwizzleComponent] {
    &[
        SwizzleComponent::Identity,
        SwizzleComponent::Zero,
        SwizzleComponent::One,
        SwizzleComponent::R,
        SwizzleComponent::G,
        SwizzleComponent::B,
        SwizzleComponent::A,
    ]
}

/// Exhaustive slice of all canonical [`FilterMode`] variants.
#[must_use]
pub const fn all_filter_modes() -> &'static [FilterMode] {
    &[FilterMode::Nearest, FilterMode::Linear]
}

/// Exhaustive slice of all canonical [`MipmapFilterMode`] variants.
#[must_use]
pub const fn all_mipmap_filter_modes() -> &'static [MipmapFilterMode] {
    &[MipmapFilterMode::Nearest, MipmapFilterMode::Linear]
}

/// Exhaustive slice of all canonical [`AddressMode`] variants.
#[must_use]
pub const fn all_address_modes() -> &'static [AddressMode] {
    &[
        AddressMode::ClampToEdge,
        AddressMode::Repeat,
        AddressMode::MirrorRepeat,
        AddressMode::ClampToBorder,
    ]
}

/// Exhaustive slice of all canonical [`CompareFunction`] variants.
#[must_use]
pub const fn all_compare_functions() -> &'static [CompareFunction] {
    &[
        CompareFunction::Never,
        CompareFunction::Less,
        CompareFunction::Equal,
        CompareFunction::LessEqual,
        CompareFunction::Greater,
        CompareFunction::NotEqual,
        CompareFunction::GreaterEqual,
        CompareFunction::Always,
    ]
}

/// Exhaustive slice of all canonical [`BorderColor`] variants.
#[must_use]
pub const fn all_border_colors() -> &'static [BorderColor] {
    &[
        BorderColor::TransparentBlack,
        BorderColor::OpaqueBlack,
        BorderColor::OpaqueWhite,
    ]
}

/// Exhaustive slice of all canonical [`ExternalMemoryKind`] variants.
#[must_use]
pub const fn all_external_memory_kinds() -> &'static [ExternalMemoryKind] {
    &[
        ExternalMemoryKind::DmaBuf,
        ExternalMemoryKind::Win32Nt,
        ExternalMemoryKind::Win32Kmt,
        ExternalMemoryKind::MetalSharedResource,
        ExternalMemoryKind::HostAllocation,
        ExternalMemoryKind::OpaqueFd,
    ]
}

/// Exhaustive slice of all canonical [`ExternalEventKind`] variants.
#[must_use]
pub const fn all_external_event_kinds() -> &'static [ExternalEventKind] {
    &[
        ExternalEventKind::TimelineSemaphore,
        ExternalEventKind::BinaryFence,
        ExternalEventKind::MetalSharedEvent,
        ExternalEventKind::SyncFileFd,
        ExternalEventKind::ImplicitQueue,
    ]
}

/// Vector of representative instances for every [`TimelineSyncProtocol`] variant.
#[must_use]
pub fn all_sync_protocols() -> Vec<TimelineSyncProtocol> {
    vec![
        TimelineSyncProtocol::TimelineSemaphore {
            timeline_id: 1,
            wait_value: 0,
            signal_value: 1,
        },
        TimelineSyncProtocol::Fence {
            fence_id: 1,
            is_signaled: false,
        },
        TimelineSyncProtocol::MetalSharedEvent {
            event_id: 1,
            signal_value: 1,
        },
        TimelineSyncProtocol::SyncFileFd { fd: 3 },
        TimelineSyncProtocol::ImplicitQueue,
    ]
}

/// Exhaustive slice of individual [`ResourcePermittedUsages`] flag constants.
#[must_use]
pub const fn all_usage_flags() -> &'static [ResourcePermittedUsages] {
    &[
        ResourcePermittedUsages::SAMPLED,
        ResourcePermittedUsages::STORAGE_READ,
        ResourcePermittedUsages::STORAGE_WRITE,
        ResourcePermittedUsages::COLOR_ATTACHMENT,
        ResourcePermittedUsages::DEPTH_STENCIL_ATTACHMENT,
        ResourcePermittedUsages::TRANSFER_SRC,
        ResourcePermittedUsages::TRANSFER_DST,
        ResourcePermittedUsages::EXTERNAL_IMPORT,
        ResourcePermittedUsages::EXTERNAL_EXPORT,
        ResourcePermittedUsages::PRESENTATION,
    ]
}

/// Exhaustive slice of all canonical [`ResourceLayoutState`] variants.
#[must_use]
pub const fn all_layout_states() -> &'static [ResourceLayoutState] {
    &[
        ResourceLayoutState::Undefined,
        ResourceLayoutState::General,
        ResourceLayoutState::ShaderReadOnly,
        ResourceLayoutState::ColorAttachmentOptimal,
        ResourceLayoutState::DepthStencilOptimal,
        ResourceLayoutState::TransferSrcOptimal,
        ResourceLayoutState::TransferDstOptimal,
        ResourceLayoutState::PresentSrc,
    ]
}

/// Exhaustive slice of canonical string names for lifetime states.
#[must_use]
pub const fn all_lifetime_state_kinds() -> &'static [&'static str] {
    &[
        "transient",
        "retained",
        "leased",
        "external_pinned",
        "swapchain_presented",
    ]
}

/// Exhaustive slice of canonical string names for provenance kinds.
#[must_use]
pub const fn all_provenance_kinds() -> &'static [&'static str] {
    &["internal_allocation", "external_import", "view_derived"]
}

/// Exhaustive slice of canonical string names for alias set kinds.
#[must_use]
pub const fn all_alias_set_kinds() -> &'static [&'static str] {
    &["none", "group", "disjoint_planes"]
}
