//! Sampling state and the capabilities for sharing memory across APIs.
//!
//! Filter, address and compare modes describe how a device reads an image;
//! external memory and external event capabilities describe what a device can
//! share with another API without a host copy. Both are reported facts about a
//! device rather than descriptions of image memory.

use serde::{Deserialize, Serialize};

/// Filter mode for texture sampling minification/magnification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterMode {
    /// Nearest neighbor filtering.
    Nearest,
    /// Linear interpolation filtering.
    Linear,
}

/// Filter mode for mipmap level selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MipmapFilterMode {
    /// Nearest mipmap level.
    Nearest,
    /// Linear blend between adjacent mipmap levels.
    Linear,
}

/// Texture coordinate addressing/wrapping mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressMode {
    /// Clamp texture coordinates to edge pixels.
    ClampToEdge,
    /// Repeat texture coordinates periodically.
    Repeat,
    /// Mirror and repeat texture coordinates periodically.
    MirrorRepeat,
    /// Clamp to a fixed border color.
    ClampToBorder,
}

/// Comparison function for depth/stencil texture sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareFunction {
    /// Never pass comparison.
    Never,
    /// Pass if sampled value is less than reference.
    Less,
    /// Pass if sampled value equals reference.
    Equal,
    /// Pass if sampled value is less than or equal to reference.
    LessEqual,
    /// Pass if sampled value is greater than reference.
    Greater,
    /// Pass if sampled value is not equal to reference.
    NotEqual,
    /// Pass if sampled value is greater than or equal to reference.
    GreaterEqual,
    /// Always pass comparison.
    Always,
}

/// Border color used with `AddressMode::ClampToBorder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BorderColor {
    /// Transparent black (0.0, 0.0, 0.0, 0.0).
    TransparentBlack,
    /// Opaque black (0.0, 0.0, 0.0, 1.0).
    OpaqueBlack,
    /// Opaque white (1.0, 1.0, 1.0, 1.0).
    OpaqueWhite,
}

/// Domain-neutral sampler configuration descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplerDescriptor {
    /// Minification filter mode.
    pub filter_min: FilterMode,
    /// Magnification filter mode.
    pub filter_mag: FilterMode,
    /// Mipmap level filter mode.
    pub mipmap_filter: MipmapFilterMode,
    /// U coordinate address mode.
    pub address_mode_u: AddressMode,
    /// V coordinate address mode.
    pub address_mode_v: AddressMode,
    /// W coordinate address mode.
    pub address_mode_w: AddressMode,
    /// Optional depth comparison function.
    pub compare: Option<CompareFunction>,
    /// Minimum level of detail clamp.
    pub lod_min_bits: u32,
    /// Maximum level of detail clamp.
    pub lod_max_bits: u32,
    /// Maximum anisotropic filtering sample count (1 = disabled).
    pub max_anisotropy: u16,
    /// Border color for clamp-to-border addressing.
    pub border_color: BorderColor,
}

impl SamplerDescriptor {
    /// Standard linear clamp-to-edge sampler.
    #[must_use]
    pub const fn linear_clamp() -> Self {
        Self {
            filter_min: FilterMode::Linear,
            filter_mag: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            compare: None,
            lod_min_bits: 0,
            lod_max_bits: 0x41f0_0000, // 30.0f32
            max_anisotropy: 1,
            border_color: BorderColor::TransparentBlack,
        }
    }

    /// Standard nearest clamp-to-edge sampler.
    #[must_use]
    pub const fn nearest_clamp() -> Self {
        Self {
            filter_min: FilterMode::Nearest,
            filter_mag: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            compare: None,
            lod_min_bits: 0,
            lod_max_bits: 0x41f0_0000, // 30.0f32
            max_anisotropy: 1,
            border_color: BorderColor::TransparentBlack,
        }
    }
}

/// Sampler capability descriptor reported by a device/backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplerCapability {
    /// Maximum supported anisotropy level.
    pub max_anisotropy: u16,
    /// Whether comparison samplers are supported.
    pub supports_compare: bool,
    /// Whether clamp-to-border address mode is supported.
    pub supports_clamp_to_border: bool,
    /// Whether custom border colors are supported.
    pub supports_custom_border_color: bool,
}

/// External memory handle kinds for zero-copy OS/graphics interop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalMemoryKind {
    /// Linux DMA-BUF file descriptor.
    DmaBuf,
    /// Windows NT handle for shared memory.
    Win32Nt,
    /// Windows KMT handle for shared memory.
    Win32Kmt,
    /// Apple Metal shared texture / buffer allocation.
    MetalSharedResource,
    /// Host pinned virtual address memory.
    HostAllocation,
    /// POSIX opaque file descriptor.
    OpaqueFd,
}

/// External memory capability descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalMemoryCapability {
    /// External memory kind.
    pub kind: ExternalMemoryKind,
    /// Whether this backend supports importing this memory kind.
    pub can_import: bool,
    /// Whether this backend supports exporting this memory kind.
    pub can_export: bool,
    /// Whether dedicated allocation is required for import/export.
    pub requires_dedicated_allocation: bool,
    /// Hardware alignment requirement in bytes.
    pub alignment_bytes: u64,
}

/// Domain-neutral external synchronization event kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEventKind {
    /// Monotonically increasing timeline semaphore.
    TimelineSemaphore,
    /// Binary GPU fence.
    BinaryFence,
    /// Metal shared event.
    MetalSharedEvent,
    /// POSIX sync file descriptor.
    SyncFileFd,
    /// Direct in-order queue synchronization.
    ImplicitQueue,
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

impl TimelineSyncProtocol {
    /// Return the event kind of this protocol instance.
    #[must_use]
    pub const fn event_kind(&self) -> ExternalEventKind {
        match self {
            Self::TimelineSemaphore { .. } => ExternalEventKind::TimelineSemaphore,
            Self::Fence { .. } => ExternalEventKind::BinaryFence,
            Self::MetalSharedEvent { .. } => ExternalEventKind::MetalSharedEvent,
            Self::SyncFileFd { .. } => ExternalEventKind::SyncFileFd,
            Self::ImplicitQueue => ExternalEventKind::ImplicitQueue,
        }
    }

    /// Whether this protocol is a monotonically increasing timeline point.
    #[must_use]
    pub const fn is_timeline(&self) -> bool {
        matches!(
            self,
            Self::TimelineSemaphore { .. } | Self::MetalSharedEvent { .. }
        )
    }
}

/// External event capability descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalEventCapability {
    /// Kind of external synchronization primitive.
    pub kind: ExternalEventKind,
    /// Whether wait operations are supported.
    pub can_wait: bool,
    /// Whether signal operations are supported.
    pub can_signal: bool,
    /// Whether cross-process export is supported.
    pub can_export: bool,
    /// Whether this event kind supports timeline values.
    pub is_timeline: bool,
}
