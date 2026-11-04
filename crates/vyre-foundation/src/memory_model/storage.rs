//! Closed physical and logical storage tier / memory space representations.

use super::scope::MemoryScope;

/// Closed physical and logical storage tier / memory space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum StorageDomain {
    /// Register file / private thread-local storage.
    Register,
    /// Workgroup shared memory / scratchpad (L1/shared SRAM).
    Scratchpad,
    /// Workgroup-local memory tier.
    WorkgroupLocal,
    /// Device global High Bandwidth Memory (HBM/VRAM).
    DeviceGlobal,
    /// Host pinned / zero-copy system memory.
    HostPinned,
    /// Host paged / virtual system memory.
    HostPaged,
    /// Constant / read-only uniform cache storage.
    Constant,
    /// Texture / surface / specialized hardware cache domain.
    Texture,
}

impl StorageDomain {
    /// Every storage domain variant in the closed contract.
    pub const ALL: [Self; 8] = [
        Self::Register,
        Self::Scratchpad,
        Self::WorkgroupLocal,
        Self::DeviceGlobal,
        Self::HostPinned,
        Self::HostPaged,
        Self::Constant,
        Self::Texture,
    ];

    /// Stable wire tag for storage domain.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Register => 0,
            Self::Scratchpad => 1,
            Self::WorkgroupLocal => 2,
            Self::DeviceGlobal => 3,
            Self::HostPinned => 4,
            Self::HostPaged => 5,
            Self::Constant => 6,
            Self::Texture => 7,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `StorageDomain`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Register),
            1 => Ok(Self::Scratchpad),
            2 => Ok(Self::WorkgroupLocal),
            3 => Ok(Self::DeviceGlobal),
            4 => Ok(Self::HostPinned),
            5 => Ok(Self::HostPaged),
            6 => Ok(Self::Constant),
            7 => Ok(Self::Texture),
            other => Err(format!(
                "InvalidDiscriminant: storage domain tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this storage domain.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Register => "register",
            Self::Scratchpad => "scratchpad",
            Self::WorkgroupLocal => "workgroup_local",
            Self::DeviceGlobal => "device_global",
            Self::HostPinned => "host_pinned",
            Self::HostPaged => "host_paged",
            Self::Constant => "constant",
            Self::Texture => "texture",
        }
    }

    /// Whether this storage domain is shared across multiple threads in a workgroup.
    #[must_use]
    #[inline]
    pub const fn is_shared_across_threads(self) -> bool {
        match self {
            Self::Scratchpad
            | Self::WorkgroupLocal
            | Self::DeviceGlobal
            | Self::HostPinned
            | Self::HostPaged
            | Self::Constant
            | Self::Texture => true,
            Self::Register => false,
        }
    }

    /// Whether this storage domain is directly accessible by host CPU.
    #[must_use]
    #[inline]
    pub const fn is_host_accessible(self) -> bool {
        match self {
            Self::HostPinned | Self::HostPaged => true,
            Self::Register
            | Self::Scratchpad
            | Self::WorkgroupLocal
            | Self::DeviceGlobal
            | Self::Constant
            | Self::Texture => false,
        }
    }

    /// Whether this storage domain resides on-chip (registers, scratchpad/L1 SRAM).
    #[must_use]
    #[inline]
    pub const fn is_on_chip(self) -> bool {
        match self {
            Self::Register | Self::Scratchpad | Self::Constant => true,
            Self::WorkgroupLocal
            | Self::DeviceGlobal
            | Self::HostPinned
            | Self::HostPaged
            | Self::Texture => false,
        }
    }

    /// Default memory visibility scope required for coherence in this domain.
    #[must_use]
    pub const fn default_memory_scope(self) -> MemoryScope {
        match self {
            Self::Register => MemoryScope::Thread,
            Self::Scratchpad | Self::WorkgroupLocal => MemoryScope::Workgroup,
            Self::DeviceGlobal | Self::Constant | Self::Texture => MemoryScope::Device,
            Self::HostPinned | Self::HostPaged => MemoryScope::System,
        }
    }
}

/// Exhaustive compile-time check for [`StorageDomain`].
#[must_use]
pub const fn exhaustiveness_check_storage_domain(val: StorageDomain) -> &'static str {
    match val {
        StorageDomain::Register => "Register",
        StorageDomain::Scratchpad => "Scratchpad",
        StorageDomain::WorkgroupLocal => "WorkgroupLocal",
        StorageDomain::DeviceGlobal => "DeviceGlobal",
        StorageDomain::HostPinned => "HostPinned",
        StorageDomain::HostPaged => "HostPaged",
        StorageDomain::Constant => "Constant",
        StorageDomain::Texture => "Texture",
    }
}
