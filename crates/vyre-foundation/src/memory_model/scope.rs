//! Closed memory and execution synchronization scopes.

/// Closed memory visibility and coherence scope across execution hierarchy.
///
/// Defines the domain of participant threads across which a memory operation
/// is guaranteed to be coherent and visible.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryScope {
    /// Single thread / invocation visibility only.
    Thread,
    /// Subgroup-coherent visibility.
    Subgroup,
    /// Workgroup / threadblock coherent visibility.
    Workgroup,
    /// Cluster of cooperating workgroups.
    Cluster,
    /// Full device (GPU / accelerator) global memory coherence.
    Device,
    /// Cross-device / host-heterogeneous system coherence.
    System,
}

impl MemoryScope {
    /// Every memory scope variant in the closed contract.
    pub const ALL: [Self; 6] = [
        Self::Thread,
        Self::Subgroup,
        Self::Workgroup,
        Self::Cluster,
        Self::Device,
        Self::System,
    ];

    /// Stable wire tag for memory scope.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Thread => 0,
            Self::Subgroup => 1,
            Self::Workgroup => 2,
            Self::Cluster => 3,
            Self::Device => 4,
            Self::System => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `MemoryScope`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Thread),
            1 => Ok(Self::Subgroup),
            2 => Ok(Self::Workgroup),
            3 => Ok(Self::Cluster),
            4 => Ok(Self::Device),
            5 => Ok(Self::System),
            other => Err(format!(
                "InvalidDiscriminant: memory scope tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this memory scope.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Thread => "thread",
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::Device => "device",
            Self::System => "system",
        }
    }

    /// Whether this scope crosses workgroup boundaries.
    #[must_use]
    #[inline]
    pub const fn is_cross_workgroup(self) -> bool {
        match self {
            Self::Cluster | Self::Device | Self::System => true,
            Self::Thread | Self::Subgroup | Self::Workgroup => false,
        }
    }

    /// Whether this scope encompasses the entire device.
    #[must_use]
    #[inline]
    pub const fn is_device_wide(self) -> bool {
        match self {
            Self::Device | Self::System => true,
            Self::Thread | Self::Subgroup | Self::Workgroup | Self::Cluster => false,
        }
    }

    /// Widen two memory scopes to the minimum scope enclosing both.
    #[must_use]
    pub const fn widen(self, other: Self) -> Self {
        if self.wire_tag() >= other.wire_tag() {
            self
        } else {
            other
        }
    }

    /// Whether this scope includes or encloses `other`.
    #[must_use]
    #[inline]
    pub const fn includes(self, other: Self) -> bool {
        self.wire_tag() >= other.wire_tag()
    }
}

/// Closed execution synchronization scope defining participant domain.
///
/// Defines which execution threads rendezvous or participate in execution barriers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ExecutionScope {
    /// Single thread execution.
    Thread,
    /// Subgroup execution rendezvous.
    Subgroup,
    /// Workgroup / threadblock execution rendezvous.
    Workgroup,
    /// Cluster execution rendezvous across cooperative workgroups.
    Cluster,
    /// Grid / dispatch execution rendezvous across the entire launch.
    Grid,
    /// Distributed device mesh rendezvous across multiple physical devices.
    DeviceMesh,
}

impl ExecutionScope {
    /// Every execution scope variant in the closed contract.
    pub const ALL: [Self; 6] = [
        Self::Thread,
        Self::Subgroup,
        Self::Workgroup,
        Self::Cluster,
        Self::Grid,
        Self::DeviceMesh,
    ];

    /// Stable wire tag for execution scope.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Thread => 0,
            Self::Subgroup => 1,
            Self::Workgroup => 2,
            Self::Cluster => 3,
            Self::Grid => 4,
            Self::DeviceMesh => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `ExecutionScope`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Thread),
            1 => Ok(Self::Subgroup),
            2 => Ok(Self::Workgroup),
            3 => Ok(Self::Cluster),
            4 => Ok(Self::Grid),
            5 => Ok(Self::DeviceMesh),
            other => Err(format!(
                "InvalidDiscriminant: execution scope tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this execution scope.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Thread => "thread",
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::Grid => "grid",
            Self::DeviceMesh => "device_mesh",
        }
    }

    /// Whether this execution scope crosses thread block / workgroup boundaries.
    #[must_use]
    #[inline]
    pub const fn is_cross_block(self) -> bool {
        match self {
            Self::Cluster | Self::Grid | Self::DeviceMesh => true,
            Self::Thread | Self::Subgroup | Self::Workgroup => false,
        }
    }

    /// Whether this execution scope represents a whole-grid or device-mesh rendezvous.
    #[must_use]
    #[inline]
    pub const fn is_grid_or_mesh(self) -> bool {
        match self {
            Self::Grid | Self::DeviceMesh => true,
            Self::Thread | Self::Subgroup | Self::Workgroup | Self::Cluster => false,
        }
    }

    /// Widen two execution scopes to the minimum scope enclosing both.
    #[must_use]
    pub const fn widen(self, other: Self) -> Self {
        if self.wire_tag() >= other.wire_tag() {
            self
        } else {
            other
        }
    }

    /// Whether this scope includes or encloses `other`.
    #[must_use]
    #[inline]
    pub const fn includes(self, other: Self) -> bool {
        self.wire_tag() >= other.wire_tag()
    }
}

/// Exhaustive compile-time check for [`MemoryScope`].
#[must_use]
pub const fn exhaustiveness_check_memory_scope(val: MemoryScope) -> &'static str {
    match val {
        MemoryScope::Thread => "Thread",
        MemoryScope::Subgroup => "Subgroup",
        MemoryScope::Workgroup => "Workgroup",
        MemoryScope::Cluster => "Cluster",
        MemoryScope::Device => "Device",
        MemoryScope::System => "System",
    }
}

/// Exhaustive compile-time check for [`ExecutionScope`].
#[must_use]
pub const fn exhaustiveness_check_execution_scope(val: ExecutionScope) -> &'static str {
    match val {
        ExecutionScope::Thread => "Thread",
        ExecutionScope::Subgroup => "Subgroup",
        ExecutionScope::Workgroup => "Workgroup",
        ExecutionScope::Cluster => "Cluster",
        ExecutionScope::Grid => "Grid",
        ExecutionScope::DeviceMesh => "DeviceMesh",
    }
}
