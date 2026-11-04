//! Closed collective communication group and topology representations.

use super::scope::ExecutionScope;

/// Closed collective communication group and topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum CollectiveGroup {
    /// Subgroup collective: shuffle or vote across SIMD lanes.
    Subgroup,
    /// Workgroup collective (shared memory reduction / scan).
    Workgroup,
    /// Cluster collective across cooperating workgroups.
    Cluster,
    /// Device mesh collective across grid partition.
    DeviceMesh,
    /// Cross-device ring topology collective.
    CrossDeviceRing,
    /// Custom topology collective with explicit participant mapping.
    CustomTopology,
}

impl CollectiveGroup {
    /// Every collective group variant in the closed contract.
    pub const ALL: [Self; 6] = [
        Self::Subgroup,
        Self::Workgroup,
        Self::Cluster,
        Self::DeviceMesh,
        Self::CrossDeviceRing,
        Self::CustomTopology,
    ];

    /// Stable wire tag for collective group.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Subgroup => 0,
            Self::Workgroup => 1,
            Self::Cluster => 2,
            Self::DeviceMesh => 3,
            Self::CrossDeviceRing => 4,
            Self::CustomTopology => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `CollectiveGroup`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Subgroup),
            1 => Ok(Self::Workgroup),
            2 => Ok(Self::Cluster),
            3 => Ok(Self::DeviceMesh),
            4 => Ok(Self::CrossDeviceRing),
            5 => Ok(Self::CustomTopology),
            other => Err(format!(
                "InvalidDiscriminant: collective group tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for collective group.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::DeviceMesh => "device_mesh",
            Self::CrossDeviceRing => "cross_device_ring",
            Self::CustomTopology => "custom_topology",
        }
    }

    /// Whether this collective group spans across physical devices.
    #[must_use]
    #[inline]
    pub const fn is_inter_device(self) -> bool {
        match self {
            Self::CrossDeviceRing => true,
            Self::Subgroup
            | Self::Workgroup
            | Self::Cluster
            | Self::DeviceMesh
            | Self::CustomTopology => false,
        }
    }

    /// Whether this collective group is restricted to single-subgroup SIMD lanes.
    #[must_use]
    #[inline]
    pub const fn is_subgroup_only(self) -> bool {
        matches!(self, Self::Subgroup)
    }

    /// Associated minimum execution scope required for this collective group.
    #[must_use]
    pub const fn execution_scope(self) -> ExecutionScope {
        match self {
            Self::Subgroup => ExecutionScope::Subgroup,
            Self::Workgroup => ExecutionScope::Workgroup,
            Self::Cluster => ExecutionScope::Cluster,
            Self::DeviceMesh | Self::CustomTopology => ExecutionScope::Grid,
            Self::CrossDeviceRing => ExecutionScope::DeviceMesh,
        }
    }
}

/// Exhaustive compile-time check for [`CollectiveGroup`].
#[must_use]
pub const fn exhaustiveness_check_collective_group(val: CollectiveGroup) -> &'static str {
    match val {
        CollectiveGroup::Subgroup => "Subgroup",
        CollectiveGroup::Workgroup => "Workgroup",
        CollectiveGroup::Cluster => "Cluster",
        CollectiveGroup::DeviceMesh => "DeviceMesh",
        CollectiveGroup::CrossDeviceRing => "CrossDeviceRing",
        CollectiveGroup::CustomTopology => "CustomTopology",
    }
}
