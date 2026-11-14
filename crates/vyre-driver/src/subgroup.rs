//! Backend-neutral subgroup operation taxonomy.

/// Subgroup capability record shared by validation and optimizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubgroupCaps {
    /// Native subgroup operations are available for compute.
    pub supports_subgroup: bool,
    /// Subgroup operations are available in vertex-stage contexts.
    pub supports_subgroup_vertex: bool,
    /// Subgroup size in lanes; `0` means unknown.
    pub subgroup_size: u32,
}

impl SubgroupCaps {
    /// Capability record for native subgroup intrinsics.
    #[must_use]
    pub const fn native(subgroup_size: u32) -> Self {
        Self {
            supports_subgroup: true,
            supports_subgroup_vertex: false,
            subgroup_size,
        }
    }

    /// Capability record from a feature bit and reported lane-size range.
    #[must_use]
    pub const fn from_feature_range(
        supports_feature: bool,
        supports_vertex_stage: bool,
        min_size: u32,
        max_size: u32,
    ) -> Self {
        let supports_subgroup = supports_feature && min_size > 0 && max_size >= min_size;
        Self {
            supports_subgroup,
            supports_subgroup_vertex: supports_vertex_stage && supports_subgroup,
            subgroup_size: if supports_subgroup { min_size } else { 0 },
        }
    }

    /// Return true when native subgroup operations are usable.
    #[must_use]
    pub const fn is_usable(self) -> bool {
        self.supports_subgroup && self.subgroup_size > 0
    }
}
