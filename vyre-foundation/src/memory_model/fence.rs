//! Closed memory fence synchronization semantics and barrier participation disciplines.

/// Closed memory fence synchronization semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum FenceSemantics {
    /// Acquire fence: orders subsequent reads after prior acquire operations.
    Acquire,
    /// Release fence: orders prior writes before subsequent release operations.
    Release,
    /// Acquire-Release fence: bi-directional memory ordering barrier.
    AcqRel,
    /// Sequentially consistent fence: total order across all participant operations.
    SequentiallyConsistent,
}

impl FenceSemantics {
    /// Every fence semantics variant in the closed contract.
    pub const ALL: [Self; 4] = [
        Self::Acquire,
        Self::Release,
        Self::AcqRel,
        Self::SequentiallyConsistent,
    ];

    /// Stable wire tag for fence semantics.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Acquire => 0,
            Self::Release => 1,
            Self::AcqRel => 2,
            Self::SequentiallyConsistent => 3,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `FenceSemantics`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Acquire),
            1 => Ok(Self::Release),
            2 => Ok(Self::AcqRel),
            3 => Ok(Self::SequentiallyConsistent),
            other => Err(format!(
                "InvalidDiscriminant: fence semantics tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for fence semantics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::SequentiallyConsistent => "sequentially_consistent",
        }
    }

    /// Whether this fence orders subsequent reads.
    #[must_use]
    #[inline]
    pub const fn orders_reads(self) -> bool {
        match self {
            Self::Acquire | Self::AcqRel | Self::SequentiallyConsistent => true,
            Self::Release => false,
        }
    }

    /// Whether this fence orders prior writes.
    #[must_use]
    #[inline]
    pub const fn orders_writes(self) -> bool {
        match self {
            Self::Release | Self::AcqRel | Self::SequentiallyConsistent => true,
            Self::Acquire => false,
        }
    }

    /// Whether this fence is bi-directional (both acquire and release).
    #[must_use]
    #[inline]
    pub const fn is_bidirectional(self) -> bool {
        match self {
            Self::AcqRel | Self::SequentiallyConsistent => true,
            Self::Acquire | Self::Release => false,
        }
    }
}

/// Closed barrier participant discipline and divergence contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum BarrierParticipation {
    /// All participants in the execution scope participate unconditionally.
    Uniform,
    /// Converged control flow: participants currently active at the rendezvous point.
    Converged,
    /// Exactly one elected participant executes; others observe completion.
    ElectOne,
    /// Dynamically masked participants explicitly tracked via active bitmask.
    DynamicMask,
    /// Participation restricted strictly to subgroup scope.
    SubgroupOnly,
    /// Participation restricted strictly to workgroup scope.
    WorkgroupOnly,
}

impl BarrierParticipation {
    /// Every barrier participation variant in the closed contract.
    pub const ALL: [Self; 6] = [
        Self::Uniform,
        Self::Converged,
        Self::ElectOne,
        Self::DynamicMask,
        Self::SubgroupOnly,
        Self::WorkgroupOnly,
    ];

    /// Stable wire tag for barrier participation.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Uniform => 0,
            Self::Converged => 1,
            Self::ElectOne => 2,
            Self::DynamicMask => 3,
            Self::SubgroupOnly => 4,
            Self::WorkgroupOnly => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `BarrierParticipation`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Uniform),
            1 => Ok(Self::Converged),
            2 => Ok(Self::ElectOne),
            3 => Ok(Self::DynamicMask),
            4 => Ok(Self::SubgroupOnly),
            5 => Ok(Self::WorkgroupOnly),
            other => Err(format!(
                "InvalidDiscriminant: barrier participation tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for barrier participation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Converged => "converged",
            Self::ElectOne => "elect_one",
            Self::DynamicMask => "dynamic_mask",
            Self::SubgroupOnly => "subgroup_only",
            Self::WorkgroupOnly => "workgroup_only",
        }
    }

    /// Whether this participation discipline requires uniform control flow.
    #[must_use]
    #[inline]
    pub const fn requires_uniform_control_flow(self) -> bool {
        match self {
            Self::Uniform => true,
            Self::Converged
            | Self::ElectOne
            | Self::DynamicMask
            | Self::SubgroupOnly
            | Self::WorkgroupOnly => false,
        }
    }

    /// Whether this participation discipline permits divergent execution across lanes.
    #[must_use]
    #[inline]
    pub const fn allows_divergence(self) -> bool {
        !self.requires_uniform_control_flow()
    }
}

/// Exhaustive compile-time check for [`FenceSemantics`].
#[must_use]
pub const fn exhaustiveness_check_fence_semantics(val: FenceSemantics) -> &'static str {
    match val {
        FenceSemantics::Acquire => "Acquire",
        FenceSemantics::Release => "Release",
        FenceSemantics::AcqRel => "AcqRel",
        FenceSemantics::SequentiallyConsistent => "SequentiallyConsistent",
    }
}

/// Exhaustive compile-time check for [`BarrierParticipation`].
#[must_use]
pub const fn exhaustiveness_check_barrier_participation(val: BarrierParticipation) -> &'static str {
    match val {
        BarrierParticipation::Uniform => "Uniform",
        BarrierParticipation::Converged => "Converged",
        BarrierParticipation::ElectOne => "ElectOne",
        BarrierParticipation::DynamicMask => "DynamicMask",
        BarrierParticipation::SubgroupOnly => "SubgroupOnly",
        BarrierParticipation::WorkgroupOnly => "WorkgroupOnly",
    }
}
