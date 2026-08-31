//! The compiler levels a type, pass, or record can own.

/// Level of the compiler pipeline a declaration owns.
///
/// Higher levels state semantics, dataflow, logical work, effects, layouts, and
/// constraints. Only the schedule and physical-kernel levels choose tiling,
/// physical ids, launch geometry, memory placement, synchronization,
/// persistence, device partitioning, or target instruction strategy, so a
/// declaration that states its level also states which of those it may touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum IrLevel {
    /// Whole-program graph: connected regions, their dataflow, and identity.
    WholeGraph,
    /// Logical region or algorithm IR: iteration domains, index maps, effects.
    Logical,
    /// Selected schedule IR: tiling, roles, geometry, synchronization.
    Schedule,
    /// Physical kernel IR: register and shared layouts, transactions, phases.
    PhysicalKernel,
    /// Target payload: the emitted module for one backend.
    TargetPayload,
}

impl IrLevel {
    /// Stable name used in reports and generated projections.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::WholeGraph => "whole_graph",
            Self::Logical => "logical",
            Self::Schedule => "schedule",
            Self::PhysicalKernel => "physical_kernel",
            Self::TargetPayload => "target_payload",
        }
    }

    /// Whether this level may state physical execution policy.
    ///
    /// A rewrite at a semantic level that encodes a hardware fact, and a
    /// physical-level rewrite that changes what a program computes, are the two
    /// failure modes the level declaration exists to separate.
    #[must_use]
    pub const fn admits_physical_policy(self) -> bool {
        matches!(
            self,
            Self::Schedule | Self::PhysicalKernel | Self::TargetPayload
        )
    }

    /// Every level, ordered from whole program to target payload.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::WholeGraph,
            Self::Logical,
            Self::Schedule,
            Self::PhysicalKernel,
            Self::TargetPayload,
        ]
    }
}
