//! Stable legality failures a schedule transform can report.
//!
//! Held apart from the schema so a reader auditing what a transform proves
//! reads only the proof machinery, and a caller matching on a failure reads
//! only the failure set.

use thiserror::Error;

use super::{MappingLevel, ScheduleAxis, SchedulePhaseId};

/// Stable legality failure for a backend-neutral schedule transform.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ScheduleLegalityError {
    /// The persisted schema version is unsupported.
    #[error("schedule schema {found} is unsupported; expected {expected}")]
    UnsupportedVersion {
        /// Persisted version.
        found: u16,
        /// Current version.
        expected: u16,
    },
    /// A required collection is empty.
    #[error("schedule requires a nonempty {0}")]
    Empty(&'static str),
    /// A required dimension or bound is zero.
    #[error("schedule requires a nonzero {0}")]
    Zero(&'static str),
    /// A selected phase does not exist.
    #[error("schedule phase {0:?} does not exist")]
    MissingPhase(SchedulePhaseId),
    /// A logical region does not exist in the selected phase.
    #[error("schedule region {0} does not exist in the selected phase")]
    MissingRegion(u32),
    /// A selected axis does not exist in the selected phase.
    #[error("schedule axis {axis:?} does not exist in phase {phase:?}")]
    MissingAxis {
        /// Referenced phase.
        phase: SchedulePhaseId,
        /// Referenced axis.
        axis: ScheduleAxis,
    },
    /// A split or vector width does not divide its source extent.
    #[error("schedule factor {factor} does not divide extent {extent}")]
    NonDivisible {
        /// Source extent.
        extent: u64,
        /// Requested factor.
        factor: u32,
    },
    /// A phase identity occurs more than once.
    #[error("schedule phase {0:?} occurs more than once")]
    DuplicatePhase(SchedulePhaseId),
    /// A logical region occurs in more than one selected phase.
    #[error("schedule region {0} occurs in more than one phase")]
    DuplicateRegion(u32),
    /// A transform repeats one of its phase operands.
    #[error("schedule transform phase operands must be distinct")]
    DuplicateTransformPhase,
    /// Phase fission would leave an empty phase.
    #[error("schedule fission of phase {0:?} would leave an empty phase")]
    InvalidFission(SchedulePhaseId),
    /// Reorder is not a complete axis permutation.
    #[error("schedule reorder is not a permutation of phase {0:?}")]
    InvalidPermutation(SchedulePhaseId),
    /// A pipeline does not contain positive producer and consumer roles.
    #[error("schedule pipeline requires nonzero producer and consumer role groups")]
    InvalidPipelineRoles,
    /// Spatial partitioning used an invocation-level mapping.
    #[error("schedule spatial partition cannot use {0:?}")]
    InvalidPartitionLevel(MappingLevel),
    /// A resource bound overflowed its representation.
    #[error("schedule resource bound `{0}` overflowed")]
    ResourceOverflow(&'static str),
    /// Allocating a new phase identity overflowed.
    #[error("schedule phase identity overflowed")]
    PhaseIdOverflow,
    /// A selected dependency introduces a cycle.
    #[error("schedule dependency {from:?} -> {to:?} is cyclic")]
    DependencyCycle {
        /// Source phase.
        from: SchedulePhaseId,
        /// Destination phase.
        to: SchedulePhaseId,
    },
    /// An applied transform lacks typed source or inverse provenance.
    #[error("schedule transform lacks typed source or inverse provenance")]
    MissingProvenance,
    /// A persisted transform's typed proof differs from deterministic replay.
    #[error("schedule transform {0} has invalid precondition or provenance evidence")]
    InvalidTransformProof(usize),
    /// Persisted final phases or resources differ from deterministic replay.
    #[error("schedule final state differs from deterministic transform replay")]
    ReplayMismatch,
    /// Naming the inner axis of a tile or split overflowed the axis index.
    #[error("schedule axis index overflowed in phase {0:?}")]
    AxisIndexOverflow(SchedulePhaseId),
    /// Canonical schedule identity serialization failed.
    #[error("schedule identity encoding failed: {0}")]
    Identity(String),
}
