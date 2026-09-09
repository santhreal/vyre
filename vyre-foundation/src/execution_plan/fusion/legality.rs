//! Legality classification and rejection reasons for region-level fusion.
//!
//! Fusion candidates that cannot legally satisfy dataflow, dependency, or
//! synchronization invariants are rejected by name with stable diagnostic codes.

use serde::{Deserialize, Serialize};

use crate::ir::Program;
use crate::logical::LogicalRegion;
use super::dependence::{classify_program_handoff, HandoffLocation};
use super::region::{RegionFusionPlanner, RegionRelation};

/// Stable machine-readable reason that prevents two regions or programs from fusing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FusionRejectionReason {
    /// A referenced node or value is absent from the graph.
    UnknownGraphMember,
    /// The proposed value does not connect the producer and consumer.
    NotProducerConsumer,
    /// The value crosses a retained-state or host-lifecycle boundary.
    LifecycleBoundary,
    /// The value is consumed by more than one downstream node.
    MultipleConsumers,
    /// The programs declare conflicting or non-unifiable workgroup geometry.
    WorkgroupMismatch,
    /// The programs declare differing workgroup geometry and one pins its geometry.
    SynchronizationBoundary,
    /// Contracting the proposed group would create a dependency cycle.
    DependencyCycle,
    /// Iteration spaces have incompatible ranks, extents, or tiling shapes.
    IncompatibleIterationSpace,
    /// Required workgroup shared memory exceeds target or adapter budget.
    ExcessiveSharedMemory,
    /// Register pressure of the fused kernel exceeds hardware bounds.
    ExcessiveRegisters,
    /// The geometry semantics or invocation guards explicitly conflict.
    GenuinelyIllegalGeometry,
}

impl FusionRejectionReason {
    /// Machine-readable stable diagnostic code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnknownGraphMember => "MKL001_UNKNOWN_GRAPH_MEMBER",
            Self::NotProducerConsumer => "MKL002_NOT_PRODUCER_CONSUMER",
            Self::LifecycleBoundary => "MKL003_LIFECYCLE_BOUNDARY",
            Self::MultipleConsumers => "MKL004_MULTIPLE_CONSUMERS",
            Self::WorkgroupMismatch => "MKL005_WORKGROUP_MISMATCH",
            Self::SynchronizationBoundary => "MKL006_SYNCHRONIZATION_BOUNDARY",
            Self::DependencyCycle => "MKL007_DEPENDENCY_CYCLE",
            Self::IncompatibleIterationSpace => "MKL008_INCOMPATIBLE_ITERATION_SPACE",
            Self::ExcessiveSharedMemory => "MKL009_EXCESSIVE_SHARED_MEMORY",
            Self::ExcessiveRegisters => "MKL010_EXCESSIVE_REGISTERS",
            Self::GenuinelyIllegalGeometry => "MKL011_GENUINELY_ILLEGAL_GEOMETRY",
        }
    }
}

impl std::fmt::Display for FusionRejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.code(), match self {
            Self::UnknownGraphMember => "unknown graph member",
            Self::NotProducerConsumer => "not producer-consumer connected",
            Self::LifecycleBoundary => "crosses lifecycle boundary",
            Self::MultipleConsumers => "multiple consumers for intermediate value",
            Self::WorkgroupMismatch => "workgroup mismatch",
            Self::SynchronizationBoundary => "synchronization boundary",
            Self::DependencyCycle => "dependency cycle",
            Self::IncompatibleIterationSpace => "incompatible iteration space",
            Self::ExcessiveSharedMemory => "excessive shared memory",
            Self::ExcessiveRegisters => "excessive registers",
            Self::GenuinelyIllegalGeometry => "genuinely illegal geometry",
        })
    }
}

/// Legality verdict for a proposed fusion decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionLegalityVerdict {
    /// Fusion is legal with the identified earliest safe handoff location.
    Legal(HandoffLocation),
    /// Fusion is rejected for the stable named reason.
    Rejected(FusionRejectionReason),
}

impl FusionLegalityVerdict {
    /// Return whether this verdict is legal.
    #[must_use]
    pub const fn is_legal(&self) -> bool {
        matches!(self, Self::Legal(_))
    }

    /// Convert to a Result<(), FusionRejectionReason>.
    pub fn into_result(self) -> Result<HandoffLocation, FusionRejectionReason> {
        match self {
            Self::Legal(handoff) => Ok(handoff),
            Self::Rejected(reason) => Err(reason),
        }
    }
}

/// Analyze legality of fusing two logical regions.
#[must_use]
pub fn analyze_region_fusion_legality(
    producer: &LogicalRegion,
    consumer: &LogicalRegion,
) -> FusionLegalityVerdict {
    let relation = RegionFusionPlanner::classify_relation(producer, consumer);
    match relation {
        RegionRelation::PointwiseMatch => FusionLegalityVerdict::Legal(HandoffLocation::Register),
        RegionRelation::ReductionConsumer | RegionRelation::StencilWindow => {
            FusionLegalityVerdict::Legal(HandoffLocation::WorkgroupShared)
        }
        RegionRelation::TileCompatible => {
            FusionLegalityVerdict::Legal(HandoffLocation::PipelinedStage(2))
        }
        RegionRelation::IndependentFrontier => {
            FusionLegalityVerdict::Legal(HandoffLocation::Independent)
        }
        RegionRelation::Incompatible => {
            FusionLegalityVerdict::Rejected(FusionRejectionReason::IncompatibleIterationSpace)
        }
    }
}

/// Analyze legality of fusing two Programs under the region/tile fusion model.
#[must_use]
pub fn analyze_program_fusion_legality(
    producer: &Program,
    consumer: &Program,
) -> FusionLegalityVerdict {
    if producer.is_non_composable_with_self() && producer.entry_op_id() == consumer.entry_op_id() {
        return FusionLegalityVerdict::Rejected(FusionRejectionReason::GenuinelyIllegalGeometry);
    }

    let prod_wg = producer.workgroup_size();
    let cons_wg = consumer.workgroup_size();

    let prod_sched_only = producer.workgroup_size_is_schedule_only();
    let cons_sched_only = consumer.workgroup_size_is_schedule_only();

    if prod_wg != cons_wg {
        if !prod_sched_only || !cons_sched_only {
            if producer.stats().has_node_barrier() || consumer.stats().has_node_barrier() {
                return FusionLegalityVerdict::Rejected(FusionRejectionReason::SynchronizationBoundary);
            }
            return FusionLegalityVerdict::Rejected(FusionRejectionReason::WorkgroupMismatch);
        }
    }

    let handoff = classify_program_handoff(producer, consumer);
    FusionLegalityVerdict::Legal(handoff)
}
