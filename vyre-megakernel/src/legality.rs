use serde::{Deserialize, Serialize};
use vyre_foundation::ir::{Program, ProgramGraph, ValueLifetime};

use crate::{workgroup_scratch_declarations, ArtifactNodeId, ArtifactValueId};

/// Stable reason that prevents two graph nodes from sharing one generated kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FusionRejectionReason {
    /// A referenced node or value is absent from the graph.
    UnknownGraphMember,
    /// The value does not connect the proposed producer and consumer.
    NotProducerConsumer,
    /// The value crosses an invocation or retained-state boundary.
    LifecycleBoundary,
    /// More than one node consumes the value.
    MultipleConsumers,
    /// The programs declare different workgroup geometry.
    WorkgroupMismatch,
    /// The programs declare different workgroup geometry and one of them
    /// reasons about the size of its own workgroup, so no fused geometry works.
    SynchronizationBoundary,
    /// Contracting the proposed group would create a dependency cycle.
    DependencyCycle,
}

impl FusionRejectionReason {
    /// Stable machine-readable diagnostic code.
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
        }
    }
}

/// Legality result for one proposed producer-consumer fusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionDecision {
    /// Fusion preserves the graph contract.
    Legal,
    /// Fusion is forbidden for the stable reason.
    Rejected(FusionRejectionReason),
}

/// Checks whether one dataflow edge may be internalized into a fused group.
///
/// A barrier does not by itself forbid fusion. `merge_programs_shared`
/// concatenates the arms and inserts a barrier between a writer arm and a
/// later reader arm, and the validator has already proven every barrier
/// workgroup-uniform, so at one shared geometry the fused kernel reaches every
/// barrier from every invocation. What fusion cannot do is rewrite an arm for a
/// different workgroup, which is why the two questions are asked together. The
/// search cannot widen such a group either: `group_workgroup` holds a group at
/// its declared shape unless every member tolerates a proposed width.
///
/// Admitting a barrier at one geometry is what makes a fused attention block
/// expressible: scores written to a workgroup tile, one barrier, then the value
/// pass reading that tile, as a single kernel instead of two dispatches.
#[must_use]
pub fn analyze_fusion_pair(
    graph: &ProgramGraph,
    from: ArtifactNodeId,
    to: ArtifactNodeId,
    value: ArtifactValueId,
) -> FusionDecision {
    let Some(producer) = graph.nodes().get(from.0 as usize) else {
        return FusionDecision::Rejected(FusionRejectionReason::UnknownGraphMember);
    };
    let Some(consumer) = graph.nodes().get(to.0 as usize) else {
        return FusionDecision::Rejected(FusionRejectionReason::UnknownGraphMember);
    };
    let Some(value) = graph.values().get(value.0 as usize) else {
        return FusionDecision::Rejected(FusionRejectionReason::UnknownGraphMember);
    };
    if value.producer.map(|id| id.0) != Some(from.0)
        || !value.consumers.iter().any(|id| id.0 == to.0)
    {
        return FusionDecision::Rejected(FusionRejectionReason::NotProducerConsumer);
    }
    if value.contract.lifetime != ValueLifetime::Invocation {
        return FusionDecision::Rejected(FusionRejectionReason::LifecycleBoundary);
    }
    if value.consumers.len() != 1 {
        return FusionDecision::Rejected(FusionRejectionReason::MultipleConsumers);
    }
    let pinned = pins_workgroup_geometry(&producer.program)
        || pins_workgroup_geometry(&consumer.program);
    if producer.program.workgroup_size != consumer.program.workgroup_size {
        if pinned {
            return FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary);
        }
        return FusionDecision::Rejected(FusionRejectionReason::WorkgroupMismatch);
    }
    FusionDecision::Legal
}

/// Does this program reason about the size of its own workgroup?
///
/// A barrier orders the invocations of one workgroup and a workgroup-scoped
/// buffer is sized for one workgroup, so either one fixes the geometry the
/// program was written for. `reject_workgroup_geometry_change` in
/// `vyre-foundation` owns the same judgement for the merge itself and asks the
/// same two questions; this reads the megakernel's own scratch accessor so the
/// two agree on what a workgroup buffer is.
fn pins_workgroup_geometry(program: &Program) -> bool {
    program.stats().has_node_barrier() || workgroup_scratch_declarations(program).next().is_some()
}
