use serde::{Deserialize, Serialize};
use vyre_foundation::ir::{ProgramGraph, ValueLifetime};

use crate::{ArtifactNodeId, ArtifactValueId};

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
    /// The programs require incompatible workgroup geometry.
    WorkgroupMismatch,
    /// One of the programs contains an explicit synchronization point.
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
    if producer.program.workgroup_size != consumer.program.workgroup_size {
        return FusionDecision::Rejected(FusionRejectionReason::WorkgroupMismatch);
    }
    if producer.program.stats().has_node_barrier() || consumer.program.stats().has_node_barrier() {
        return FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary);
    }
    FusionDecision::Legal
}
