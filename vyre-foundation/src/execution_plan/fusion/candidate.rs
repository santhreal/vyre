//! Fusion candidate kinds, candidate sets, and candidate legality evaluation.
//!
//! Every fusion strategy is an explicit, named candidate kind. Illegality is
//! rejected by name before ranking, never priced into a cost, and the unfused
//! baseline is preserved in the candidate set for every domain.

use serde::{Deserialize, Serialize};

use super::dependence::HandoffLocation;
use super::legality::FusionRejectionReason;
use super::region::{RegionFusionPlanner, RegionRelation};
use super::tile::ScheduleTile;
use crate::logical::LogicalRegion;

/// Explicit classification of a fusion candidate strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FusionCandidateKind {
    /// Unfused baseline preserving isolated dispatches for every domain.
    UnfusedBaseline,
    /// Pointwise elementwise forwarding in registers with 0 intermediate buffer writes.
    RegisterForwarding,
    /// Workgroup-level shared memory tile forwarding with a tile barrier.
    SharedMemoryForwarding,
    /// Pre-processing or layout transformation fused into the consumer head.
    PrologueFusion,
    /// Post-processing, activation, or bias addition fused into the producer tail.
    EpilogueFusion,
    /// Multi-stage software-pipelined tiles overlapping compute and load stages.
    PipelinedTiles,
    /// Partial fusion combining a compatible subset while leaving remainder separate.
    PartialFusion,
    /// Explicit dispatch cut where device-wide cross-workgroup ordering is required.
    ExplicitDispatchCut,
}

impl FusionCandidateKind {
    /// Exhaustive roster of all fusion candidate kinds derived from source.
    pub const ALL: &'static [Self] = &[
        Self::UnfusedBaseline,
        Self::RegisterForwarding,
        Self::SharedMemoryForwarding,
        Self::PrologueFusion,
        Self::EpilogueFusion,
        Self::PipelinedTiles,
        Self::PartialFusion,
        Self::ExplicitDispatchCut,
    ];

    /// Human-readable candidate name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UnfusedBaseline => "unfused_baseline",
            Self::RegisterForwarding => "register_forwarding",
            Self::SharedMemoryForwarding => "shared_memory_forwarding",
            Self::PrologueFusion => "prologue_fusion",
            Self::EpilogueFusion => "epilogue_fusion",
            Self::PipelinedTiles => "pipelined_tiles",
            Self::PartialFusion => "partial_fusion",
            Self::ExplicitDispatchCut => "explicit_dispatch_cut",
        }
    }

    /// Machine-readable stable diagnostic code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnfusedBaseline => "FCK001_UNFUSED_BASELINE",
            Self::RegisterForwarding => "FCK002_REGISTER_FORWARDING",
            Self::SharedMemoryForwarding => "FCK003_SHARED_MEMORY_FORWARDING",
            Self::PrologueFusion => "FCK004_PROLOGUE_FUSION",
            Self::EpilogueFusion => "FCK005_EPILOGUE_FUSION",
            Self::PipelinedTiles => "FCK006_PIPELINED_TILES",
            Self::PartialFusion => "FCK007_PARTIAL_FUSION",
            Self::ExplicitDispatchCut => "FCK008_EXPLICIT_DISPATCH_CUT",
        }
    }

    /// Whether this candidate represents the unfused baseline.
    #[must_use]
    pub const fn is_baseline(self) -> bool {
        matches!(self, Self::UnfusedBaseline)
    }

    /// Evaluate legality for this candidate kind over a producer and consumer region.
    ///
    /// # Errors
    ///
    /// Returns [`FusionRejectionReason`] when the proposed candidate kind cannot
    /// legally satisfy the dataflow or synchronization contracts of the regions.
    pub fn evaluate_legality(
        self,
        producer: &LogicalRegion,
        consumer: &LogicalRegion,
    ) -> Result<(), FusionRejectionReason> {
        let relation = RegionFusionPlanner::classify_relation(producer, consumer);
        match self {
            Self::UnfusedBaseline => Ok(()),
            Self::RegisterForwarding => {
                if relation != RegionRelation::PointwiseMatch {
                    return Err(FusionRejectionReason::IncompatibleIterationSpace);
                }
                if producer.effects.synchronizes || consumer.effects.synchronizes {
                    return Err(FusionRejectionReason::SynchronizationBoundary);
                }
                Ok(())
            }
            Self::SharedMemoryForwarding => {
                if relation == RegionRelation::Incompatible {
                    return Err(FusionRejectionReason::IncompatibleIterationSpace);
                }
                Ok(())
            }
            Self::PrologueFusion => {
                if producer.effects.retained_state {
                    return Err(FusionRejectionReason::LifecycleBoundary);
                }
                Ok(())
            }
            Self::EpilogueFusion => {
                if consumer.effects.retained_state {
                    return Err(FusionRejectionReason::LifecycleBoundary);
                }
                Ok(())
            }
            Self::PipelinedTiles => {
                if relation == RegionRelation::Incompatible {
                    return Err(FusionRejectionReason::IncompatibleIterationSpace);
                }
                Ok(())
            }
            Self::PartialFusion => Ok(()),
            Self::ExplicitDispatchCut => Ok(()),
        }
    }
}

/// One concrete fusion candidate for a pair or set of regions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionCandidate {
    /// Strategy kind.
    pub kind: FusionCandidateKind,
    /// Earliest legal handoff location.
    pub handoff: HandoffLocation,
    /// Schedule tile configuration, if tiled.
    pub tile: Option<ScheduleTile>,
    /// Indices of participating arms or regions.
    pub regions: Vec<usize>,
}

impl FusionCandidate {
    /// Construct an unfused baseline candidate.
    #[must_use]
    pub fn baseline(regions: Vec<usize>) -> Self {
        Self {
            kind: FusionCandidateKind::UnfusedBaseline,
            handoff: HandoffLocation::DispatchCut,
            tile: None,
            regions,
        }
    }

    /// Construct a register forwarding candidate.
    #[must_use]
    pub fn register_forwarding(producer: usize, consumer: usize) -> Self {
        Self {
            kind: FusionCandidateKind::RegisterForwarding,
            handoff: HandoffLocation::Register,
            tile: None,
            regions: vec![producer, consumer],
        }
    }

    /// Construct a shared-memory forwarding candidate.
    #[must_use]
    pub fn shared_forwarding(producer: usize, consumer: usize, tile: ScheduleTile) -> Self {
        Self {
            kind: FusionCandidateKind::SharedMemoryForwarding,
            handoff: HandoffLocation::WorkgroupShared,
            tile: Some(tile),
            regions: vec![producer, consumer],
        }
    }
}

/// Set of fusion candidates generated for a domain or graph, guaranteeing baseline presence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionCandidateSet {
    /// All derived candidate plans, baseline first.
    pub candidates: Vec<FusionCandidate>,
}

impl FusionCandidateSet {
    /// Construct a candidate set ensuring the unfused baseline is always present.
    #[must_use]
    pub fn new(region_count: usize) -> Self {
        let indices = (0..region_count).collect();
        Self {
            candidates: vec![FusionCandidate::baseline(indices)],
        }
    }

    /// Add a legal candidate to the set.
    pub fn add(&mut self, candidate: FusionCandidate) {
        if !self.candidates.contains(&candidate) {
            self.candidates.push(candidate);
        }
    }

    /// Return whether the unfused baseline is present.
    #[must_use]
    pub fn has_baseline(&self) -> bool {
        self.candidates.iter().any(|c| c.kind.is_baseline())
    }
}
