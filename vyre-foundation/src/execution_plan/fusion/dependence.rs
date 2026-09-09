//! Dependence analysis and earliest legal producer-consumer handoff identification.
//!
//! A barrier is placed because a dataflow dependence requires synchronization,
//! not merely because two buffer declarations overlap. This module computes the
//! exact dependence DAG and identifies the earliest legal handoff point
//! (register, workgroup shared memory, pipelined stage, or dispatch cut).

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::region::{RegionFusionPlanner, RegionRelation};
use crate::ir::{BufferAccess, Program};
use crate::logical::LogicalRegion;

/// Earliest legal handoff location between a producer and a consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HandoffLocation {
    /// Direct register forwarding within the same thread invocation (0 barriers).
    Register,
    /// Workgroup shared-memory tile forwarding with a workgroup barrier at the tile boundary.
    WorkgroupShared,
    /// Software-pipelined tile handoff overlapped across ring stages.
    PipelinedStage(u32),
    /// Device-wide ordering cut requiring a grid-level barrier or separate dispatch.
    DispatchCut,
    /// Independent frontiers with no dataflow dependency (0 barriers).
    Independent,
}

impl HandoffLocation {
    /// Whether this handoff location requires an explicit workgroup-level barrier.
    #[must_use]
    pub const fn requires_workgroup_barrier(self) -> bool {
        matches!(self, Self::WorkgroupShared)
    }

    /// Whether this handoff location requires a device-wide grid sync or dispatch cut.
    #[must_use]
    pub const fn requires_dispatch_cut(self) -> bool {
        matches!(self, Self::DispatchCut)
    }

    /// Whether this handoff executes purely in registers with zero memory barriers.
    #[must_use]
    pub const fn is_register_or_independent(self) -> bool {
        matches!(self, Self::Register | Self::Independent)
    }
}

/// Data dependence edge between two logical regions or programs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionDependence {
    /// Source (producer) arm index.
    pub producer: usize,
    /// Destination (consumer) arm index.
    pub consumer: usize,
    /// Name of the carrier buffer or value connecting them.
    pub carrier: String,
    /// Earliest legal handoff location.
    pub handoff: HandoffLocation,
}

/// Graph of dataflow and effect dependencies across regions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionDependenceGraph {
    /// All directed dependence edges.
    pub edges: Vec<RegionDependence>,
    /// Set of arm pairs that are mutually independent (independent frontiers).
    pub independent_frontiers: FxHashSet<(usize, usize)>,
}

impl RegionDependenceGraph {
    /// Analyze dependencies between a list of programs and find earliest legal handoffs.
    #[must_use]
    pub fn from_programs(programs: &[Program]) -> Self {
        let mut edges = Vec::new();
        let mut write_arms_per_buf: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut read_arms_per_buf: FxHashMap<String, Vec<usize>> = FxHashMap::default();

        for (arm_idx, prog) in programs.iter().enumerate() {
            for buf in prog.buffers() {
                let name = buf.name().to_string();
                match buf.access() {
                    BufferAccess::ReadOnly | BufferAccess::Uniform => {
                        read_arms_per_buf
                            .entry(name.clone())
                            .or_default()
                            .push(arm_idx);
                        if let Some(writers) = write_arms_per_buf.get(&name) {
                            for &w in writers {
                                let handoff = classify_program_handoff(&programs[w], prog);
                                edges.push(RegionDependence {
                                    producer: w,
                                    consumer: arm_idx,
                                    carrier: name.clone(),
                                    handoff,
                                });
                            }
                        }
                    }
                    BufferAccess::ReadWrite | BufferAccess::WriteOnly => {
                        write_arms_per_buf
                            .entry(name.clone())
                            .or_default()
                            .push(arm_idx);
                        if let Some(readers) = read_arms_per_buf.get(&name) {
                            for &r in readers {
                                let handoff = if prog.stats().has_node_barrier() {
                                    HandoffLocation::WorkgroupShared
                                } else {
                                    HandoffLocation::Register
                                };
                                edges.push(RegionDependence {
                                    producer: r,
                                    consumer: arm_idx,
                                    carrier: name.clone(),
                                    handoff,
                                });
                            }
                        }
                    }
                    BufferAccess::Workgroup | _ => {}
                }
            }
        }

        let mut independent_frontiers = FxHashSet::default();
        for i in 0..programs.len() {
            for j in (i + 1)..programs.len() {
                let has_edge = edges.iter().any(|e| {
                    (e.producer == i && e.consumer == j) || (e.producer == j && e.consumer == i)
                });
                if !has_edge {
                    independent_frontiers.insert((i, j));
                }
            }
        }

        Self {
            edges,
            independent_frontiers,
        }
    }

    /// Return the earliest handoff location between `producer` and `consumer`, if an edge exists.
    #[must_use]
    pub fn earliest_handoff_between(
        &self,
        producer: usize,
        consumer: usize,
    ) -> Option<HandoffLocation> {
        self.edges
            .iter()
            .filter(|e| e.producer == producer && e.consumer == consumer)
            .map(|e| e.handoff)
            .min()
    }
}

/// Helper to classify the handoff location between two programs based on access patterns and geometry.
#[must_use]
pub fn classify_program_handoff(producer: &Program, consumer: &Program) -> HandoffLocation {
    let prod_wg = producer.workgroup_size();
    let cons_wg = consumer.workgroup_size();

    let prod_has_barrier = producer.stats().has_node_barrier();
    let cons_has_barrier = consumer.stats().has_node_barrier();
    let uses_shared = producer
        .buffers()
        .iter()
        .any(|b| b.access() == BufferAccess::Workgroup)
        || consumer
            .buffers()
            .iter()
            .any(|b| b.access() == BufferAccess::Workgroup);

    // If both are 1:1 schedule-only elementwise programs without barriers/shared memory:
    // the earliest legal handoff is REGISTER forwarding (0 barriers).
    if producer.workgroup_size_is_schedule_only()
        && consumer.workgroup_size_is_schedule_only()
        && !uses_shared
        && !prod_has_barrier
        && !cons_has_barrier
    {
        return HandoffLocation::Register;
    }

    // If they share workgroup scope and use intra-workgroup barriers or shared memory:
    if prod_wg == cons_wg && (uses_shared || prod_has_barrier || cons_has_barrier) {
        return HandoffLocation::WorkgroupShared;
    }

    // If there is a cross-workgroup launch-dependent write, an explicit dispatch cut is required.
    if !producer.workgroup_size_is_schedule_only()
        && !consumer.workgroup_size_is_schedule_only()
        && prod_wg != cons_wg
    {
        return HandoffLocation::DispatchCut;
    }

    HandoffLocation::Register
}

/// Analyze earliest legal handoff between two [`LogicalRegion`]s.
#[must_use]
pub fn earliest_region_handoff(
    producer: &LogicalRegion,
    consumer: &LogicalRegion,
) -> HandoffLocation {
    let relation = RegionFusionPlanner::classify_relation(producer, consumer);
    match relation {
        RegionRelation::PointwiseMatch => HandoffLocation::Register,
        RegionRelation::ReductionConsumer | RegionRelation::StencilWindow => {
            HandoffLocation::WorkgroupShared
        }
        RegionRelation::TileCompatible => HandoffLocation::PipelinedStage(2),
        RegionRelation::IndependentFrontier => HandoffLocation::Independent,
        RegionRelation::Incompatible => HandoffLocation::DispatchCut,
    }
}
