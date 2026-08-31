use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vyre_foundation::{
    logical::LogicalProgramGraph,
    schedule::{
        MappingLevel, ScheduleLegalityError, SchedulePhaseId, ScheduleTransform, SelectedSchedule,
    },
};

use crate::certificate::LawCitation;
use crate::facts::{DataflowEdge, PlanningFacts};
use crate::grammar::{DerivationStep, ScheduleProduction};

/// Spatial and concurrency execution topology of a candidate plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTopology {
    /// Sequential stage execution on a single queue (the baseline topology).
    Sequential,
    /// Concurrent execution of independent stages/arms across concurrent hardware queues/streams.
    ConcurrentQueue {
        /// Number of concurrent queues utilized.
        queues: u32,
    },
    /// Resident spatial partition across compute units.
    ResidentPartition {
        /// Number of spatial partitions / compute-unit domains allocated.
        partitions: u32,
        /// How spatial placement and progress are enforced.
        mode: ResidentPartitionMode,
    },
}

impl Default for ExecutionTopology {
    fn default() -> Self {
        Self::Sequential
    }
}

impl ExecutionTopology {
    /// Number of executable arms this topology submits work on.
    ///
    /// One arm is the sequential baseline. A concurrent or resident topology
    /// declaring no arm is a contradiction the caller cannot express as a
    /// submission, so it reads as the baseline rather than as zero queues.
    #[must_use]
    pub const fn arm_width(self) -> u32 {
        match self {
            Self::Sequential => 1,
            Self::ConcurrentQueue { queues } => {
                if queues == 0 {
                    1
                } else {
                    queues
                }
            }
            Self::ResidentPartition { partitions, .. } => {
                if partitions == 0 {
                    1
                } else {
                    partitions
                }
            }
        }
    }
}

/// Mode governing spatial placement and forward progress for resident partitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidentPartitionMode {
    /// Fixed hardware-enforceable spatial mask across compute units.
    /// Only legal when the target exposes an enforceable spatial partitioning capability.
    FixedSpatialMask,
    /// Bounded resident work queue whose scheduler preserves forward progress.
    /// Requires cooperative launch capability to ensure all resident blocks make progress without deadlock.
    BoundedWorkQueue,
}
/// Frontier-density traversal topology selected by the megakernel compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierTopology {
    /// Ultra-low-density frontier expansion where one subgroup owns sparse active
    /// nodes and avoids block-wide work distribution overhead.
    SubgroupSparseFrontier,
    /// Low-density frontier expansion with queue-like work distribution.
    SparseFrontier,
    /// Very high-density propagation where a block owns coalesced bitset lanes
    /// and amortizes shared-memory scans across many active facts.
    BlockDenseFrontier,
    /// Dense bitset-style propagation with coalesced scans.
    DenseFrontier,
    /// Mixed sparse/dense execution when density is in the transition band.
    HybridFrontier,
    /// Fused adjacent waves when launch and readback pressure dominate.
    FusedWave,
}

impl Default for FrontierTopology {
    fn default() -> Self {
        Self::SparseFrontier
    }
}

impl FrontierTopology {
    /// Baseline frontier topology with minimal scratch and memory pressure.
    #[must_use]
    pub const fn baseline() -> Self {
        Self::SparseFrontier
    }

    /// Whether this topology is the baseline sparse topology.
    #[must_use]
    pub const fn is_baseline(self) -> bool {
        matches!(self, Self::SparseFrontier)
    }

    /// Downgrade this topology to the baseline sparse topology.
    #[must_use]
    pub const fn fallback_baseline(self) -> Self {
        Self::SparseFrontier
    }
}

/// Constant density and pressure bands for frontier topology selection.
pub(crate) const SUBGROUP_SPARSE_DENSITY: f64 = 0.03125;
pub(crate) const SPARSE_DENSITY: f64 = 0.125;
pub(crate) const DENSE_DENSITY: f64 = 0.70;
pub(crate) const BLOCK_DENSE_DENSITY: f64 = 0.85;
pub(crate) const FUSION_PRESSURE: f64 = 0.70;
pub(crate) const FUSION_PRESSURE_HYSTERESIS: f64 = 0.10;
pub(crate) const FRONTIER_HYSTERESIS: f64 = 0.025;
pub(crate) const MEMORY_RED_ZONE_BPS: u32 = 9_000;
pub(crate) const MEMORY_HYSTERESIS_BPS: u32 = 250;
pub(crate) const LAUNCH_PRESSURE_BPS: u32 = 1_500;
pub(crate) const LAUNCH_HYSTERESIS_BPS: u32 = 250;
pub(crate) const FUSION_READBACK_BYTES: u64 = 4_096;
pub(crate) const DENSE_AVERAGE_DEGREE_BPS: u64 = 20_000;
pub(crate) const SUBGROUP_SPARSE_AVERAGE_DEGREE_BPS: u64 = 80_000;

/// Telemetry sample facts for frontier traversal topology selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierExecutionSample {
    /// Observed candidate dispatch cost in nanoseconds.
    pub dispatch_cost_ns: f64,
    /// Observed active-frontier density in `[0, 1]`.
    pub frontier_density: f64,
    /// Observed final readback byte volume.
    pub readback_bytes: u64,
}

impl FrontierExecutionSample {
    /// Name of the observed fact that lies outside its declared domain.
    ///
    /// Topology selection is total: it saturates a fact it cannot price so
    /// ranking stays defined for every input. Saturation makes an unmeasurable
    /// fact indistinguishable from a measured extreme, so a caller supplying
    /// measured telemetry checks the domain first and reports the fact instead
    /// of ranking against a substituted value.
    #[must_use]
    pub fn unrepresentable_fact(self) -> Option<&'static str> {
        if !self.dispatch_cost_ns.is_finite() || self.dispatch_cost_ns < 0.0 {
            return Some("dispatch_cost_ns");
        }
        if !self.frontier_density.is_finite()
            || self.frontier_density < 0.0
            || self.frontier_density > 1.0
        {
            return Some("frontier_density");
        }
        None
    }
}

/// Static graph shape facts for frontier traversal topology selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierGraphShape {
    /// Logical graph node count.
    pub node_count: u64,
    /// Logical graph edge count.
    pub edge_count: u64,
}

/// Device memory budget facts for frontier traversal topology selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierMemoryBudget {
    /// Estimated resident plus transient bytes required by candidate plan.
    pub required_bytes: u64,
    /// Caller-approved device-memory budget for the plan.
    pub budget_bytes: u64,
}

/// Decision output for frontier traversal topology selection with pressure metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierTopologyDecision {
    /// Selected frontier topology.
    pub topology: FrontierTopology,
    /// Required/budget memory pressure in basis points.
    pub memory_pressure_bps: u32,
    /// Edge/node average degree proxy in basis points.
    pub average_degree_bps: u64,
    /// Launch overhead divided by observed dispatch cost in basis points.
    pub launch_pressure_bps: u32,
}

impl FrontierTopologyDecision {
    /// Stable single-line explanation for release logs and scheduler debugging.
    #[must_use]
    pub fn stable_explanation(&self) -> String {
        format!(
            "megakernel-topology-v1|topology={:?}|memory_pressure_bps={}|average_degree_bps={}|launch_pressure_bps={}|reason={}",
            self.topology,
            self.memory_pressure_bps,
            self.average_degree_bps,
            self.launch_pressure_bps,
            self.reason_code()
        )
    }

    fn reason_code(&self) -> &'static str {
        match self.topology {
            FrontierTopology::SubgroupSparseFrontier => "ultra_sparse_subgroup_specialized",
            FrontierTopology::SparseFrontier if self.memory_pressure_bps >= 9_000 => {
                "memory_pressure_sparse_safety"
            }
            FrontierTopology::SparseFrontier => "low_density_sparse_queue",
            FrontierTopology::BlockDenseFrontier => "high_density_block_specialized",
            FrontierTopology::DenseFrontier => "dense_coalesced_frontier",
            FrontierTopology::HybridFrontier => "transition_band_hybrid",
            FrontierTopology::FusedWave => "launch_and_readback_pressure_fused",
        }
    }
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Select the frontier execution topology for one candidate wave.
#[must_use]
pub fn select_frontier_topology(
    sample: FrontierExecutionSample,
    graph: FrontierGraphShape,
    memory: FrontierMemoryBudget,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    supports_device_wide_barrier: bool,
) -> FrontierTopologyDecision {
    let memory_pressure_bps = if memory.budget_bytes == 0 {
        10_000
    } else {
        u32::try_from(
            (memory.required_bytes.min(memory.budget_bytes) * 10_000) / memory.budget_bytes,
        )
        .unwrap_or(10_000)
    };
    let average_degree_bps = if graph.node_count == 0 {
        0
    } else {
        (graph.edge_count.saturating_mul(10_000)) / graph.node_count
    };
    let launch_pressure_bps =
        if sample.dispatch_cost_ns <= 0.0 || !sample.dispatch_cost_ns.is_finite() {
            0
        } else {
            let ratio = (launch_overhead_ns.max(0.0) / sample.dispatch_cost_ns) * 10_000.0;
            if ratio.is_finite() {
                (ratio.round() as u64).min(u32::MAX as u64) as u32
            } else {
                u32::MAX
            }
        };
    let density = finite_unit(sample.frontier_density);
    let effective_fusion_pressure = if supports_device_wide_barrier {
        finite_unit(fusion_pressure)
    } else {
        0.0
    };
    let topology = if memory_pressure_bps >= MEMORY_RED_ZONE_BPS {
        FrontierTopology::SparseFrontier
    } else if effective_fusion_pressure >= FUSION_PRESSURE
        && launch_pressure_bps >= LAUNCH_PRESSURE_BPS
        && sample.readback_bytes >= FUSION_READBACK_BYTES
        && memory_pressure_bps <= MEMORY_RED_ZONE_BPS.saturating_sub(500)
    {
        FrontierTopology::FusedWave
    } else if density <= SUBGROUP_SPARSE_DENSITY
        && average_degree_bps <= SUBGROUP_SPARSE_AVERAGE_DEGREE_BPS
    {
        FrontierTopology::SubgroupSparseFrontier
    } else if density <= SPARSE_DENSITY {
        FrontierTopology::SparseFrontier
    } else if density >= BLOCK_DENSE_DENSITY && average_degree_bps >= DENSE_AVERAGE_DEGREE_BPS {
        FrontierTopology::BlockDenseFrontier
    } else if density >= DENSE_DENSITY && average_degree_bps >= DENSE_AVERAGE_DEGREE_BPS {
        FrontierTopology::DenseFrontier
    } else {
        FrontierTopology::HybridFrontier
    };
    FrontierTopologyDecision {
        topology,
        memory_pressure_bps,
        average_degree_bps,
        launch_pressure_bps,
    }
}

/// Select frontier execution topology with previous-topology hysteresis.
#[must_use]
pub fn select_frontier_topology_stable(
    sample: FrontierExecutionSample,
    graph: FrontierGraphShape,
    memory: FrontierMemoryBudget,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    previous_topology: FrontierTopology,
    supports_device_wide_barrier: bool,
) -> FrontierTopologyDecision {
    let mut decision = select_frontier_topology(
        sample,
        graph,
        memory,
        launch_overhead_ns,
        fusion_pressure,
        supports_device_wide_barrier,
    );
    let effective_fusion_pressure = if supports_device_wide_barrier {
        finite_unit(fusion_pressure)
    } else {
        0.0
    };
    decision.topology = stabilize_frontier_topology(
        decision,
        sample,
        effective_fusion_pressure,
        previous_topology,
    );
    decision
}

fn stabilize_frontier_topology(
    decision: FrontierTopologyDecision,
    sample: FrontierExecutionSample,
    fusion_pressure: f64,
    previous_topology: FrontierTopology,
) -> FrontierTopology {
    if decision.memory_pressure_bps >= MEMORY_RED_ZONE_BPS {
        return decision.topology;
    }
    let density = finite_unit(sample.frontier_density);
    let fusion = finite_unit(fusion_pressure);
    if matches!(
        previous_topology,
        FrontierTopology::SparseFrontier | FrontierTopology::SubgroupSparseFrontier
    ) && decision.memory_pressure_bps
        >= MEMORY_RED_ZONE_BPS.saturating_sub(MEMORY_HYSTERESIS_BPS)
    {
        return FrontierTopology::SparseFrontier;
    }

    match previous_topology {
        FrontierTopology::SubgroupSparseFrontier
            if density <= SUBGROUP_SPARSE_DENSITY + FRONTIER_HYSTERESIS
                && decision.average_degree_bps <= SUBGROUP_SPARSE_AVERAGE_DEGREE_BPS =>
        {
            FrontierTopology::SubgroupSparseFrontier
        }
        FrontierTopology::SparseFrontier if density <= SPARSE_DENSITY + FRONTIER_HYSTERESIS => {
            FrontierTopology::SparseFrontier
        }
        FrontierTopology::HybridFrontier
            if decision.topology == FrontierTopology::SparseFrontier
                && density >= SPARSE_DENSITY - FRONTIER_HYSTERESIS =>
        {
            FrontierTopology::HybridFrontier
        }
        FrontierTopology::HybridFrontier
            if matches!(
                decision.topology,
                FrontierTopology::DenseFrontier | FrontierTopology::BlockDenseFrontier
            ) && density <= DENSE_DENSITY + FRONTIER_HYSTERESIS =>
        {
            FrontierTopology::HybridFrontier
        }
        FrontierTopology::DenseFrontier
            if density >= DENSE_DENSITY - FRONTIER_HYSTERESIS
                && decision.average_degree_bps >= DENSE_AVERAGE_DEGREE_BPS =>
        {
            FrontierTopology::DenseFrontier
        }
        FrontierTopology::BlockDenseFrontier
            if density >= BLOCK_DENSE_DENSITY - FRONTIER_HYSTERESIS
                && decision.average_degree_bps >= DENSE_AVERAGE_DEGREE_BPS =>
        {
            FrontierTopology::BlockDenseFrontier
        }
        FrontierTopology::FusedWave
            if fusion >= FUSION_PRESSURE - FUSION_PRESSURE_HYSTERESIS
                && decision.launch_pressure_bps
                    >= LAUNCH_PRESSURE_BPS.saturating_sub(LAUNCH_HYSTERESIS_BPS)
                && sample.readback_bytes >= FUSION_READBACK_BYTES
                && decision.memory_pressure_bps
                    <= MEMORY_RED_ZONE_BPS.saturating_sub(MEMORY_HYSTERESIS_BPS) =>
        {
            FrontierTopology::FusedWave
        }
        _ => decision.topology,
    }
}

/// Cheap structural identity of one candidate plan.
pub(crate) type CandidateKey = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CandidatePlan {
    pub(crate) node_groups: Vec<u32>,
    pub(crate) fused_edges: Vec<DataflowEdge>,
    /// Launch width this candidate proposes for every group whose members all
    /// tolerate one, or `None` to launch every group at its declared width.
    pub(crate) workgroup_width: Option<u32>,
    /// Execution topology proposed for this candidate.
    pub(crate) topology: ExecutionTopology,
    /// Frontier-density traversal topology proposed for this candidate.
    pub(crate) frontier_topology: FrontierTopology,
    /// Typed neutral schedule transformed by candidate search.
    pub(crate) schedule: SelectedSchedule,
    /// Fail-closed transform diagnostic retained until candidate rejection.
    pub(crate) schedule_error: Option<ScheduleLegalityError>,
    /// Grammar productions applied to the baseline, in application order.
    pub(crate) derivation: Vec<DerivationStep>,
    /// Node programs a declared law rewrote, and the laws that authorized each.
    ///
    /// Empty for a candidate the schedule grammar alone derived.
    pub(crate) law_derivation: Vec<LawCitation>,
    /// Measurements this candidate is priced and judged against.
    ///
    /// A law-derived candidate runs a rewritten program, so pricing it against
    /// the graph's measurements would rank a program nobody emits. Shared
    /// rather than cloned per descendant: the grammar expands a law-derived
    /// candidate like any other, and every descendant runs the same rewritten
    /// node.
    pub(crate) law_facts: Option<Arc<PlanningFacts>>,
}

impl CandidatePlan {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn baseline(node_count: usize) -> Self {
        Self::baseline_with_schedule(vyre_test_support::selected_schedules::synthetic(node_count))
    }

    #[must_use]
    pub(crate) fn baseline_for(logical: &LogicalProgramGraph<'_>) -> Self {
        Self::baseline_with_schedule(crate::baseline::baseline_schedule(logical))
    }

    fn baseline_with_schedule(schedule: SelectedSchedule) -> Self {
        let node_count = schedule.phases.len();
        Self {
            node_groups: (0..node_count)
                .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
                .collect(),
            fused_edges: Vec::new(),
            derivation: Vec::new(),
            workgroup_width: None,
            topology: ExecutionTopology::Sequential,
            frontier_topology: FrontierTopology::SparseFrontier,
            schedule,
            schedule_error: None,
            law_derivation: Vec::new(),
            law_facts: None,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_edges(node_count: usize, edges: &[DataflowEdge]) -> Self {
        let (node_groups, fused_edges) = Self::groups_from_edges(node_count, edges);
        let (schedule, schedule_error) = schedule_for_groups(
            vyre_test_support::selected_schedules::synthetic(node_count),
            &node_groups,
        );
        Self {
            node_groups,
            fused_edges,
            derivation: Vec::new(),
            workgroup_width: None,
            topology: ExecutionTopology::Sequential,
            frontier_topology: FrontierTopology::SparseFrontier,
            schedule,
            schedule_error,
            law_derivation: Vec::new(),
            law_facts: None,
        }
    }

    /// Group nodes that one fused edge set joins, and the edges in stable order.
    fn groups_from_edges(
        node_count: usize,
        edges: &[DataflowEdge],
    ) -> (Vec<u32>, Vec<DataflowEdge>) {
        let mut parent: Vec<usize> = (0..node_count).collect();
        for edge in edges {
            let from = edge.from.0 as usize;
            let to = edge.to.0 as usize;
            if from >= node_count || to >= node_count {
                continue;
            }
            let from_root = root(&mut parent, from);
            let to_root = root(&mut parent, to);
            if from_root != to_root {
                let first = from_root.min(to_root);
                let second = from_root.max(to_root);
                parent[second] = first;
            }
        }

        let mut roots = Vec::<usize>::new();
        let mut node_groups = Vec::with_capacity(node_count);
        for node in 0..node_count {
            let root = root(&mut parent, node);
            let group = match roots.iter().position(|candidate| *candidate == root) {
                Some(group) => group,
                None => {
                    roots.push(root);
                    roots.len() - 1
                }
            };
            node_groups.push(u32::try_from(group).unwrap_or(u32::MAX));
        }
        let mut fused_edges = edges.to_vec();
        fused_edges.sort_by_key(|edge| (edge.from, edge.to, edge.value));
        fused_edges.dedup();
        (node_groups, fused_edges)
    }

    #[must_use]
    pub(crate) fn from_edges_for(
        logical: &LogicalProgramGraph<'_>,
        edges: &[DataflowEdge],
    ) -> Self {
        let (node_groups, fused_edges) =
            Self::groups_from_edges(logical.graph().nodes().len(), edges);
        let (schedule, schedule_error) =
            schedule_for_groups(crate::baseline::baseline_schedule(logical), &node_groups);
        Self {
            node_groups,
            fused_edges,
            derivation: Vec::new(),
            workgroup_width: None,
            topology: ExecutionTopology::Sequential,
            frontier_topology: FrontierTopology::SparseFrontier,
            schedule,
            schedule_error,
            law_derivation: Vec::new(),
            law_facts: None,
        }
    }

    /// Same grouping launched at `width` instead of the declared widths.
    #[must_use]
    pub(crate) fn with_workgroup_width(&self, width: Option<u32>) -> Self {
        let mut candidate = self.clone();
        candidate.workgroup_width = width;
        candidate
    }

    /// Same grouping executed with `topology`.
    #[must_use]
    pub(crate) fn with_topology(&self, topology: ExecutionTopology) -> Self {
        let mut candidate = self.clone();
        candidate.topology = topology;
        if candidate.schedule_error.is_none() {
            let phases = candidate
                .schedule
                .phases
                .iter()
                .map(|phase| phase.id)
                .collect::<Vec<_>>();
            for phase in phases {
                let transform = match topology {
                    ExecutionTopology::Sequential | ExecutionTopology::ConcurrentQueue { .. } => {
                        None
                    }
                    ExecutionTopology::ResidentPartition {
                        partitions,
                        mode: ResidentPartitionMode::FixedSpatialMask,
                    } => Some(ScheduleTransform::SpatialPartition {
                        phase,
                        partitions,
                        level: MappingLevel::ComputeUnitPartition,
                    }),
                    ExecutionTopology::ResidentPartition {
                        partitions,
                        mode: ResidentPartitionMode::BoundedWorkQueue,
                    } => Some(ScheduleTransform::PersistentQueue {
                        phase,
                        capacity: partitions,
                    }),
                };
                if let Some(transform) = transform {
                    if let Err(error) = candidate.schedule.apply(transform) {
                        candidate.schedule_error = Some(error);
                        break;
                    }
                }
            }
        }
        candidate
    }

    #[must_use]
    pub(crate) fn topology(&self) -> ExecutionTopology {
        self.topology
    }

    #[must_use]
    pub(crate) fn frontier_topology(&self) -> FrontierTopology {
        self.frontier_topology
    }

    #[must_use]
    pub(crate) fn with_frontier_topology(&self, frontier_topology: FrontierTopology) -> Self {
        let mut candidate = self.clone();
        candidate.frontier_topology = frontier_topology;
        candidate
    }

    #[must_use]
    pub(crate) fn group_count(&self) -> usize {
        self.node_groups
            .iter()
            .copied()
            .max()
            .map_or(0, |group| group as usize + 1)
    }

    /// Nodes belonging to one fusion group, in node order.
    pub(crate) fn group_members(&self, group: u32) -> impl Iterator<Item = usize> + '_ {
        self.node_groups
            .iter()
            .enumerate()
            .filter(move |(_, member)| **member == group)
            .map(|(node, _)| node)
    }

    /// Workgroup dimensions this candidate launches one group with.
    ///
    /// A proposed width applies only when every member of the group tolerates
    /// one; a single member that observes its launch width holds the whole group
    /// at the declared shape, because the group emits one module.
    #[must_use]
    pub(crate) fn group_workgroup(&self, group: u32, facts: &PlanningFacts) -> [u32; 3] {
        let declared = self
            .group_members(group)
            .filter_map(|node| facts.node_declared_workgroup.get(node).copied())
            .next()
            .unwrap_or([1, 1, 1]);
        let Some(width) = self.workgroup_width else {
            return declared;
        };
        let uniform = self
            .group_members(group)
            .all(|node| facts.node_accepts_width.get(node).copied().unwrap_or(false));
        if uniform {
            [width, 1, 1]
        } else {
            declared
        }
    }

    /// Invocations per workgroup this candidate launches one group with.
    #[must_use]
    pub(crate) fn group_invocations(&self, group: u32, facts: &PlanningFacts) -> u64 {
        let workgroup = self.group_workgroup(group, facts);
        u64::from(workgroup[0])
            .saturating_mul(u64::from(workgroup[1]))
            .saturating_mul(u64::from(workgroup[2]))
            .max(1)
    }

    pub(crate) fn selected_schedule(
        &self,
        facts: &PlanningFacts,
    ) -> Result<SelectedSchedule, ScheduleLegalityError> {
        if let Some(error) = &self.schedule_error {
            return Err(error.clone());
        }
        let mut schedule = self.schedule.clone();
        let phases = schedule
            .phases
            .iter()
            .filter_map(|phase| {
                phase
                    .source_regions
                    .first()
                    .map(|region| (phase.id, *region, phase.workgroup))
            })
            .collect::<Vec<_>>();
        for (phase, region, current) in phases {
            let group = self
                .node_groups
                .get(region as usize)
                .copied()
                .ok_or(ScheduleLegalityError::MissingRegion(region))?;
            let shape = self.group_workgroup(group, facts);
            if shape == current {
                continue;
            }
            schedule.apply(ScheduleTransform::SetWorkgroup { phase, shape })?;
        }
        schedule.validate()?;
        Ok(schedule)
    }

    #[must_use]
    pub(crate) fn schedule_error(&self) -> Option<&ScheduleLegalityError> {
        self.schedule_error.as_ref()
    }

    /// Apply one grammar step to this candidate.
    ///
    /// The step's transforms are applied to the schedule, and everything the
    /// candidate states about grouping is then re-derived from the schedule, so
    /// the schedule stays the single authority over what the plan is.
    pub(crate) fn derive(
        &self,
        step: &DerivationStep,
        facts: &PlanningFacts,
    ) -> Result<Self, ScheduleLegalityError> {
        if let Some(error) = &self.schedule_error {
            return Err(error.clone());
        }
        let mut candidate = self.clone();
        for transform in &step.transforms {
            candidate.schedule.apply(transform.clone())?;
        }
        if step.production == ScheduleProduction::LaunchWidth {
            if let Some(width) = step
                .transforms
                .iter()
                .find_map(|transform| match transform {
                    ScheduleTransform::SetWorkgroup { shape, .. } => Some(shape[0]),
                    _ => None,
                })
            {
                candidate.workgroup_width = Some(width);
            }
        }
        candidate.derivation.push(step.clone());
        candidate.topology = resident_topology(&candidate.schedule).unwrap_or(candidate.topology);
        candidate.regroup(facts)?;
        Ok(candidate)
    }

    /// Re-derive fusion grouping from the phases of the selected schedule.
    fn regroup(&mut self, facts: &PlanningFacts) -> Result<(), ScheduleLegalityError> {
        let mut phases = self.schedule.phases.iter().collect::<Vec<_>>();
        phases.sort_by_key(|phase| phase.id);
        let mut groups = vec![None; self.node_groups.len()];
        for (group, phase) in phases.iter().enumerate() {
            for region in &phase.source_regions {
                let slot = groups
                    .get_mut(*region as usize)
                    .ok_or(ScheduleLegalityError::MissingRegion(*region))?;
                *slot = Some(u32::try_from(group).unwrap_or(u32::MAX));
            }
        }
        let mut node_groups = Vec::with_capacity(groups.len());
        for (node, group) in groups.into_iter().enumerate() {
            node_groups.push(group.ok_or(ScheduleLegalityError::MissingRegion(
                u32::try_from(node).unwrap_or(u32::MAX),
            ))?);
        }
        self.fused_edges = facts
            .dataflow
            .iter()
            .copied()
            .filter(|edge| {
                node_groups.get(edge.from.0 as usize) == node_groups.get(edge.to.0 as usize)
            })
            .collect();
        self.node_groups = node_groups;
        Ok(())
    }

    /// Structural key that recognizes a candidate already derived.
    ///
    /// Content identity would serialize and hash the whole schedule for every
    /// derivation, which is the dominant cost of a bounded search. A structural
    /// hash is cheap, and a collision only drops one candidate from the set.
    #[must_use]
    pub(crate) fn canonical_key(&self) -> CandidateKey {
        let mut hasher = DefaultHasher::new();
        self.schedule.phases.hash(&mut hasher);
        self.schedule.transforms.hash(&mut hasher);
        self.node_groups.hash(&mut hasher);
        self.workgroup_width.hash(&mut hasher);
        self.topology.hash(&mut hasher);
        self.frontier_topology.hash(&mut hasher);
        self.law_derivation.hash(&mut hasher);
        hasher.finish()
    }

    /// This candidate with one node's program replaced by a law-derived one.
    ///
    /// The grouping, the launch width and the topology are untouched: a law
    /// states an equality between programs, not a schedule. The candidate is
    /// then expanded by the grammar like any other, so a law-derived program
    /// reaches every schedule the grammar can propose for it.
    #[must_use]
    pub(crate) fn with_law_derivation(
        &self,
        citation: LawCitation,
        facts: Arc<PlanningFacts>,
    ) -> Self {
        let mut candidate = self.clone();
        candidate.law_derivation.push(citation);
        candidate.law_facts = Some(facts);
        candidate
    }

    /// The measurements this candidate is priced and judged against.
    ///
    /// `graph` for every candidate the grammar derived, and the rewritten
    /// node's measurements for one a law derived. Pricing a law-derived
    /// candidate against the graph's measurements would rank a program nobody
    /// emits.
    #[must_use]
    pub(crate) fn priced_against<'a>(&'a self, graph: &'a PlanningFacts) -> &'a PlanningFacts {
        self.law_facts.as_deref().unwrap_or(graph)
    }
}

/// Resident topology the schedule itself records, if it records one.
///
/// Spatial partitioning and a bounded resident queue are schedule transforms, so
/// the topology a candidate reports is read off the schedule instead of being
/// carried beside it. Concurrent queues are a submission arrangement the
/// schedule does not express, so they are not derived here.
fn resident_topology(schedule: &SelectedSchedule) -> Option<ExecutionTopology> {
    let mut partitions = 0_u32;
    let mut mode = None;
    for record in &schedule.transforms {
        match record.transform {
            ScheduleTransform::SpatialPartition {
                partitions: count, ..
            } => {
                partitions = partitions.max(count);
                mode = mode.or(Some(ResidentPartitionMode::FixedSpatialMask));
            }
            ScheduleTransform::PersistentQueue { capacity, .. } => {
                partitions = partitions.max(capacity);
                mode = Some(ResidentPartitionMode::BoundedWorkQueue);
            }
            _ => {}
        }
    }
    mode.map(|mode| ExecutionTopology::ResidentPartition { partitions, mode })
}

fn schedule_for_groups(
    mut schedule: SelectedSchedule,
    node_groups: &[u32],
) -> (SelectedSchedule, Option<ScheduleLegalityError>) {
    let mut phases_by_group = BTreeMap::<u32, Vec<SchedulePhaseId>>::new();
    for (node, group) in node_groups.iter().copied().enumerate() {
        phases_by_group
            .entry(group)
            .or_default()
            .push(SchedulePhaseId(u32::try_from(node).unwrap_or(u32::MAX)));
    }
    for phases in phases_by_group
        .into_values()
        .filter(|phases| phases.len() > 1)
    {
        if let Err(error) = schedule.apply(ScheduleTransform::Fuse { phases }) {
            return (schedule, Some(error));
        }
    }
    (schedule, None)
}

fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}
