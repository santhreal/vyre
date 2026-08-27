use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use vyre_foundation::{
    logical::LogicalProgramGraph,
    schedule::{
        MappingLevel, ScheduleLegalityError, SchedulePhaseId, ScheduleTransform, SelectedSchedule,
    },
};

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
    /// Typed neutral schedule transformed by candidate search.
    pub(crate) schedule: SelectedSchedule,
    /// Fail-closed transform diagnostic retained until candidate rejection.
    pub(crate) schedule_error: Option<ScheduleLegalityError>,
    /// Grammar productions applied to the baseline, in application order.
    pub(crate) derivation: Vec<DerivationStep>,
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
            schedule,
            schedule_error: None,
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
            schedule,
            schedule_error,
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
            schedule,
            schedule_error,
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
        hasher.finish()
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
