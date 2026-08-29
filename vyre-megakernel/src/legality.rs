use serde::{Deserialize, Serialize};
use vyre_foundation::ir::{Program, ProgramGraph, ValueLifetime};

use crate::candidate::{CandidatePlan, ExecutionTopology, ResidentPartitionMode};
use crate::dependency_order::group_stages;
use crate::facts::PlanningFacts;
use crate::{
    ArtifactNodeId, ArtifactValueId, DependencyEdge, DependencyEndpoint, DependencyKind,
    DeviceFacts, FusionGroupId,
};
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

/// Stable reason that prevents a candidate execution topology from executing on a device or graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub(crate) enum TopologyRejectionReason {
    /// Target device does not report or support required concurrent queues.
    InsufficientConcurrentQueues,
    /// Target device does not report or support required compute units.
    InsufficientComputeUnits,
    /// Spatial masking requested on a target without enforceable spatial partitioning capability.
    UnenforceableSpatialMasking,
    /// Bounded resident queue or device-wide join requested on a device without cooperative launch.
    RequiresCooperativeLaunch,
    /// RAW/WAR/WAW hazard or resource alias between concurrent arms.
    ResourceConflict,
    /// Cross-arm control dependency or effect that cannot be satisfied by concurrent queues.
    ControlDependencyOrEffect,
    /// Asymmetric or divergent join across resident boundary without cooperative join or GridSync cut.
    IllegalAsymmetricJoin,
    /// Candidate has no independent arms to execute concurrently.
    NoIndependentConcurrency,
    /// Occupancy or scratch budget exceeded for resident execution.
    OccupancyExceeded,
}

impl TopologyRejectionReason {
    /// Stable machine-readable diagnostic code.
    #[must_use]
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::InsufficientConcurrentQueues => "MKL010_INSUFFICIENT_CONCURRENT_QUEUES",
            Self::InsufficientComputeUnits => "MKL011_INSUFFICIENT_COMPUTE_UNITS",
            Self::UnenforceableSpatialMasking => "MKL012_UNENFORCEABLE_SPATIAL_MASKING",
            Self::RequiresCooperativeLaunch => "MKL013_REQUIRES_COOPERATIVE_LAUNCH",
            Self::ResourceConflict => "MKL014_RESOURCE_CONFLICT",
            Self::ControlDependencyOrEffect => "MKL015_CONTROL_DEPENDENCY_OR_EFFECT",
            Self::IllegalAsymmetricJoin => "MKL016_ILLEGAL_ASYMMETRIC_JOIN",
            Self::NoIndependentConcurrency => "MKL017_NO_INDEPENDENT_CONCURRENCY",
            Self::OccupancyExceeded => "MKL018_OCCUPANCY_EXCEEDED",
        }
    }
}

/// Legality result for one proposed candidate execution topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TopologyDecision {
    /// Execution topology is legal and valid on the target device.
    Legal,
    /// Execution topology is rejected for the stable reason.
    Rejected(TopologyRejectionReason),
}

/// Validate candidate execution topology against graph dependencies, independence analysis, and device capabilities.
#[must_use]
pub(crate) fn analyze_topology_legality(
    candidate: &CandidatePlan,
    graph: &ProgramGraph,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
) -> TopologyDecision {
    if candidate.frontier_topology == crate::candidate::FrontierTopology::FusedWave {
        if !device.supports_cooperative_launch() {
            return TopologyDecision::Rejected(TopologyRejectionReason::RequiresCooperativeLaunch);
        }
        if candidate.group_count() >= candidate.node_groups.len() {
            return TopologyDecision::Rejected(TopologyRejectionReason::NoIndependentConcurrency);
        }
    }
    match candidate.topology {
        ExecutionTopology::Sequential => TopologyDecision::Legal,
        ExecutionTopology::ConcurrentQueue { queues } => {
            if queues < 2 || device.concurrent_queues() < queues {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::InsufficientConcurrentQueues,
                );
            }
            let group_count = candidate.group_count();
            if group_count < 2 {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            }
            let node_groups: Vec<FusionGroupId> = candidate
                .node_groups
                .iter()
                .copied()
                .map(FusionGroupId)
                .collect();
            let Ok(stages) = group_stages(group_count, dependencies, &node_groups) else {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            };
            let stage_count = stages.iter().copied().max().map_or(0, |s| s as usize + 1);
            let mut stage_groups = vec![Vec::new(); stage_count];
            for (group, &stage) in stages.iter().enumerate() {
                stage_groups[stage as usize].push(group as u32);
            }
            let has_concurrent_stage = stage_groups.iter().any(|groups| groups.len() >= 2);
            if !has_concurrent_stage {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            }

            for groups in &stage_groups {
                if groups.len() < 2 {
                    continue;
                }
                for i in 0..groups.len() {
                    for j in (i + 1)..groups.len() {
                        let g1 = groups[i];
                        let g2 = groups[j];
                        if let Some(reason) =
                            check_arm_conflicts(candidate, graph, dependencies, g1, g2)
                        {
                            return TopologyDecision::Rejected(reason);
                        }
                    }
                }
            }
            TopologyDecision::Legal
        }
        ExecutionTopology::ResidentPartition { partitions, mode } => {
            if partitions < 2 || device.compute_units() < partitions {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::InsufficientComputeUnits,
                );
            }
            match mode {
                ResidentPartitionMode::FixedSpatialMask => {
                    if !device.supports_spatial_partitioning() {
                        return TopologyDecision::Rejected(
                            TopologyRejectionReason::UnenforceableSpatialMasking,
                        );
                    }
                }
                ResidentPartitionMode::BoundedWorkQueue => {
                    if !device.supports_cooperative_launch() {
                        return TopologyDecision::Rejected(
                            TopologyRejectionReason::RequiresCooperativeLaunch,
                        );
                    }
                }
            }
            let group_count = candidate.group_count();
            if group_count < 2 {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            }
            let node_groups: Vec<FusionGroupId> = candidate
                .node_groups
                .iter()
                .copied()
                .map(FusionGroupId)
                .collect();
            let Ok(stages) = group_stages(group_count, dependencies, &node_groups) else {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            };
            let stage_count = stages.iter().copied().max().map_or(0, |s| s as usize + 1);
            let mut stage_groups = vec![Vec::new(); stage_count];
            for (group, &stage) in stages.iter().enumerate() {
                stage_groups[stage as usize].push(group as u32);
            }
            let has_concurrent_stage = stage_groups.iter().any(|groups| groups.len() >= 2);
            if !has_concurrent_stage {
                return TopologyDecision::Rejected(
                    TopologyRejectionReason::NoIndependentConcurrency,
                );
            }

            for groups in &stage_groups {
                if groups.len() < 2 {
                    continue;
                }
                let mut aggregate_scratch = 0_u64;
                let mut aggregate_live = 0_u64;
                for &g in groups {
                    for node in candidate.group_members(g) {
                        aggregate_live = aggregate_live
                            .saturating_add(facts.node_live_values.get(node).copied().unwrap_or(0));
                        for (_, bytes) in facts
                            .node_workgroup_scratch
                            .get(node)
                            .map(Vec::as_slice)
                            .unwrap_or_default()
                        {
                            aggregate_scratch = aggregate_scratch.saturating_add(*bytes);
                        }
                    }
                }
                let register_ceiling = device.hardware_registers_per_invocation();
                if register_ceiling > 0 && aggregate_live > u64::from(register_ceiling) {
                    return TopologyDecision::Rejected(TopologyRejectionReason::OccupancyExceeded);
                }
                if device.shared_scratch_bytes_per_workgroup() > 0
                    && aggregate_scratch > u64::from(device.shared_scratch_bytes_per_workgroup())
                {
                    return TopologyDecision::Rejected(TopologyRejectionReason::OccupancyExceeded);
                }

                for i in 0..groups.len() {
                    for j in (i + 1)..groups.len() {
                        let g1 = groups[i];
                        let g2 = groups[j];
                        if let Some(reason) =
                            check_arm_conflicts(candidate, graph, dependencies, g1, g2)
                        {
                            return TopologyDecision::Rejected(reason);
                        }
                    }
                }
            }

            if !device.supports_cooperative_launch() {
                if let Some(reason) = check_asymmetric_joins(candidate, dependencies, &stages) {
                    return TopologyDecision::Rejected(reason);
                }
            }

            TopologyDecision::Legal
        }
    }
}

fn check_arm_conflicts(
    candidate: &CandidatePlan,
    graph: &ProgramGraph,
    dependencies: &[DependencyEdge],
    g1: u32,
    g2: u32,
) -> Option<TopologyRejectionReason> {
    for edge in dependencies {
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            continue;
        };
        let Some(&from_group) = candidate.node_groups.get(from.0 as usize) else {
            continue;
        };
        let Some(&to_group) = candidate.node_groups.get(to.0 as usize) else {
            continue;
        };
        if (from_group == g1 && to_group == g2) || (from_group == g2 && to_group == g1) {
            return Some(TopologyRejectionReason::ResourceConflict);
        }
    }

    let mut g1_reads = Vec::new();
    let mut g1_writes = Vec::new();
    for node_idx in candidate.group_members(g1) {
        if let Some(node) = graph.nodes().get(node_idx) {
            if crate::grid_sync::requires_grid_sync(&node.program) {
                return Some(TopologyRejectionReason::ControlDependencyOrEffect);
            }
            for input in &node.inputs {
                g1_reads.push(input.value.0);
            }
            for output in &node.outputs {
                g1_writes.push(output.0);
            }
        }
    }

    let mut g2_reads = Vec::new();
    let mut g2_writes = Vec::new();
    for node_idx in candidate.group_members(g2) {
        if let Some(node) = graph.nodes().get(node_idx) {
            if crate::grid_sync::requires_grid_sync(&node.program) {
                return Some(TopologyRejectionReason::ControlDependencyOrEffect);
            }
            for input in &node.inputs {
                g2_reads.push(input.value.0);
            }
            for output in &node.outputs {
                g2_writes.push(output.0);
            }
        }
    }

    if g1_writes.iter().any(|w1| g2_writes.contains(w1)) {
        return Some(TopologyRejectionReason::ResourceConflict);
    }
    if g1_writes.iter().any(|w1| g2_reads.contains(w1)) {
        return Some(TopologyRejectionReason::ResourceConflict);
    }
    if g1_reads.iter().any(|r1| g2_writes.contains(r1)) {
        return Some(TopologyRejectionReason::ResourceConflict);
    }

    None
}

fn check_asymmetric_joins(
    candidate: &CandidatePlan,
    dependencies: &[DependencyEdge],
    stages: &[u32],
) -> Option<TopologyRejectionReason> {
    for edge in dependencies {
        if edge.kind != DependencyKind::Data {
            continue;
        }
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            continue;
        };
        let Some(&from_group) = candidate.node_groups.get(from.0 as usize) else {
            continue;
        };
        let Some(&to_group) = candidate.node_groups.get(to.0 as usize) else {
            continue;
        };
        if from_group == to_group {
            continue;
        }
        let from_stage = stages.get(from_group as usize).copied().unwrap_or(0);
        let to_stage = stages.get(to_group as usize).copied().unwrap_or(0);
        if to_stage > from_stage.saturating_add(1) {
            return Some(TopologyRejectionReason::IllegalAsymmetricJoin);
        }
    }
    None
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
    let pinned =
        pins_workgroup_geometry(&producer.program) || pins_workgroup_geometry(&consumer.program);
    if producer.program.workgroup_size != consumer.program.workgroup_size {
        if pinned {
            return FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary);
        }
        return FusionDecision::Rejected(FusionRejectionReason::WorkgroupMismatch);
    }
    FusionDecision::Legal
}

/// Whether changing the declared workgroup would change program semantics.
///
/// The semantic IR owner classifies geometry observability. Fusion uses the
/// same classification as logical identity and schedule-width search.
fn pins_workgroup_geometry(program: &Program) -> bool {
    !program.workgroup_size_is_schedule_only()
}

#[cfg(test)]
mod tests {
    use crate::graph_fixtures::{independent_two_arm_graph, two_arm_graph};
    use std::collections::BTreeMap;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
    use vyre_foundation::validate::BackendCapabilities;
    use vyre_test_support::pass_programs::copy_program;

    use super::*;
    use crate::facts::derive as derive_planning_facts;
    use crate::normalize::normalize;
    #[test]
    fn topology_variants_have_exhaustive_decisions() {
        let variants = [
            ExecutionTopology::Sequential,
            ExecutionTopology::ConcurrentQueue { queues: 2 },
            ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::FixedSpatialMask,
            },
            ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::BoundedWorkQueue,
            },
        ];

        for variant in variants {
            match variant {
                ExecutionTopology::Sequential => {}
                ExecutionTopology::ConcurrentQueue { queues } => {
                    assert!(queues >= 2);
                }
                ExecutionTopology::ResidentPartition { partitions, mode } => {
                    assert!(partitions >= 2);
                    match mode {
                        ResidentPartitionMode::FixedSpatialMask => {}
                        ResidentPartitionMode::BoundedWorkQueue => {}
                    }
                }
            }
        }
    }

    #[test]
    fn topology_rejection_reasons_have_stable_unique_codes() {
        let reasons = [
            TopologyRejectionReason::InsufficientConcurrentQueues,
            TopologyRejectionReason::InsufficientComputeUnits,
            TopologyRejectionReason::UnenforceableSpatialMasking,
            TopologyRejectionReason::RequiresCooperativeLaunch,
            TopologyRejectionReason::ResourceConflict,
            TopologyRejectionReason::ControlDependencyOrEffect,
            TopologyRejectionReason::IllegalAsymmetricJoin,
            TopologyRejectionReason::NoIndependentConcurrency,
            TopologyRejectionReason::OccupancyExceeded,
        ];

        let mut codes = std::collections::BTreeSet::new();
        for reason in reasons {
            let code = reason.code();
            assert!(
                code.starts_with("MKL0"),
                "diagnostic code must follow MKL convention: {code}"
            );
            assert!(
                codes.insert(code),
                "diagnostic code must be unique across reasons: {code}"
            );
        }
        assert_eq!(codes.len(), reasons.len());
    }

    fn bindings() -> BTreeMap<String, u64> {
        BTreeMap::from([("items".into(), 64)])
    }

    fn planning_facts(
        graph: &ProgramGraph,
        dependencies: &[DependencyEdge],
    ) -> crate::facts::PlanningFacts {
        let bindings = bindings();
        let logical =
            vyre_foundation::logical::LogicalProgramGraph::validate(graph, &bindings).unwrap();
        derive_planning_facts(&logical, dependencies, &bindings).unwrap()
    }
    fn test_device() -> DeviceFacts {
        DeviceFacts::new(BackendCapabilities::default(), 256)
            .with_occupancy(128, 4096)
            .with_compute_units(8)
            .with_concurrent_queues(4)
            .with_spatial_partitioning(true)
            .with_cooperative_launch(true)
    }

    #[test]
    fn test_insufficient_concurrent_queues() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate = CandidatePlan::baseline(2)
            .with_topology(ExecutionTopology::ConcurrentQueue { queues: 8 });
        let device = test_device().with_concurrent_queues(4);
        let decision =
            analyze_topology_legality(&candidate, &graph, &facts, &norm.dependencies, device);
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::InsufficientConcurrentQueues)
        );
    }

    #[test]
    fn test_insufficient_compute_units() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate =
            CandidatePlan::baseline(2).with_topology(ExecutionTopology::ResidentPartition {
                partitions: 16,
                mode: ResidentPartitionMode::FixedSpatialMask,
            });
        let device = test_device().with_compute_units(8);
        let decision =
            analyze_topology_legality(&candidate, &graph, &facts, &norm.dependencies, device);
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::InsufficientComputeUnits)
        );
    }

    #[test]
    fn test_unenforceable_spatial_masking() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate =
            CandidatePlan::baseline(2).with_topology(ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::FixedSpatialMask,
            });
        let device = test_device().with_spatial_partitioning(false);
        let decision =
            analyze_topology_legality(&candidate, &graph, &facts, &norm.dependencies, device);
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::UnenforceableSpatialMasking)
        );
    }

    #[test]
    fn test_requires_cooperative_launch() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate =
            CandidatePlan::baseline(2).with_topology(ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::BoundedWorkQueue,
            });
        let device = test_device().with_cooperative_launch(false);
        let decision =
            analyze_topology_legality(&candidate, &graph, &facts, &norm.dependencies, device);
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::RequiresCooperativeLaunch)
        );
    }

    #[test]
    fn test_no_independent_concurrency() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        // 1 group only
        let mut candidate = CandidatePlan::baseline(2)
            .with_topology(ExecutionTopology::ConcurrentQueue { queues: 2 });
        candidate.node_groups = vec![0, 0];
        let decision = analyze_topology_legality(
            &candidate,
            &graph,
            &facts,
            &norm.dependencies,
            test_device(),
        );
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::NoIndependentConcurrency)
        );
    }

    #[test]
    fn test_grid_sync_in_arm_rejects_control_dependency_or_effect() {
        let prog_with_fence = Program::wrapped(
            vec![
                BufferDecl::read_write("in_a", 0, DataType::U32),
                BufferDecl::read_write("out_a", 1, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::Barrier {
                ordering: MemoryOrdering::GridSync,
            }],
        );
        let graph = two_arm_graph(prog_with_fence, copy_program("in_b", "out_b"));
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate = CandidatePlan::baseline(2)
            .with_topology(ExecutionTopology::ConcurrentQueue { queues: 2 });
        let decision = analyze_topology_legality(
            &candidate,
            &graph,
            &facts,
            &norm.dependencies,
            test_device(),
        );
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::ControlDependencyOrEffect)
        );
    }

    #[test]
    fn test_occupancy_exceeded() {
        let prog_with_scratch = Program::wrapped(
            vec![
                BufferDecl::read_write("in_a", 0, DataType::U32),
                BufferDecl::read_write("out_a", 1, DataType::U32),
                BufferDecl::workgroup("scratch", 2048, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                "out_a",
                Expr::u32(0),
                Expr::load("in_a", Expr::u32(0)),
            )],
        );
        let graph = two_arm_graph(prog_with_scratch, copy_program("in_b", "out_b"));
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate =
            CandidatePlan::baseline(2).with_topology(ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::FixedSpatialMask,
            });
        let tiny_scratch_device = test_device().with_occupancy(128, 1024);
        let decision = analyze_topology_legality(
            &candidate,
            &graph,
            &facts,
            &norm.dependencies,
            tiny_scratch_device,
        );
        assert_eq!(
            decision,
            TopologyDecision::Rejected(TopologyRejectionReason::OccupancyExceeded)
        );
    }

    #[test]
    fn test_legal_independent_concurrency() {
        let graph = independent_two_arm_graph();
        let norm = normalize(&graph).unwrap();
        let facts = planning_facts(&graph, &norm.dependencies);
        let candidate_cq = CandidatePlan::baseline(2)
            .with_topology(ExecutionTopology::ConcurrentQueue { queues: 2 });
        let decision_cq = analyze_topology_legality(
            &candidate_cq,
            &graph,
            &facts,
            &norm.dependencies,
            test_device(),
        );
        assert_eq!(decision_cq, TopologyDecision::Legal);

        let candidate_sp =
            CandidatePlan::baseline(2).with_topology(ExecutionTopology::ResidentPartition {
                partitions: 2,
                mode: ResidentPartitionMode::FixedSpatialMask,
            });
        let decision_sp = analyze_topology_legality(
            &candidate_sp,
            &graph,
            &facts,
            &norm.dependencies,
            test_device(),
        );
        assert_eq!(decision_sp, TopologyDecision::Legal);
    }
}
