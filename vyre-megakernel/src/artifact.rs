use vyre_foundation::ir::ProgramGraph;

use crate::{
    build_barriers, build_materializations, domain_digest, facts::PlanningFacts, failure,
    group_stages, select::Selection, ArtifactNodeId, CompileError, CompilerFailureKind,
    DependencyEdge, DeviceFacts, ExecutionMode, ExternalFacts, FusionGroupId, FusionRecord,
    FusionRejection, GeometryRecord, PlanMeasurement, SearchBudget, SearchWork, SelectedPlan,
};

const LEGALITY_DIGEST_DOMAIN: &[u8] = b"VYRE_FUSION_LEGALITY_V1\0";

pub(crate) struct ArtifactPlan {
    pub(crate) node_groups: Vec<FusionGroupId>,
    pub(crate) stages: Vec<u32>,
    pub(crate) geometry: Vec<GeometryRecord>,
    pub(crate) selected_plan: SelectedPlan,
}

/// Everything one candidate needs to become a recorded plan.
pub(crate) struct PlanInputs<'a> {
    pub(crate) graph: &'a ProgramGraph,
    pub(crate) dependencies: &'a [DependencyEdge],
    pub(crate) facts: &'a PlanningFacts,
    pub(crate) selection: &'a Selection,
    pub(crate) pruned_fusions: &'a [FusionRejection],
    pub(crate) external: &'a ExternalFacts,
    pub(crate) device: DeviceFacts,
    pub(crate) budget: SearchBudget,
    pub(crate) work: SearchWork,
    pub(crate) measurement: PlanMeasurement,
}

pub(crate) fn plan(inputs: PlanInputs<'_>) -> Result<ArtifactPlan, CompileError> {
    let PlanInputs {
        graph,
        dependencies,
        facts,
        selection,
        pruned_fusions,
        external,
        device,
        budget,
        work,
        measurement,
    } = inputs;
    let candidate = &selection.candidate;
    let node_groups: Vec<FusionGroupId> = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect();
    if node_groups.len() != graph.nodes().len() {
        return Err(failure(
            CompilerFailureKind::InvalidProgram,
            "planner.node_groups",
            "planner did not assign every graph node",
            "report the compiler defect",
        ));
    }
    let group_count = candidate.group_count();
    let stages = group_stages(group_count, dependencies, &node_groups)?;
    let fusion = (0..group_count)
        .map(|group| {
            let nodes: Vec<ArtifactNodeId> = node_groups
                .iter()
                .enumerate()
                .filter(|(_, node_group)| node_group.0 as usize == group)
                .map(|(node, _)| ArtifactNodeId(node as u32))
                .collect();
            let accepted_edges = candidate
                .fused_edges
                .iter()
                .filter(|edge| {
                    node_groups.get(edge.from.0 as usize).copied()
                        == Some(FusionGroupId(group as u32))
                        && node_groups.get(edge.to.0 as usize).copied()
                            == Some(FusionGroupId(group as u32))
                })
                .count();
            let evidence = if accepted_edges == 0 {
                b"MKL000_SINGLE_NODE_GROUP".as_slice()
            } else {
                b"MKL000_LEGAL_DATAFLOW".as_slice()
            };
            FusionRecord {
                id: FusionGroupId(group as u32),
                members: nodes,
                stage: stages[group],
                legality: vec![domain_digest(LEGALITY_DIGEST_DOMAIN, evidence)],
            }
        })
        .collect();
    let geometry = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(index, node)| GeometryRecord {
            node: ArtifactNodeId(node.id.0),
            workgroup_size: candidate.group_workgroup(candidate.node_groups[index], facts),
        })
        .collect();
    let barriers = build_barriers(dependencies, &node_groups, &stages)?;
    let materializations = build_materializations(graph, &node_groups, &stages);
    let execution = execution_mode(device, external, selection.cost.launches);
    Ok(ArtifactPlan {
        node_groups,
        stages,
        geometry,
        selected_plan: SelectedPlan {
            fusion,
            barriers,
            materializations,
            candidates_explored: work.candidates_explored,
            search_budget: budget,
            search_work: work,
            selection_cost: selection.cost,
            pruned_fusions: pruned_fusions.to_vec(),
            execution,
            measurement,
        },
    })
}

/// Decide how the runtime executes this plan.
///
/// One resident kernel polling a device-side work queue replaces the launches a
/// submission batch would otherwise issue: it pays the setup cost once and saves
/// one launch overhead per launch it removes. The trade is profitable only when
/// the overhead removed exceeds the setup paid, and only a device that can hold
/// the whole grid resident can run a kernel that waits on other workgroups, so
/// cooperative launch is a precondition. An unmeasured launch overhead leaves
/// nothing to amortize and selects static execution rather than a guess.
fn execution_mode(
    device: DeviceFacts,
    external: &ExternalFacts,
    launches_per_submission: u64,
) -> ExecutionMode {
    if !device.supports_cooperative_launch() || device.per_launch_overhead_ns() == 0 {
        return ExecutionMode::Static;
    }
    let launches = u128::from(external.expected_launch_batch)
        .saturating_mul(u128::from(launches_per_submission));
    if launches < 2 {
        return ExecutionMode::Static;
    }
    let removed = launches.saturating_mul(u128::from(device.per_launch_overhead_ns()));
    let setup = u128::from(device.persistent_setup_overhead_ns());
    if removed <= setup {
        return ExecutionMode::Static;
    }
    ExecutionMode::Persistent {
        saved_ns: u64::try_from(removed - setup).unwrap_or(u64::MAX),
    }
}
