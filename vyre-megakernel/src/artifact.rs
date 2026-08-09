use vyre_foundation::ir::ProgramGraph;

use crate::{
    build_barriers, build_materializations, domain_digest, failure, group_stages, ArtifactNodeId,
    CompileError, DiagnosticCode, FusionGroupId, FusionRecord, FusionRejection, SearchBudget,
    SelectedPlan,
};

const LEGALITY_DIGEST_DOMAIN: &[u8] = b"VYRE_FUSION_LEGALITY_V1\0";

pub(crate) struct ArtifactPlan {
    pub(crate) node_groups: Vec<FusionGroupId>,
    pub(crate) stages: Vec<u32>,
    pub(crate) selected_plan: SelectedPlan,
}

pub(crate) fn plan(
    graph: &ProgramGraph,
    dependencies: &[crate::DependencyEdge],
    budget: SearchBudget,
) -> Result<ArtifactPlan, CompileError> {
    let facts = crate::facts::derive(graph, dependencies);
    let search = crate::search::explore(graph, &facts, dependencies, budget);
    let selection = crate::select::choose(search.candidates, &facts, dependencies);
    let candidate = selection.candidate;
    let node_groups: Vec<FusionGroupId> = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect();
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
    let pruned_fusions = search
        .rejected
        .into_iter()
        .map(|rejection| FusionRejection {
            from: rejection.edge.from,
            to: rejection.edge.to,
            value: rejection.edge.value,
            reason: rejection.reason,
        })
        .collect();
    let barriers = build_barriers(dependencies, &node_groups, &stages)?;
    let materializations = build_materializations(graph, &node_groups, &stages);
    if node_groups.len() != graph.nodes().len() {
        return Err(failure(
            DiagnosticCode::InvalidProgram,
            "planner.node_groups",
            "planner did not assign every graph node",
            "report the compiler defect",
        ));
    }
    Ok(ArtifactPlan {
        node_groups,
        stages,
        selected_plan: SelectedPlan {
            fusion,
            barriers,
            materializations,
            candidates_explored: search.work.candidates_explored,
            search_budget: budget,
            search_work: search.work,
            selection_cost: selection.cost,
            pruned_fusions,
        },
    })
}
