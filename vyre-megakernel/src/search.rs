use vyre_foundation::ir::ProgramGraph;

use crate::{
    candidate::CandidatePlan,
    facts::{DataflowEdge, PlanningFacts},
    legality::{analyze_fusion_pair, FusionDecision, FusionRejectionReason},
    DependencyEdge, DeviceFacts, FusionGroupId, SearchBudget, SearchWork,
};

/// Launch widths the search crosses every fusion candidate with.
///
/// The set stops at 32 and 256 on purpose. Below 32 a workgroup cannot fill a
/// subgroup on any supported device, and 256 is the widest group any recorded
/// `vyre-bench` case uses (`foundation.reduce.sum.1m` tiles at 256). A wider
/// group is admissible only on evidence the analytic rank does not have, so the
/// search does not propose one.
pub(crate) const WORKGROUP_SEARCH_WIDTHS: &[u32] = &[32, 64, 128, 256];

#[derive(Debug)]
pub(crate) struct RejectedEdge {
    pub(crate) edge: DataflowEdge,
    pub(crate) reason: FusionRejectionReason,
}

#[derive(Debug)]
pub(crate) struct SearchResult {
    pub(crate) candidates: Vec<CandidatePlan>,
    pub(crate) rejected: Vec<RejectedEdge>,
    pub(crate) work: SearchWork,
}

pub(crate) fn explore(
    graph: &ProgramGraph,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    budget: SearchBudget,
    device: DeviceFacts,
) -> SearchResult {
    let mut groupings = vec![CandidatePlan::baseline(graph.nodes().len())];
    let mut rejected = Vec::new();
    let mut legal_edges = Vec::new();
    let mut cpu_work = 0_u64;

    for edge in &facts.dataflow {
        if !can_spend(cpu_work, budget) {
            break;
        }
        cpu_work = cpu_work.saturating_add(1);
        match analyze_fusion_pair(graph, edge.from, edge.to, edge.value) {
            FusionDecision::Legal => {
                let candidate = CandidatePlan::from_edges(graph.nodes().len(), &[*edge]);
                if candidate_is_acyclic(&candidate, dependencies) {
                    legal_edges.push(*edge);
                    if groupings.len() < budget.max_candidates as usize {
                        groupings.push(candidate);
                    }
                } else {
                    rejected.push(RejectedEdge {
                        edge: *edge,
                        reason: FusionRejectionReason::DependencyCycle,
                    });
                }
            }
            FusionDecision::Rejected(reason) => rejected.push(RejectedEdge {
                edge: *edge,
                reason,
            }),
        }
    }

    if legal_edges.len() > 1
        && groupings.len() < budget.max_candidates as usize
        && can_spend(cpu_work, budget)
    {
        cpu_work = cpu_work.saturating_add(1);
        let mut accepted = Vec::new();
        for edge in legal_edges {
            let mut proposed = accepted.clone();
            proposed.push(edge);
            let candidate = CandidatePlan::from_edges(graph.nodes().len(), &proposed);
            if candidate_is_acyclic(&candidate, dependencies) {
                accepted = proposed;
            } else {
                rejected.push(RejectedEdge {
                    edge,
                    reason: FusionRejectionReason::DependencyCycle,
                });
            }
        }
        groupings.push(CandidatePlan::from_edges(graph.nodes().len(), &accepted));
    }
    groupings.sort_by(|left, right| left.node_groups.cmp(&right.node_groups));
    groupings.dedup_by(|left, right| left.node_groups == right.node_groups);

    let mut candidates = Vec::with_capacity(groupings.len());
    for grouping in groupings {
        for width in WORKGROUP_SEARCH_WIDTHS {
            if candidates.len().saturating_add(1) >= budget.max_candidates as usize
                || !can_spend(cpu_work, budget)
            {
                break;
            }
            if u64::from(*width) > u64::from(device.max_invocations_per_workgroup()) {
                continue;
            }
            if !width_moves_any_group(&grouping, facts, *width) {
                continue;
            }
            cpu_work = cpu_work.saturating_add(1);
            candidates.push(grouping.with_workgroup_width(Some(*width)));
        }
        candidates.push(grouping);
    }

    SearchResult {
        work: SearchWork {
            candidates_explored: u32::try_from(candidates.len()).unwrap_or(u32::MAX),
            cpu_work,
            target_compilations: 0,
            measurements: 0,
            elapsed_ns: cpu_work.min(budget.max_elapsed_ns),
        },
        candidates,
        rejected,
    }
}

/// Whether proposing `width` changes the launch shape of any group.
///
/// A width every group either rejects or already declares produces a candidate
/// identical to the one without it, so the search does not spend a slot on it.
fn width_moves_any_group(candidate: &CandidatePlan, facts: &PlanningFacts, width: u32) -> bool {
    (0..u32::try_from(candidate.group_count()).unwrap_or(u32::MAX)).any(|group| {
        let declared = candidate.group_workgroup(group, facts);
        let proposed = candidate
            .with_workgroup_width(Some(width))
            .group_workgroup(group, facts);
        declared != proposed
    })
}

fn candidate_is_acyclic(candidate: &CandidatePlan, dependencies: &[DependencyEdge]) -> bool {
    let groups = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect::<Vec<_>>();
    crate::group_stages(candidate.group_count(), dependencies, &groups).is_ok()
}

fn can_spend(cpu_work: u64, budget: SearchBudget) -> bool {
    cpu_work < budget.max_cpu_work && cpu_work < budget.max_elapsed_ns
}
