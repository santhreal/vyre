use crate::{
    candidate::CandidatePlan,
    cost::{evaluate, CostBreakdown},
    facts::PlanningFacts,
    DependencyEdge, DeviceFacts,
};

#[derive(Debug)]
pub(crate) struct Selection {
    pub(crate) candidate: CandidatePlan,
    pub(crate) cost: CostBreakdown,
}

/// Every scored candidate, cheapest first.
///
/// Ordering is total cost, then the group vector, then the proposed launch
/// width, so two candidates that cost the same are ordered by content and one
/// compilation of one graph selects one plan.
pub(crate) fn rank(
    candidates: Vec<CandidatePlan>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
) -> Vec<Selection> {
    let mut ranked = candidates
        .into_iter()
        .map(|candidate| {
            let cost = evaluate(&candidate, facts, dependencies, device);
            Selection { candidate, cost }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.cost
            .total
            .cmp(&right.cost.total)
            .then_with(|| left.candidate.node_groups.cmp(&right.candidate.node_groups))
            .then_with(|| {
                left.candidate
                    .workgroup_width
                    .cmp(&right.candidate.workgroup_width)
            })
    });
    ranked
}
