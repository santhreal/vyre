use crate::{
    candidate::CandidatePlan,
    cost::{evaluate, CostBreakdown},
    facts::PlanningFacts,
    DependencyEdge,
};

#[derive(Debug)]
pub(crate) struct Selection {
    pub(crate) candidate: CandidatePlan,
    pub(crate) cost: CostBreakdown,
}

/// Lowest-cost candidate, or nothing when the search scored no plan at all.
pub(crate) fn choose(
    candidates: Vec<CandidatePlan>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
) -> Option<Selection> {
    candidates
        .into_iter()
        .map(|candidate| {
            let cost = evaluate(&candidate, facts, dependencies);
            Selection { candidate, cost }
        })
        .min_by(|left, right| {
            left.cost
                .total
                .cmp(&right.cost.total)
                .then_with(|| left.candidate.node_groups.cmp(&right.candidate.node_groups))
        })
}
