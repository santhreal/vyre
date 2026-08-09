use serde::{Deserialize, Serialize};

use crate::{
    candidate::CandidatePlan, facts::PlanningFacts, DependencyEdge, DependencyEndpoint,
    DependencyKind,
};

const LAUNCH_WEIGHT: u64 = 1_000;
const MATERIALIZATION_WEIGHT: u64 = 100;

/// Reproducible components of the open compiler selection cost model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Sum of semantic IR nodes in the complete graph.
    pub semantic_work: u64,
    /// Number of generated kernel launches.
    pub launches: u64,
    /// Number of values crossing generated-kernel boundaries.
    pub materializations: u64,
    /// Weighted total minimized by candidate selection.
    pub total: u64,
}

pub(crate) fn evaluate(
    candidate: &CandidatePlan,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
) -> CostBreakdown {
    let semantic_work = facts
        .node_work
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let launches = u64::try_from(candidate.group_count()).unwrap_or(u64::MAX);
    let materializations = dependencies
        .iter()
        .filter(|edge| {
            if edge.kind != DependencyKind::Data {
                return false;
            }
            let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) =
                (edge.from, edge.to)
            else {
                return false;
            };
            candidate.node_groups.get(from.0 as usize) != candidate.node_groups.get(to.0 as usize)
        })
        .count() as u64;
    let total = semantic_work
        .saturating_add(launches.saturating_mul(LAUNCH_WEIGHT))
        .saturating_add(materializations.saturating_mul(MATERIALIZATION_WEIGHT));
    CostBreakdown {
        semantic_work,
        launches,
        materializations,
        total,
    }
}
