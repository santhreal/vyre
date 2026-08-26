use vyre_foundation::logical::LogicalProgramGraph;

use crate::{
    candidate::CandidatePlan,
    certificate::SearchCertificate,
    derive,
    facts::{DataflowEdge, PlanningFacts},
    legality::{analyze_fusion_pair, FusionDecision, FusionRejectionReason},
    DependencyEdge, DeviceFacts, SearchBudget, SearchWork,
};

#[derive(Debug)]
pub(crate) struct RejectedEdge {
    pub(crate) edge: DataflowEdge,
    pub(crate) reason: FusionRejectionReason,
}

#[derive(Debug)]
pub(crate) struct SearchResult {
    pub(crate) candidates: Vec<CandidatePlan>,
    pub(crate) rejected: Vec<RejectedEdge>,
    pub(crate) certificate: SearchCertificate,
    pub(crate) work: SearchWork,
}

/// Derive every candidate the grammar reaches within one budget.
///
/// Fusion legality is decided per producer-consumer pair before derivation, so
/// an illegal pair is recorded with its stable reason in the artifact instead of
/// being derived and eliminated anonymously.
pub(crate) fn explore(
    logical: &LogicalProgramGraph<'_>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    budget: SearchBudget,
    device: DeviceFacts,
) -> SearchResult {
    let graph = logical.graph();
    let mut rejected = Vec::new();
    let mut cpu_work = 0_u64;
    for edge in &facts.dataflow {
        if !can_spend(cpu_work, budget) {
            break;
        }
        cpu_work = cpu_work.saturating_add(1);
        if let FusionDecision::Rejected(reason) =
            analyze_fusion_pair(graph, edge.from, edge.to, edge.value)
        {
            rejected.push(RejectedEdge {
                edge: *edge,
                reason,
            });
        }
    }

    let derived = derive::derive(logical, facts, dependencies, budget, device);
    let cpu_work = cpu_work
        .saturating_add(derived.cpu_work)
        .min(budget.max_cpu_work);
    let candidates = derived.candidates;

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
        certificate: derived.certificate,
    }
}

fn can_spend(cpu_work: u64, budget: SearchBudget) -> bool {
    cpu_work < budget.max_cpu_work && cpu_work < budget.max_elapsed_ns
}
