//! The bounded worklist that derives one candidate set from the grammar.
//!
//! Search starts from the unfused, unspecialized baseline and expands it by
//! applying one production at a time. Every derived candidate passes constraint
//! propagation before it can be expanded again, so an illegal plan is
//! eliminated with a stable reason instead of ranked. The result is
//! deterministic: productions are visited in grammar order, operands in
//! ascending order, and a candidate already derived along another path is
//! recognized by its canonical identity rather than re-explored.

use std::collections::BTreeSet;

use vyre_foundation::logical::LogicalProgramGraph;

use crate::{
    candidate::{CandidateKey, CandidatePlan, ExecutionTopology},
    certificate::{PruneReason, SearchCertificate},
    constraints::{self, ConstraintContext},
    cost,
    facts::PlanningFacts,
    grammar::{self, GrammarContext, ScheduleProduction, SCHEDULE_GRAMMAR_VERSION},
    legality::{analyze_topology_legality, TopologyDecision},
    objective::{CompileObjective, MetricFigures},
    DependencyEdge, DeviceFacts, SearchBudget,
};

/// Productions one candidate may accumulate.
///
/// Three is the shortest chain that reaches a plan which fuses a producer into
/// its consumer, launches the fused phase at a selected width, and places the
/// value they share. A deeper chain is reachable only by raising the budget,
/// which is where the bound belongs.
const MAX_DEPTH: u32 = 3;

/// One bounded derivation: the admitted candidates and the record of the search.
pub(crate) struct Derivation {
    /// Admitted candidates in derivation order, baseline first.
    pub(crate) candidates: Vec<CandidatePlan>,
    /// Reproducible record of what was derived, admitted, and eliminated.
    pub(crate) certificate: SearchCertificate,
    /// Abstract CPU work charged against the budget.
    pub(crate) cpu_work: u64,
}

/// Derive every candidate the grammar reaches within `budget`.
pub(crate) fn derive(
    logical: &LogicalProgramGraph<'_>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    budget: SearchBudget,
    device: DeviceFacts,
    objective: &CompileObjective,
) -> Derivation {
    let grammar = GrammarContext { facts };
    let constraint = ConstraintContext {
        graph: logical.graph(),
        facts,
        dependencies,
        device,
    };
    let baseline = arrange(
        CandidatePlan::baseline_for(logical),
        &constraint,
        facts,
        dependencies,
    );
    let mut certificate = SearchCertificate::new(SCHEDULE_GRAMMAR_VERSION);
    let mut seen = BTreeSet::new();
    seen.insert(baseline.canonical_key());
    let mut incumbent = primary_figure(
        &cost::evaluate(&baseline, facts, dependencies, device),
        device,
        objective,
    );
    let mut candidates = vec![baseline.clone()];
    let mut frontier = vec![baseline];
    let mut cpu_work = 0_u64;

    for depth in 1..=MAX_DEPTH {
        if frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        let exhausted = expand(
            &frontier,
            Expansion {
                grammar: &grammar,
                constraint: &constraint,
                facts,
                dependencies,
                device,
                budget,
                objective,
                remaining: MAX_DEPTH.saturating_sub(depth),
            },
            &mut Accumulator {
                certificate: &mut certificate,
                seen: &mut seen,
                candidates: &mut candidates,
                next: &mut next,
                incumbent: &mut incumbent,
                cpu_work: &mut cpu_work,
            },
        );
        if !next.is_empty() {
            certificate.reached_depth(depth);
        }
        frontier = next;
        if exhausted {
            certificate.exhausted();
            break;
        }
    }
    certificate.canonicalize();
    Derivation {
        candidates,
        certificate,
        cpu_work,
    }
}

/// Everything one expansion reads.
struct Expansion<'a, 'graph> {
    grammar: &'a GrammarContext<'a>,
    constraint: &'a ConstraintContext<'graph>,
    facts: &'a PlanningFacts,
    dependencies: &'a [DependencyEdge],
    device: DeviceFacts,
    budget: SearchBudget,
    objective: &'a CompileObjective,
    /// Productions a descendant of this frontier may still apply.
    remaining: u32,
}

/// Everything one expansion records.
struct Accumulator<'a> {
    certificate: &'a mut SearchCertificate,
    seen: &'a mut BTreeSet<CandidateKey>,
    candidates: &'a mut Vec<CandidatePlan>,
    next: &'a mut Vec<CandidatePlan>,
    /// Lowest primary-metric figure any admitted candidate has reached.
    incumbent: &'a mut u64,
    cpu_work: &'a mut u64,
}

/// Expand one frontier, returning whether a bound stopped the expansion.
fn expand(
    frontier: &[CandidatePlan],
    expansion: Expansion<'_, '_>,
    into: &mut Accumulator<'_>,
) -> bool {
    for parent in frontier {
        for production in ScheduleProduction::ALL {
            for step in grammar::propose(*production, &parent.schedule, expansion.grammar) {
                if !can_spend(*into.cpu_work, expansion.budget)
                    || into.candidates.len() >= expansion.budget.max_candidates as usize
                {
                    return true;
                }
                *into.cpu_work = into.cpu_work.saturating_add(1);
                into.certificate.derived(*production);
                let candidate = match parent.derive(&step, expansion.facts) {
                    Ok(candidate) => arrange(
                        candidate,
                        expansion.constraint,
                        expansion.facts,
                        expansion.dependencies,
                    ),
                    Err(_) => {
                        into.certificate
                            .pruned(*production, PruneReason::ScheduleLegality);
                        continue;
                    }
                };
                if !into.seen.insert(candidate.canonical_key()) {
                    continue;
                }
                if let Err(reason) = constraints::admit(&candidate, expansion.constraint) {
                    into.certificate.pruned(*production, reason);
                    continue;
                }
                let bound = objective_bound(
                    &candidate,
                    expansion.remaining,
                    expansion.device,
                    expansion.objective,
                );
                if bound > 0 && bound >= *into.incumbent {
                    into.certificate
                        .pruned(*production, PruneReason::ObjectiveDominated);
                    continue;
                }
                let scored = cost::evaluate(
                    &candidate,
                    expansion.facts,
                    expansion.dependencies,
                    expansion.device,
                );
                let figure = primary_figure(&scored, expansion.device, expansion.objective);
                *into.incumbent = (*into.incumbent).min(figure);
                into.certificate.admitted(*production);
                into.candidates.push(candidate.clone());
                into.next.push(candidate);
            }
        }
    }
    false
}

/// The widest submission arrangement the device grants and the graph allows.
///
/// Concurrent queues are a submission arrangement the schedule does not express,
/// so the arrangement is selected here from the authenticated queue count and
/// the dependence analysis rather than enumerated as a separate candidate. An
/// arrangement the analysis rejects leaves the candidate on the arrangement it
/// already holds, so a graph with cross-arm hazards keeps its sequential plan
/// instead of disappearing from the search.
fn arrange(
    candidate: CandidatePlan,
    constraint: &ConstraintContext<'_>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
) -> CandidatePlan {
    let queues = constraint.device.concurrent_queues();
    if queues < 2 || candidate.topology() != ExecutionTopology::Sequential {
        return candidate;
    }
    let concurrent = candidate.with_topology(ExecutionTopology::ConcurrentQueue { queues });
    match analyze_topology_legality(
        &concurrent,
        constraint.graph,
        facts,
        dependencies,
        constraint.device,
    ) {
        TopologyDecision::Legal => concurrent,
        TopologyDecision::Rejected(_) => candidate,
    }
}

/// Whether the search may charge one more unit of work.
fn can_spend(cpu_work: u64, budget: SearchBudget) -> bool {
    cpu_work < budget.max_cpu_work && cpu_work < budget.max_elapsed_ns
}

/// The objective's primary figure for one scored candidate.
///
/// Search compares candidates on the metric the objective ranks by, not on the
/// cost model's own total, so raising an artifact-byte bound or asking for
/// throughput over a horizon changes which candidates survive rather than only
/// which one is reported first.
fn primary_figure(
    cost: &cost::CostBreakdown,
    device: DeviceFacts,
    objective: &CompileObjective,
) -> u64 {
    let horizon = objective.amortization_launches();
    let per_class = objective
        .workload()
        .as_slice()
        .iter()
        .map(|class| MetricFigures::derive(cost, device, *class, horizon))
        .collect::<Vec<_>>();
    objective
        .aggregate(&per_class)
        .get(objective.primary())
        .unwrap_or(u64::MAX)
}

/// Best primary figure this candidate or any descendant of it can reach.
///
/// Fusion is the only production that removes a generated launch, and it
/// contracts one pair at a time, so a descendant reachable within `remaining`
/// productions issues at least `groups - remaining` launches. Concurrent queues
/// and resident partitions can issue those launches together, so the bound
/// divides by the widest arrangement the device grants. The traffic, occupancy
/// and setup terms are non-negative and are left out, which keeps the figure a
/// bound rather than an estimate: a candidate eliminated against it cannot beat
/// the incumbent, whatever it is later specialized into.
///
/// Zero means this objective has no derivable lower bound on its primary metric,
/// and the search prunes nothing on it.
fn objective_bound(
    candidate: &CandidatePlan,
    remaining: u32,
    device: DeviceFacts,
    objective: &CompileObjective,
) -> u64 {
    let groups = u64::try_from(candidate.group_count()).unwrap_or(u64::MAX);
    let unavoidable_ns = groups
        .saturating_sub(u64::from(remaining))
        .max(1)
        .div_ceil(launch_divisor(device))
        .saturating_mul(cost::launch_cost_ns(device));
    let per_class = objective
        .workload()
        .as_slice()
        .iter()
        .map(|class| MetricFigures::unavoidable(unavoidable_ns, *class))
        .collect::<Vec<_>>();
    objective
        .aggregate(&per_class)
        .get(objective.primary())
        .unwrap_or(0)
}

/// Widest set of generated launches this device can issue at once.
fn launch_divisor(device: DeviceFacts) -> u64 {
    let mut divisor = 1_u64;
    if device.concurrent_queues() >= 2 {
        divisor = u64::from(device.concurrent_queues());
    }
    if device.supports_spatial_partitioning() || device.supports_cooperative_launch() {
        divisor = divisor.max(u64::from(
            grammar::PARTITION_COUNTS.iter().copied().max().unwrap_or(1),
        ));
    }
    divisor.max(1)
}
