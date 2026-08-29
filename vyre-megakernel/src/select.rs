//! Scoring, the legal Pareto frontier, and the order the objective states.
//!
//! Ranking is two decisions, not one. The first is which candidates are worth
//! keeping at all: a candidate no better than another on every metric the
//! objective reads cannot win under that objective, whatever it is later
//! measured at, so it is dominated and never measured. The second is the order
//! among the rest, which is the objective's primary metric, then its tie
//! breakers, then a content order so one compilation of one graph selects one
//! plan.
//!
//! A hard bound is neither of those. A candidate that exceeds one is rejected
//! rather than ranked last, and a compile whose whole legal set exceeds one
//! fails with the bound it came nearest to meeting.

use crate::{
    candidate::CandidatePlan,
    certificate::{PruneReason, SearchCertificate},
    cost::{evaluate, CostBreakdown},
    facts::PlanningFacts,
    grammar::ScheduleProduction,
    objective::{BoundViolation, CompileObjective, MetricFigures},
    DependencyEdge, DeviceFacts, RequiredSchedule,
};

#[derive(Debug)]
pub(crate) struct Selection {
    pub(crate) candidate: CandidatePlan,
    pub(crate) cost: CostBreakdown,
    /// One figure per ordering metric, in
    /// [`CompileObjective::ordering_metrics`] order, aggregated over the
    /// objective's workload classes.
    pub(crate) figures: Vec<u64>,
    /// Whether no other admitted candidate is at least as good on every
    /// ordering metric and better on one.
    pub(crate) on_frontier: bool,
}

/// Every candidate the objective admits, best first, and the bound it refused
/// the rest for.
#[derive(Debug)]
pub(crate) struct Ranking {
    /// Admitted candidates in objective order.
    pub(crate) admitted: Vec<Selection>,
    /// Bound the whole legal set exceeded, when nothing was admitted.
    pub(crate) refused: Option<BoundViolation>,
    /// Schedule family the caller required that no legal candidate exercised.
    pub(crate) unreachable_schedule: Option<RequiredSchedule>,
}

/// Score, bound, and order every candidate under `objective`.
///
/// Ordering is the objective's metric vector, then the number of grammar
/// productions the candidate applied, then the group vector, the proposed launch
/// width and the topology. Derivation length comes before content, so a
/// production that does not pay for itself never displaces the baseline, and two
/// candidates that tie on every metric are ordered by content.
///
/// A stated `required` family is enforced here rather than in derivation, so
/// every candidate is still derived and every legality decision is still made
/// and recorded. A candidate outside the family is eliminated with
/// [`PruneReason::ScheduleRequirement`], and a family no candidate exercises is
/// reported instead of quietly replaced by the family that ranked next.
pub(crate) fn rank(
    candidates: Vec<CandidatePlan>,
    facts: &PlanningFacts,
    dependencies: &[DependencyEdge],
    device: DeviceFacts,
    objective: &CompileObjective,
    certificate: &mut SearchCertificate,
    required: Option<RequiredSchedule>,
) -> Ranking {
    let classes = objective.workload().as_slice();
    let horizon = objective.amortization_launches();
    let ordering = objective.ordering_metrics();
    let mut refused: Option<BoundViolation> = None;
    let mut admitted = Vec::with_capacity(candidates.len());
    let mut satisfying = 0_usize;
    for candidate in candidates {
        if let Some(required) = required {
            if !required.admits(&candidate.derivation) {
                certificate.pruned(charged_to(&candidate), PruneReason::ScheduleRequirement);
                continue;
            }
        }
        satisfying += 1;
        let cost = evaluate(&candidate, facts, dependencies, device);
        let per_class = classes
            .iter()
            .map(|class| MetricFigures::derive(&cost, device, *class, horizon))
            .collect::<Vec<_>>();
        let aggregate = objective.aggregate(&per_class);
        if let Some(violation) = objective.bounds().first_violation(aggregate.as_array()) {
            certificate.pruned(charged_to(&candidate), PruneReason::ObjectiveBound);
            refused = Some(match refused {
                Some(previous) => tightest(previous, violation),
                None => violation,
            });
            continue;
        }
        let figures = ordering
            .iter()
            .map(|metric| aggregate.get(*metric).unwrap_or(u64::MAX))
            .collect();
        admitted.push(Selection {
            candidate,
            cost,
            figures,
            on_frontier: false,
        });
    }

    mark_frontier(&mut admitted);
    admitted.sort_by(|left, right| {
        left.figures
            .cmp(&right.figures)
            .then_with(|| {
                left.candidate
                    .derivation
                    .len()
                    .cmp(&right.candidate.derivation.len())
            })
            .then_with(|| left.candidate.node_groups.cmp(&right.candidate.node_groups))
            .then_with(|| {
                left.candidate
                    .workgroup_width
                    .cmp(&right.candidate.workgroup_width)
            })
            .then_with(|| left.candidate.topology.cmp(&right.candidate.topology))
    });
    let unreachable_schedule = match required {
        Some(required) if satisfying == 0 => Some(required),
        _ => None,
    };
    let refused = if admitted.is_empty() { refused } else { None };
    Ranking {
        admitted,
        refused,
        unreachable_schedule,
    }
}

/// Production the elimination of one candidate is charged to.
///
/// The last step that derived the plan is what put it over the bound. A plan
/// with no derivation is the baseline, which is charged to the fusion family it
/// would have been contracted by, so the certificate never reports an
/// elimination against no family at all.
fn charged_to(candidate: &CandidatePlan) -> ScheduleProduction {
    candidate
        .derivation
        .last()
        .map_or(ScheduleProduction::Fusion, |step| step.production)
}

/// The violation a caller is told about: the one whose achieved figure is
/// closest to its own limit, so the diagnostic names the bound the workload came
/// nearest to meeting instead of whichever candidate was scored first.
fn tightest(left: BoundViolation, right: BoundViolation) -> BoundViolation {
    let overshoot = |violation: &BoundViolation| violation.achieved.saturating_sub(violation.limit);
    if overshoot(&right) < overshoot(&left) {
        right
    } else {
        left
    }
}

/// Mark every candidate no other candidate dominates.
///
/// One candidate dominates another when it is at least as good on every
/// ordering metric and strictly better on one. The comparison is quadratic in
/// the candidate count, which the search budget already bounds.
fn mark_frontier(scored: &mut [Selection]) {
    for index in 0..scored.len() {
        let dominated = scored.iter().enumerate().any(|(other, candidate)| {
            other != index && dominates(&candidate.figures, &scored[index].figures)
        });
        scored[index].on_frontier = !dominated;
    }
}

/// Whether `left` is at least as good as `right` everywhere and better once.
fn dominates(left: &[u64], right: &[u64]) -> bool {
    let mut strictly_better = false;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        if left_value > right_value {
            return false;
        }
        if left_value < right_value {
            strictly_better = true;
        }
    }
    strictly_better
}
