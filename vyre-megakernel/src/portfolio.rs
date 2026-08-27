//! Joint selection of the artifact set one objective retains.
//!
//! One compile of one graph does not always retain one artifact. A profile that
//! states an interactive submission and a thousand-launch batch is not served by
//! one schedule, and an objective whose coverage policy is
//! [`CoveragePolicy::EveryWorkloadClass`](crate::CoveragePolicy::EveryWorkloadClass)
//! states that. Retaining a set is one decision rather than several independent
//! ones: optimizing every class on its own maximizes the retained set, and every
//! retained artifact costs compile work, bytes, and load time the objective
//! bounds in aggregate.
//!
//! Selection enumerates every partition of the stated classes the objective
//! admits, compiles each part once under the objective narrowed to that part,
//! and orders whole partitions under the objective read over every stated class.
//! A profile holds at most [`WorkloadProfile::CAPACITY`] classes, so the
//! enumeration is exhaustive: the retained set is the best legal set under the
//! stated objective, not the first set a greedy merge reached.

use std::collections::BTreeMap;

use crate::compile::{compile, compile_measured, FinalistEvaluator};
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::objective::{
    CompileObjective, MetricFigures, ObjectiveMetric, PortfolioPolicy, WorkloadClass,
    WorkloadProfile,
};
use crate::request::ValidatedCompileRequest;
use crate::schema::Artifact;

/// The artifacts one compile retains, and which artifact serves each workload
/// class.
///
/// The assignment is what makes the set usable: a runtime holding three
/// artifacts and no record of which submission arrangement each was selected for
/// has to guess, and a guess is how a batch runs the interactive schedule.
#[derive(Clone, Debug)]
pub struct ArtifactPortfolio {
    artifacts: Vec<Artifact>,
    assignment: Vec<u32>,
    objective: CompileObjective,
    aggregate_bytes: u64,
}

impl ArtifactPortfolio {
    /// Retained artifacts, indexed by [`Self::assignment`].
    #[must_use]
    pub fn artifacts(&self) -> &[Artifact] {
        &self.artifacts
    }

    /// Retained artifact count.
    #[must_use]
    pub fn variants(&self) -> u32 {
        u32::try_from(self.artifacts.len()).unwrap_or(u32::MAX)
    }

    /// Retained artifact index per stated workload class, indexed like
    /// [`WorkloadProfile::as_slice`].
    #[must_use]
    pub fn assignment(&self) -> &[u32] {
        &self.assignment
    }

    /// Artifact serving the workload class at `class_index`.
    #[must_use]
    pub fn artifact_for_class(&self, class_index: usize) -> Option<&Artifact> {
        let artifact = *self.assignment.get(class_index)? as usize;
        self.artifacts.get(artifact)
    }

    /// Canonical serialized bytes every retained artifact holds together.
    #[must_use]
    pub const fn aggregate_bytes(&self) -> u64 {
        self.aggregate_bytes
    }

    /// Objective the set was selected under.
    #[must_use]
    pub const fn objective(&self) -> &CompileObjective {
        &self.objective
    }
}

/// Compile the artifact set the stated objective retains, ranked by the open
/// cost model alone.
///
/// Every partition of the stated workload classes the coverage policy and the
/// variant bounds admit is scored, and the partition the objective orders first
/// is retained. A compile that budgets on-device measurements is rejected here
/// for the same reason [`compile`] rejects one: only
/// [`compile_portfolio_measured`] can spend that budget.
///
/// # Errors
///
/// Returns an error when no admitted partition compiles, when every admitted
/// partition exceeds a hard bound, or when the variant bounds retain nothing.
pub fn compile_portfolio(
    request: &ValidatedCompileRequest,
) -> Result<ArtifactPortfolio, CompileError> {
    select(request, None)
}

/// Compile the retained artifact set with each part's finalists emitted for the
/// target and timed on the device.
///
/// Each part is compiled through [`compile_measured`], so the measurement budget
/// the request states is spent per retained artifact rather than once for the
/// whole set: two parts optimize for different arrangements and a measurement of
/// one says nothing about the other.
///
/// # Errors
///
/// Returns an error under the same conditions as [`compile_portfolio`], and when
/// a part cannot be emitted or timed on the device.
pub fn compile_portfolio_measured(
    request: &ValidatedCompileRequest,
    evaluator: &dyn FinalistEvaluator,
) -> Result<ArtifactPortfolio, CompileError> {
    select(request, Some(evaluator))
}

/// One scored partition of the stated workload classes.
struct Scored {
    figures: Vec<u64>,
    parts: u32,
    aggregate_bytes: u64,
    assignment: Vec<u32>,
}

fn select(
    request: &ValidatedCompileRequest,
    evaluator: Option<&dyn FinalistEvaluator>,
) -> Result<ArtifactPortfolio, CompileError> {
    let objective = *request.objective();
    let classes = objective.workload().as_slice().to_vec();
    let limit = variant_limit(&objective);
    let minimum = objective
        .portfolio()
        .coverage()
        .minimum_variants(classes.len());
    if limit < minimum {
        return Err(failure(
            CompilerFailureKind::ObjectiveBoundViolated,
            "request.objective.bounds.variant_count",
            format!(
                "variant-count bound retains {limit} artifacts and coverage `{}` over {} workload classes needs {minimum}",
                objective.portfolio().coverage().name(),
                classes.len()
            ),
            "raise the variant-count bound, or state a coverage policy that retains fewer artifacts",
        ));
    }
    let mut compiled: BTreeMap<u16, (Artifact, u64)> = BTreeMap::new();
    let mut refused: Option<CompileError> = None;
    let mut best: Option<Scored> = None;
    for assignment in partitions(classes.len(), limit) {
        let parts = part_count(&assignment);
        if !objective.portfolio().admits(parts, classes.len()) {
            continue;
        }
        let masks = (0..parts)
            .map(|part| part_mask(&assignment, part))
            .collect::<Vec<_>>();
        let mut compilable = true;
        for mask in &masks {
            if compiled.contains_key(mask) {
                continue;
            }
            match compile_part(request, &objective, &classes, *mask, evaluator) {
                Ok(part) => {
                    compiled.insert(*mask, part);
                }
                Err(error) => {
                    refused = refused.or(Some(error));
                    compilable = false;
                    break;
                }
            }
        }
        if !compilable {
            continue;
        }
        let aggregate_bytes = masks
            .iter()
            .map(|mask| compiled[mask].1)
            .fold(0_u64, u64::saturating_add);
        if let Some(ceiling) = objective.portfolio().max_aggregate_bytes() {
            if aggregate_bytes > ceiling {
                refused = refused.or_else(|| {
                    Some(failure(
                        CompilerFailureKind::ObjectiveBoundViolated,
                        "request.objective.portfolio.max_aggregate_bytes",
                        format!(
                            "aggregate artifact bound is {ceiling} bytes and the retained set holds {aggregate_bytes} bytes"
                        ),
                        "raise the aggregate byte bound, or state a coverage policy that retains fewer artifacts",
                    ))
                });
                continue;
            }
        }
        let per_class = classes
            .iter()
            .enumerate()
            .map(|(index, class)| {
                let mask = part_mask(&assignment, assignment[index]);
                let cost = &compiled[&mask].0.selected_plan().selection_cost;
                MetricFigures::derive(
                    cost,
                    request.device(),
                    *class,
                    objective.amortization_launches(),
                )
            })
            .collect::<Vec<_>>();
        let aggregate = objective.aggregate(&per_class);
        if let Some(violation) = objective.bounds().first_violation(aggregate.as_array()) {
            refused = refused.or_else(|| {
                Some(failure(
                    CompilerFailureKind::ObjectiveBoundViolated,
                    "request.objective.bounds",
                    violation.statement(),
                    "raise the bound the objective states, or state a coverage policy that retains a set within it",
                ))
            });
            continue;
        }
        let figures = objective
            .ordering_metrics()
            .into_iter()
            .map(|metric| aggregate.get(metric).unwrap_or(u64::MAX))
            .collect::<Vec<_>>();
        let scored = Scored {
            figures,
            parts,
            aggregate_bytes,
            assignment,
        };
        if best.as_ref().is_none_or(|held| better(&scored, held)) {
            best = Some(scored);
        }
    }
    let selected = best.ok_or_else(|| {
        refused.unwrap_or_else(|| {
            failure(
                CompilerFailureKind::PortfolioCoverageUnsatisfied,
                "request.objective.portfolio",
                "no partition of the stated workload classes was enumerated",
                "report the compiler defect",
            )
        })
    })?;
    let mut artifacts = Vec::with_capacity(selected.parts as usize);
    for part in 0..selected.parts {
        let mask = part_mask(&selected.assignment, part);
        artifacts.push(compiled[&mask].0.clone());
    }
    Ok(ArtifactPortfolio {
        artifacts,
        assignment: selected.assignment,
        objective,
        aggregate_bytes: selected.aggregate_bytes,
    })
}

/// Whether `left` is the set the objective retains over `right`.
///
/// The objective's metric vector decides first. A tie is broken by the smaller
/// retained set and then the smaller aggregate byte count, because two sets the
/// objective cannot separate are separated by what they cost to compile, load,
/// and keep. The assignment is the last term, so one compile of one graph
/// retains one set.
fn better(left: &Scored, right: &Scored) -> bool {
    (
        &left.figures,
        left.parts,
        left.aggregate_bytes,
        &left.assignment,
    ) < (
        &right.figures,
        right.parts,
        right.aggregate_bytes,
        &right.assignment,
    )
}

/// Compile one part of the partition under the objective narrowed to it.
fn compile_part(
    request: &ValidatedCompileRequest,
    objective: &CompileObjective,
    classes: &[WorkloadClass],
    mask: u16,
    evaluator: Option<&dyn FinalistEvaluator>,
) -> Result<(Artifact, u64), CompileError> {
    let narrowed = request.restated(part_objective(objective, classes, mask));
    let artifact = match evaluator {
        Some(evaluator) => compile_measured(&narrowed, evaluator)?,
        None => compile(&narrowed)?,
    };
    let bytes = u64::try_from(artifact.to_bytes()?.len()).unwrap_or(u64::MAX);
    Ok((artifact, bytes))
}

/// The objective one part of the partition is compiled under.
///
/// The part optimizes for its own classes alone, with their weights restated to
/// sum to one thousand permille so a weighted aggregate over a subset means what
/// it means over the whole profile. Its coverage is one artifact, because the
/// part is exactly the set of classes one artifact serves.
fn part_objective(
    objective: &CompileObjective,
    classes: &[WorkloadClass],
    mask: u16,
) -> CompileObjective {
    let members = classes
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1 << index) != 0)
        .map(|(_, class)| *class)
        .collect::<Vec<_>>();
    let total = members
        .iter()
        .fold(0_u32, |sum, class| sum + u32::from(class.weight_permille));
    let mut profile: Option<WorkloadProfile> = None;
    let mut assigned = 0_u32;
    for (position, class) in members.iter().enumerate() {
        let weight = if position + 1 == members.len() {
            1_000_u32.saturating_sub(assigned)
        } else if total == 0 {
            0
        } else {
            u32::from(class.weight_permille) * 1_000 / total
        };
        assigned += weight;
        let restated = WorkloadClass::new(
            class.launch_batch,
            class.concurrent_streams,
            u16::try_from(weight).unwrap_or(u16::MAX),
        );
        profile = Some(match profile {
            None => WorkloadProfile::of(restated),
            Some(existing) => existing.pushed(restated),
        });
    }
    let profile = profile
        .unwrap_or_else(WorkloadProfile::single)
        .with_aggregation(objective.workload().aggregation());
    objective
        .with_workload(profile)
        .with_portfolio(retained_bytes_policy(objective))
}

/// One-artifact policy for a part, keeping any aggregate byte ceiling the whole
/// set is held to.
fn retained_bytes_policy(objective: &CompileObjective) -> PortfolioPolicy {
    let policy = PortfolioPolicy::single();
    match objective.portfolio().max_aggregate_bytes() {
        Some(bytes) => policy.with_max_aggregate_bytes(bytes),
        None => policy,
    }
}

/// Artifacts the objective's own bounds allow the retained set to hold.
fn variant_limit(objective: &CompileObjective) -> usize {
    let stated = u64::from(objective.portfolio().max_variants());
    let bounded = objective
        .bounds()
        .limit(ObjectiveMetric::VariantCount)
        .unwrap_or(u64::MAX);
    let capacity = WorkloadProfile::CAPACITY as u64;
    usize::try_from(stated.min(bounded).min(capacity)).unwrap_or(WorkloadProfile::CAPACITY)
}

/// Bit mask of the classes assigned to `part`.
fn part_mask(assignment: &[u32], part: u32) -> u16 {
    assignment
        .iter()
        .enumerate()
        .filter(|(_, assigned)| **assigned == part)
        .fold(0_u16, |mask, (index, _)| mask | (1 << index))
}

/// Parts the assignment uses.
fn part_count(assignment: &[u32]) -> u32 {
    assignment.iter().copied().max().map_or(0, |last| last + 1)
}

/// Every partition of `classes` into at most `max_parts` parts.
///
/// Assignments are restricted growth strings: a class joins a part already in
/// use or opens the next one, so each partition is enumerated once instead of
/// once per relabelling of its parts.
fn partitions(classes: usize, max_parts: usize) -> Vec<Vec<u32>> {
    let mut enumerated = Vec::new();
    let mut current = vec![0_u32; classes];
    extend(0, 0, classes, max_parts, &mut current, &mut enumerated);
    enumerated
}

fn extend(
    index: usize,
    used: u32,
    classes: usize,
    max_parts: usize,
    current: &mut Vec<u32>,
    enumerated: &mut Vec<Vec<u32>>,
) {
    if index == classes {
        if used > 0 {
            enumerated.push(current.clone());
        }
        return;
    }
    for part in 0..used {
        current[index] = part;
        extend(index + 1, used, classes, max_parts, current, enumerated);
    }
    if (used as usize) < max_parts {
        current[index] = used;
        extend(index + 1, used + 1, classes, max_parts, current, enumerated);
    }
}
