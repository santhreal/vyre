//! What every portfolio selector decides the same way.
//!
//! Two selectors retain a set of artifacts under one objective: one partitions
//! the stated workload classes, the other subsets the proposed guards. They
//! enumerate different sets, and everything after the enumeration is the same
//! decision. Compile each member once and record its canonical byte length,
//! refuse a set over the aggregate byte ceiling, order sets by the objective's
//! metric vector, and break a tie by what the set costs to hold. Stating that
//! once keeps two selectors from drifting into two doctrines about which set is
//! better.

use std::collections::BTreeMap;

use crate::compile::{compile, compile_measured, FinalistEvaluator};
use crate::cost::CostBreakdown;
use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::objective::{CompileObjective, MetricFigures};
use crate::request::ValidatedCompileRequest;
use crate::schema::Artifact;

/// One scored retained set.
pub(crate) struct Scored<I> {
    /// Objective figures in ordering-metric order, lower being better.
    pub(crate) figures: Vec<u64>,
    /// Artifacts the set retains.
    pub(crate) retained: usize,
    /// Canonical bytes the whole set holds.
    pub(crate) aggregate_bytes: u64,
    /// Which members the set holds.
    pub(crate) identity: I,
}

impl<I: Ord> Scored<I> {
    /// Whether the objective retains this set over `other`.
    ///
    /// The metric vector decides first. A tie is broken by the smaller retained
    /// set and then the smaller aggregate byte count, because two sets the
    /// objective cannot separate are separated by what they cost to compile,
    /// load and keep. The identity is the last term, so one compile of one graph
    /// retains one set.
    pub(crate) fn retained_over(&self, other: &Self) -> bool {
        (
            &self.figures,
            self.retained,
            self.aggregate_bytes,
            &self.identity,
        ) < (
            &other.figures,
            other.retained,
            other.aggregate_bytes,
            &other.identity,
        )
    }
}

/// Compile one request and record the canonical byte length of its artifact.
///
/// The byte length is a ranking term and an aggregate bound is checked against
/// it, so it is the serialized length rather than an estimate of one.
///
/// # Errors
///
/// Returns the compile error of the request, or the serialization error of its
/// artifact.
pub(crate) fn compile_artifact_bytes(
    request: &ValidatedCompileRequest,
    evaluator: Option<&dyn FinalistEvaluator>,
) -> Result<(Artifact, u64), CompileError> {
    let artifact = match evaluator {
        Some(evaluator) => compile_measured(request, evaluator)?,
        None => compile(request)?,
    };
    let bytes = u64::try_from(artifact.to_bytes()?.len()).unwrap_or(u64::MAX);
    Ok((artifact, bytes))
}

/// The refusal a set over the objective's aggregate byte ceiling earns, when it
/// is over one.
///
/// `fix` states the correction in the terms of the selector that enumerated the
/// set, because a caller who stated workload classes and one who proposed guards
/// change different inputs to get under the same bound.
pub(crate) fn aggregate_bytes_refusal(
    objective: &CompileObjective,
    aggregate_bytes: u64,
    fix: &'static str,
) -> Option<CompileError> {
    let ceiling = objective.portfolio().max_aggregate_bytes()?;
    if aggregate_bytes <= ceiling {
        return None;
    }
    Some(failure(
        CompilerFailureKind::ObjectiveBoundViolated,
        "request.objective.portfolio.max_aggregate_bytes",
        format!(
            "aggregate artifact bound is {ceiling} bytes and the retained set holds {aggregate_bytes} bytes"
        ),
        fix,
    ))
}

/// Compile every member of one candidate set that is not compiled already.
///
/// A member that refuses does not fail the search: the first refusal is recorded
/// and the set is skipped, because another set may be legal and the empty set
/// always is. The refusal is reported only if no set at all was retained, which
/// is what makes it the diagnostic a caller acts on.
///
/// Returns whether every member compiled.
pub(crate) fn compile_members<K, V>(
    members: &[K],
    compiled: &mut BTreeMap<K, V>,
    refused: &mut Option<CompileError>,
    mut build: impl FnMut(&K) -> Result<V, CompileError>,
) -> bool
where
    K: Clone + Ord,
{
    for member in members {
        if compiled.contains_key(member) {
            continue;
        }
        match build(member) {
            Ok(value) => {
                compiled.insert(member.clone(), value);
            }
            Err(error) => {
                if refused.is_none() {
                    *refused = Some(error);
                }
                return false;
            }
        }
    }
    true
}

/// Every ordering figure of one candidate set under the stated objective.
///
/// `cost_of_class` states which member serves each stated workload class, in the
/// order the profile states them, because a set that assigns two classes to two
/// artifacts is ranked on what each class actually runs. The objective's hard
/// bounds are checked here, so a set that violates one is refused with the bound
/// it broke instead of being ranked against sets that stayed inside it.
///
/// # Errors
///
/// Returns an objective-bound violation naming the first bound the aggregate
/// figures exceed.
pub(crate) fn ranked_figures(
    request: &ValidatedCompileRequest,
    objective: &CompileObjective,
    fix: &'static str,
    cost_of_class: impl Fn(usize) -> CostBreakdown,
) -> Result<Vec<u64>, CompileError> {
    let per_class = objective
        .workload()
        .as_slice()
        .iter()
        .enumerate()
        .map(|(index, class)| {
            MetricFigures::derive(
                &cost_of_class(index),
                request.device(),
                *class,
                objective.amortization_launches(),
            )
        })
        .collect::<Vec<_>>();
    let aggregate = objective.aggregate(&per_class);
    if let Some(violation) = objective.bounds().first_violation(aggregate.as_array()) {
        return Err(failure(
            CompilerFailureKind::ObjectiveBoundViolated,
            "request.objective.bounds",
            violation.statement(),
            fix,
        ));
    }
    Ok(objective
        .ordering_metrics()
        .into_iter()
        .map(|metric| aggregate.get(metric).unwrap_or(u64::MAX))
        .collect())
}
