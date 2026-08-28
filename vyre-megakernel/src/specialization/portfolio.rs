//! Joint compilation of one guarded variant set and its remainder.
//!
//! A specialization is not free. Each variant costs compile work, artifact
//! bytes, load time, and cache residency, and the input it was compiled for may
//! be a thousandth of the traffic. So the set is chosen jointly: every subset of
//! the proposed guards the variant bound admits is compiled and scored, and the
//! set the objective orders first is retained. The empty subset is always among
//! them, so the unspecialized baseline stays in the candidate set and a
//! specialization that buys less than it costs is not retained.
//!
//! Scoring reads the domain, not one input. Each retained artifact serves a
//! known number of cells of the coverage proof, so the objective figure of the
//! set is each artifact's figure weighted by the part of the domain it serves.
//! A variant that improves only its own cells improves the set by exactly that
//! much.
//!
//! A range-guarded variant is compiled at the largest extent its guard admits,
//! so the schedule it receives launches over every value the guard admits and a
//! shorter input is covered rather than truncated. A guard that admits only
//! multiples of a tile leaves the tail to another guard or to the remainder, and
//! the coverage proof is what shows the tail is served.

use std::collections::BTreeMap;

use super::axis::{AxisValue, SpecializationAxis};
use super::guard::VariantGuard;
use super::{precedence_order, CoverageProof, RemainderKind, SpecializationContract};
use crate::compile::{compile, compile_measured, FinalistEvaluator};
use crate::error::{failure, CompilerFailureKind};
use crate::objective::{CompileObjective, MetricFigures, WorkloadClass};
use crate::request::ValidatedCompileRequest;
use crate::schema::Artifact;
use crate::CompileError;

/// Guards one call may propose.
///
/// Every subset is scored, so the proposal count bounds the search rather than
/// the retained set. A caller with more candidate shapes than this states a
/// coarser contract instead of asking for an exponential search.
pub const MAX_PROPOSED_VARIANTS: usize = 8;

/// One compiled variant and what selects it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioVariant {
    guard: VariantGuard,
    artifact: Artifact,
    bytes: u64,
    served_cells: usize,
}

impl PortfolioVariant {
    /// What selects this variant.
    #[must_use]
    pub const fn guard(&self) -> &VariantGuard {
        &self.guard
    }

    /// The compiled artifact.
    #[must_use]
    pub const fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// Canonical artifact byte length.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Cells of the coverage proof this variant serves.
    #[must_use]
    pub const fn served_cells(&self) -> usize {
        self.served_cells
    }
}

/// What serves the facts no guard admits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecializedRemainder {
    /// One artifact compiled without any guard's assumptions.
    ///
    /// It is retained even when the guards cover the declared domain, because a
    /// consumer can state facts outside that domain and something correct has to
    /// serve them.
    Generic {
        /// The unspecialized artifact.
        artifact: Artifact,
        /// Canonical artifact byte length.
        bytes: u64,
    },
    /// Facts no guard admits are rejected.
    Unsupported,
}

impl SpecializedRemainder {
    /// The artifact this remainder serves with, when it serves with one.
    #[must_use]
    pub const fn artifact(&self) -> Option<&Artifact> {
        match self {
            Self::Generic { artifact, .. } => Some(artifact),
            Self::Unsupported => None,
        }
    }

    /// Bytes this remainder contributes to the retained set.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        match self {
            Self::Generic { bytes, .. } => *bytes,
            Self::Unsupported => 0,
        }
    }

    /// Which remainder this is.
    #[must_use]
    pub const fn kind(&self) -> RemainderKind {
        match self {
            Self::Generic { .. } => RemainderKind::Generic,
            Self::Unsupported => RemainderKind::Unsupported,
        }
    }
}

/// The guarded artifact set one compile retains, and what it proves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecializedPortfolio {
    contract: SpecializationContract,
    variants: Vec<PortfolioVariant>,
    remainder: SpecializedRemainder,
    proof: CoverageProof,
    objective: CompileObjective,
    aggregate_bytes: u64,
}

impl SpecializedPortfolio {
    /// The contract the retained set was selected under.
    #[must_use]
    pub const fn contract(&self) -> &SpecializationContract {
        &self.contract
    }

    /// The retained variants, in canonical guard order.
    #[must_use]
    pub fn variants(&self) -> &[PortfolioVariant] {
        &self.variants
    }

    /// What serves the facts no guard admits.
    #[must_use]
    pub const fn remainder(&self) -> &SpecializedRemainder {
        &self.remainder
    }

    /// The coverage and exclusivity proof the retained guard set passed.
    #[must_use]
    pub const fn proof(&self) -> &CoverageProof {
        &self.proof
    }

    /// The objective the set was retained under.
    #[must_use]
    pub const fn objective(&self) -> &CompileObjective {
        &self.objective
    }

    /// Bytes the whole retained set holds.
    #[must_use]
    pub const fn aggregate_bytes(&self) -> u64 {
        self.aggregate_bytes
    }

    /// The artifact that serves one complete set of stated facts.
    ///
    /// The facts are checked against the declared domain first, so a workload
    /// the contract never covered is refused rather than served by the
    /// remainder, which was compiled for the domain. Guards are then evaluated
    /// in precedence order and the first that admits the facts serves them.
    /// Nothing here alters a schedule: selection returns an artifact that was
    /// compiled and scored before it was retained.
    ///
    /// # Errors
    ///
    /// Returns an error when the facts fall outside the declared domain, and
    /// when no guard admits them and the remainder is
    /// [`SpecializedRemainder::Unsupported`].
    pub fn select(
        &self,
        facts: &BTreeMap<SpecializationAxis, AxisValue>,
    ) -> Result<&Artifact, CompileError> {
        self.contract.admits_facts(facts)?;
        let guards = self
            .variants
            .iter()
            .map(|variant| variant.guard.clone())
            .collect::<Vec<_>>();
        for index in precedence_order(&guards) {
            if self.variants[index].guard.admits_facts(facts)? {
                return Ok(&self.variants[index].artifact);
            }
        }
        self.remainder.artifact().ok_or_else(|| {
            failure(
                CompilerFailureKind::UnsupportedWorkload,
                "specialization.remainder",
                "no retained variant admits the stated facts and the remainder is unsupported",
                "state facts inside a guard's domain, or compile a portfolio with a generic remainder",
            )
        })
    }
}

/// Compile the guarded artifact set the objective retains, ranked by the open
/// cost model alone.
///
/// # Errors
///
/// Returns an error when the contract names a fact the request cannot produce,
/// when a proposal is not a valid guard under the contract, when no admitted
/// subset compiles within the objective's bounds, or when the remainder is
/// declared unsupported and no admitted subset covers the domain.
pub fn compile_specialized_portfolio(
    request: &ValidatedCompileRequest,
    contract: &SpecializationContract,
    proposals: &[VariantGuard],
    remainder: RemainderKind,
) -> Result<SpecializedPortfolio, CompileError> {
    select(request, contract, proposals, remainder, None)
}

/// Compile the guarded artifact set with each retained variant's finalists
/// emitted for the target and timed on the device.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`compile_specialized_portfolio`], and when a variant cannot be emitted or
/// timed on the device.
pub fn compile_specialized_portfolio_measured(
    request: &ValidatedCompileRequest,
    contract: &SpecializationContract,
    proposals: &[VariantGuard],
    remainder: RemainderKind,
    evaluator: &dyn FinalistEvaluator,
) -> Result<SpecializedPortfolio, CompileError> {
    select(request, contract, proposals, remainder, Some(evaluator))
}

/// One scored subset of the proposed guards.
struct Scored {
    figures: Vec<u64>,
    variants: usize,
    aggregate_bytes: u64,
    guards: Vec<VariantGuard>,
    proof: CoverageProof,
}

fn select(
    request: &ValidatedCompileRequest,
    contract: &SpecializationContract,
    proposals: &[VariantGuard],
    remainder: RemainderKind,
    evaluator: Option<&dyn FinalistEvaluator>,
) -> Result<SpecializedPortfolio, CompileError> {
    contract.validate_for(request)?;
    if proposals.len() > MAX_PROPOSED_VARIANTS {
        return Err(failure(
            CompilerFailureKind::InvalidVariantGuard,
            "specialization.proposals",
            format!(
                "{} guards were proposed and every subset is scored, above the bound of {MAX_PROPOSED_VARIANTS}",
                proposals.len()
            ),
            "state a coarser contract so fewer guards cover the same domain",
        ));
    }
    let mut stated = proposals.to_vec();
    stated.sort();
    stated.dedup();
    for guard in &stated {
        contract.validate_guard(guard)?;
    }
    let objective = *request.objective();
    let limit = usize::try_from(objective.portfolio().max_variants()).unwrap_or(usize::MAX);
    let generic = match remainder {
        RemainderKind::Generic => {
            let (artifact, bytes) = build(request, evaluator)?;
            SpecializedRemainder::Generic { artifact, bytes }
        }
        RemainderKind::Unsupported => SpecializedRemainder::Unsupported,
    };
    let generic_figures = match generic.artifact() {
        Some(artifact) => Some(ordering_figures(request, &objective, artifact)?),
        None => None,
    };
    let mut compiled: BTreeMap<VariantGuard, (Artifact, u64, Vec<u64>)> = BTreeMap::new();
    let mut refused: Option<CompileError> = None;
    let mut best: Option<Scored> = None;
    for mask in 0..(1_u32 << stated.len()) {
        let guards = subset(&stated, mask);
        if guards.len() > limit {
            continue;
        }
        let proof = match contract.prove(&guards, remainder) {
            Ok(proof) => proof,
            Err(error) => {
                refused = refused.or(Some(error));
                continue;
            }
        };
        let mut compilable = true;
        for guard in &guards {
            if compiled.contains_key(guard) {
                continue;
            }
            match build(&narrow(request, contract, guard)?, evaluator) {
                Ok((artifact, bytes)) => {
                    let figures = ordering_figures(request, &objective, &artifact)?;
                    compiled.insert(guard.clone(), (artifact, bytes, figures));
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
        let aggregate_bytes = guards
            .iter()
            .map(|guard| compiled[guard].1)
            .fold(generic.bytes(), u64::saturating_add);
        if let Some(ceiling) = objective.portfolio().max_aggregate_bytes() {
            if aggregate_bytes > ceiling {
                refused = refused.or_else(|| {
                    Some(failure(
                        CompilerFailureKind::ObjectiveBoundViolated,
                        "request.objective.portfolio.max_aggregate_bytes",
                        format!(
                            "aggregate artifact bound is {ceiling} bytes and the retained set holds {aggregate_bytes} bytes"
                        ),
                        "raise the aggregate byte bound, or propose guards that retain fewer variants",
                    ))
                });
                continue;
            }
        }
        let Some(figures) = weighted(&guards, &compiled, &proof, generic_figures.as_ref()) else {
            continue;
        };
        let scored = Scored {
            figures,
            variants: guards.len(),
            aggregate_bytes,
            guards,
            proof,
        };
        if best.as_ref().is_none_or(|held| better(&scored, held)) {
            best = Some(scored);
        }
    }
    let selected = best.ok_or_else(|| {
        refused.unwrap_or_else(|| {
            failure(
                CompilerFailureKind::PortfolioCoverageUnsatisfied,
                "specialization.proposals",
                "no proposed guard subset was retained",
                "report the compiler defect",
            )
        })
    })?;
    let variants = selected
        .guards
        .iter()
        .enumerate()
        .map(|(index, guard)| {
            let (artifact, bytes, _) = &compiled[guard];
            PortfolioVariant {
                guard: guard.clone(),
                artifact: artifact.clone(),
                bytes: *bytes,
                served_cells: selected.proof.served()[index],
            }
        })
        .collect();
    Ok(SpecializedPortfolio {
        contract: contract.clone(),
        variants,
        remainder: generic,
        proof: selected.proof,
        objective,
        aggregate_bytes: selected.aggregate_bytes,
    })
}

/// Whether `left` is the set the objective retains over `right`.
///
/// The objective's metric vector decides first. A tie is broken by the smaller
/// retained set and then the smaller aggregate byte count, so a variant the
/// objective cannot distinguish from its absence is not retained. The guard set
/// is the last term, so one compile of one graph retains one set.
fn better(left: &Scored, right: &Scored) -> bool {
    (
        &left.figures,
        left.variants,
        left.aggregate_bytes,
        &left.guards,
    ) < (
        &right.figures,
        right.variants,
        right.aggregate_bytes,
        &right.guards,
    )
}

/// The objective figure of one retained set: each artifact's figure weighted by
/// the part of the domain it serves.
fn weighted(
    guards: &[VariantGuard],
    compiled: &BTreeMap<VariantGuard, (Artifact, u64, Vec<u64>)>,
    proof: &CoverageProof,
    generic: Option<&Vec<u64>>,
) -> Option<Vec<u64>> {
    let cells = u64::try_from(proof.cells()).ok()?;
    if cells == 0 {
        return None;
    }
    let width = generic
        .map(Vec::len)
        .or_else(|| guards.first().map(|guard| compiled[guard].2.len()))?;
    let mut totals = vec![0_u64; width];
    for (index, guard) in guards.iter().enumerate() {
        let weight = u64::try_from(proof.served()[index]).ok()?;
        for (total, figure) in totals.iter_mut().zip(&compiled[guard].2) {
            *total = total.saturating_add(figure.saturating_mul(weight));
        }
    }
    let gaps = u64::try_from(proof.gaps()).ok()?;
    if let Some(generic) = generic {
        for (total, figure) in totals.iter_mut().zip(generic) {
            *total = total.saturating_add(figure.saturating_mul(gaps));
        }
    } else if gaps > 0 {
        return None;
    }
    Some(totals.into_iter().map(|total| total / cells).collect())
}

/// Compile one request and record its canonical byte length.
fn build(
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

/// Every ordering figure of one artifact under the stated objective.
///
/// Hard bounds are checked per artifact rather than over the set, because a
/// retained artifact that exceeds a bound is a launch nobody can afford however
/// small a part of the domain it serves.
fn ordering_figures(
    request: &ValidatedCompileRequest,
    objective: &CompileObjective,
    artifact: &Artifact,
) -> Result<Vec<u64>, CompileError> {
    let cost = &artifact.selected_plan().selection_cost;
    let per_class = objective
        .workload()
        .as_slice()
        .iter()
        .map(|class: &WorkloadClass| {
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
        return Err(failure(
            CompilerFailureKind::ObjectiveBoundViolated,
            "request.objective.bounds",
            violation.statement(),
            "raise the bound the objective states, or propose guards whose variants stay inside it",
        ));
    }
    Ok(objective
        .ordering_metrics()
        .into_iter()
        .map(|metric| aggregate.get(metric).unwrap_or(u64::MAX))
        .collect())
}

/// The request one guard's variant is compiled from.
///
/// A guard that pins a symbolic dimension or a submission arrangement narrows
/// the compile inputs, so the variant's schedule can differ structurally from
/// the generic one in decomposition, entry points, fusion, tiling, precision,
/// layout, workspace, and topology. A guard over an axis that narrows no compile
/// input still selects, and the variant it retains is scored against the same
/// domain as every other, so it is retained only if it earns its bytes.
fn narrow(
    request: &ValidatedCompileRequest,
    contract: &SpecializationContract,
    guard: &VariantGuard,
) -> Result<ValidatedCompileRequest, CompileError> {
    let normalized = guard.normalize()?;
    let mut facts = request.facts().clone();
    for (axis, constraint) in &normalized {
        let domain = &contract.axes()[axis];
        let Some(extent) = constraint.largest_admitted(domain) else {
            continue;
        };
        match axis {
            SpecializationAxis::SymbolicDimension { dimension } => {
                facts.symbolic_bindings.insert(dimension.clone(), extent);
            }
            SpecializationAxis::LaunchBatch => {
                facts.expected_launch_batch = u32::try_from(extent).unwrap_or(u32::MAX);
            }
            SpecializationAxis::ValueLayout { .. }
            | SpecializationAxis::ValueDensity { .. }
            | SpecializationAxis::RetainedState
            | SpecializationAxis::Concurrency
            | SpecializationAxis::ConstantIdentity { .. }
            | SpecializationAxis::TargetCapability { .. }
            | SpecializationAxis::TargetResource { .. } => {}
        }
    }
    request.restated_facts(facts)
}

/// The guards one subset mask selects, in canonical order.
fn subset(stated: &[VariantGuard], mask: u32) -> Vec<VariantGuard> {
    stated
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1 << index) != 0)
        .map(|(_, guard)| guard.clone())
        .collect()
}
