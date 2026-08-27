//! What a production compile optimizes, stated as a versioned record.
//!
//! Ranking without a stated objective is ranking against whichever scalar the
//! cost model happened to total. Two callers then disagree about what "optimal"
//! meant, and a cache cannot tell a latency artifact from a throughput one
//! because neither artifact says which it is. Every production compile therefore
//! carries a [`CompileObjective`]: an ordered primary metric, tie breakers, the
//! workload arrangements it optimizes for, the risk statistic that decides a
//! measured comparison, the horizon one-time cost is amortized over, hard
//! service bounds, and the artifact-portfolio policy.
//!
//! The whole record is `Copy` and `Serialize`, so it participates in request,
//! artifact, cache, and measurement identity by value: changing any field
//! changes the request digest and cannot reuse the previous decision.

mod bounds;
mod figures;
mod metric;
mod portfolio;
mod sequence;
mod workload;

use serde::{Deserialize, Serialize};

use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::DeviceFacts;

pub use bounds::{BoundViolation, ObjectiveBounds};
pub use figures::MetricFigures;
pub use metric::{MetricSequence, ObjectiveMetric, RequiredFact};
pub use portfolio::{CoveragePolicy, PortfolioPolicy};
pub use workload::{RiskStatistic, WorkloadAggregation, WorkloadClass, WorkloadProfile};

/// Current objective schema.
///
/// A compile records this version beside the objective, so an artifact selected
/// under an older objective schema is rejected rather than compared against a
/// record whose fields mean something else.
pub const OBJECTIVE_SCHEMA_VERSION: u16 = 1;

/// What one compile optimizes, and the limits it refuses to exceed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompileObjective {
    version: u16,
    primary: ObjectiveMetric,
    tie_breakers: MetricSequence,
    workload: WorkloadProfile,
    risk: RiskStatistic,
    amortization_launches: u32,
    bounds: ObjectiveBounds,
    portfolio: PortfolioPolicy,
}

impl CompileObjective {
    /// Minimize the latency of one single-launch submission.
    ///
    /// This is the objective a caller states when it has one submission and no
    /// service limit. It is a named constructor rather than a default because a
    /// compile that never stated an objective cannot say what it optimized.
    #[must_use]
    pub const fn minimize_latency() -> Self {
        Self {
            version: OBJECTIVE_SCHEMA_VERSION,
            primary: ObjectiveMetric::Latency,
            tie_breakers: MetricSequence::empty(),
            workload: WorkloadProfile::single(),
            risk: RiskStatistic::TrimmedMean,
            amortization_launches: 1,
            bounds: ObjectiveBounds::unbounded(),
            portfolio: PortfolioPolicy::single(),
        }
    }

    /// Minimize steady-state time per launch over `amortization_launches`.
    #[must_use]
    pub const fn maximize_throughput(amortization_launches: u32) -> Self {
        Self {
            version: OBJECTIVE_SCHEMA_VERSION,
            primary: ObjectiveMetric::Throughput,
            tie_breakers: MetricSequence::empty(),
            workload: WorkloadProfile::single(),
            risk: RiskStatistic::TrimmedMean,
            amortization_launches,
            bounds: ObjectiveBounds::unbounded(),
            portfolio: PortfolioPolicy::single(),
        }
    }

    /// Rank by `primary` before every stated tie breaker.
    #[must_use]
    pub const fn with_primary(mut self, primary: ObjectiveMetric) -> Self {
        self.primary = primary;
        self
    }

    /// Append `metric` to the tie-break chain.
    #[must_use]
    pub const fn with_tie_breaker(mut self, metric: ObjectiveMetric) -> Self {
        self.tie_breakers = self.tie_breakers.pushed(metric);
        self
    }

    /// Replace the workload arrangements this objective optimizes for.
    #[must_use]
    pub const fn with_workload(mut self, workload: WorkloadProfile) -> Self {
        self.workload = workload;
        self
    }

    /// Replace the statistic that decides a measured comparison.
    #[must_use]
    pub const fn with_risk(mut self, risk: RiskStatistic) -> Self {
        self.risk = risk;
        self
    }

    /// Replace the horizon one-time device cost is amortized over.
    #[must_use]
    pub const fn with_amortization_launches(mut self, launches: u32) -> Self {
        self.amortization_launches = launches;
        self
    }

    /// Bound `metric` at `limit`.
    #[must_use]
    pub const fn with_bound(mut self, metric: ObjectiveMetric, limit: u64) -> Self {
        self.bounds = self.bounds.with_bound(metric, limit);
        self
    }

    /// Replace the retained-artifact policy.
    #[must_use]
    pub const fn with_portfolio(mut self, portfolio: PortfolioPolicy) -> Self {
        self.portfolio = portfolio;
        self
    }

    /// Objective schema version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Metric every ranking reads first.
    #[must_use]
    pub const fn primary(&self) -> ObjectiveMetric {
        self.primary
    }

    /// Tie-break chain, in stated order.
    #[must_use]
    pub const fn tie_breakers(&self) -> &MetricSequence {
        &self.tie_breakers
    }

    /// Workload arrangements and their aggregation.
    #[must_use]
    pub const fn workload(&self) -> &WorkloadProfile {
        &self.workload
    }

    /// Statistic that decides a measured comparison.
    #[must_use]
    pub const fn risk(&self) -> RiskStatistic {
        self.risk
    }

    /// Horizon one-time device cost is amortized over.
    #[must_use]
    pub const fn amortization_launches(&self) -> u32 {
        self.amortization_launches
    }

    /// Hard service bounds.
    #[must_use]
    pub const fn bounds(&self) -> &ObjectiveBounds {
        &self.bounds
    }

    /// Retained-artifact policy.
    #[must_use]
    pub const fn portfolio(&self) -> &PortfolioPolicy {
        &self.portfolio
    }

    /// Every metric this objective orders by, primary first.
    #[must_use]
    pub fn ordering_metrics(&self) -> Vec<ObjectiveMetric> {
        let mut metrics = Vec::with_capacity(1 + self.tie_breakers.len());
        metrics.push(self.primary);
        for metric in self.tie_breakers.as_slice() {
            if !metrics.contains(metric) {
                metrics.push(*metric);
            }
        }
        metrics
    }

    /// Validate the objective against the facts `device` authenticated.
    ///
    /// # Errors
    ///
    /// Returns `MKC029_INVALID_OBJECTIVE` when the record is internally
    /// inconsistent: a schema skew, an empty or over-weighted workload profile,
    /// a zero batch, stream or horizon, a repeated or unorderable ordering
    /// metric, or a portfolio ceiling that its own coverage requirement cannot
    /// reach.
    ///
    /// Returns `MKC030_MISSING_CALIBRATED_FACT` when a stated metric needs a
    /// calibrated target fact this device never reported. Ranking against an
    /// absent fact would report an order the device did not earn, so the fact is
    /// named together with the builder that supplies it.
    pub fn validate(&self, device: DeviceFacts) -> Result<(), CompileError> {
        let invalid = |path: &str, message: String, fix: &str| {
            failure(
                CompilerFailureKind::InvalidObjective,
                format!("request.objective.{path}"),
                message,
                fix,
            )
        };
        if self.version != OBJECTIVE_SCHEMA_VERSION {
            return Err(invalid(
                "version",
                format!(
                    "objective states schema {} but this compiler selects under schema {OBJECTIVE_SCHEMA_VERSION}",
                    self.version
                ),
                "rebuild the objective with this compiler's constructors",
            ));
        }
        if self.amortization_launches == 0 {
            return Err(invalid(
                "amortization_launches",
                "amortization horizon is zero launches".to_owned(),
                "state the launches one-time device cost is spread over, at least one",
            ));
        }
        if self.workload.is_empty() {
            return Err(invalid(
                "workload",
                "no workload class is stated".to_owned(),
                "state at least one submission arrangement to optimize for",
            ));
        }
        for (index, class) in self.workload.as_slice().iter().enumerate() {
            if class.launch_batch == 0 || class.concurrent_streams == 0 {
                return Err(invalid(
                    &format!("workload.classes[{index}]"),
                    format!(
                        "class states {} launches over {} streams, so it never runs",
                        class.launch_batch, class.concurrent_streams
                    ),
                    "state at least one launch and one stream per workload class",
                ));
            }
        }
        if self.workload.aggregation().reads_weights() && self.workload.weight_permille() != 1_000 {
            return Err(invalid(
                "workload.classes",
                format!(
                    "weighted aggregation needs weights summing to 1000 permille, the profile sums to {}",
                    self.workload.weight_permille()
                ),
                "restate the class weights so they sum to 1000, or aggregate by worst case",
            ));
        }
        let stated = self.tie_breakers.as_slice();
        for (index, metric) in stated.iter().enumerate() {
            if stated[..index].contains(metric) {
                return Err(invalid(
                    "tie_breakers",
                    format!(
                        "`{}` is stated twice in the tie-break chain, so the second cannot decide",
                        metric.name()
                    ),
                    "state each tie breaker once",
                ));
            }
        }
        if self.tie_breakers.contains(self.primary) {
            return Err(invalid(
                "tie_breakers",
                format!(
                    "`{}` is both the primary metric and a tie breaker, so the tie breaker can never decide",
                    self.primary.name()
                ),
                "drop the repeated metric from the tie-break chain",
            ));
        }
        for metric in self.ordering_metrics() {
            if !metric.is_orderable() {
                return Err(invalid(
                    "primary",
                    format!(
                        "`{}` is measured in {} that only exist after emission, so it cannot order a candidate",
                        metric.name(),
                        metric.unit()
                    ),
                    "order by a metric candidate scoring reports and bound this one instead",
                ));
            }
        }
        let minimum = self
            .portfolio
            .coverage()
            .minimum_variants(self.workload.len());
        if (self.portfolio.max_variants() as usize) < minimum {
            return Err(invalid(
                "portfolio.max_variants",
                format!(
                    "coverage `{}` over {} workload classes needs at least {minimum} artifacts and the ceiling is {}",
                    self.portfolio.coverage().name(),
                    self.workload.len(),
                    self.portfolio.max_variants()
                ),
                "raise the artifact ceiling or state a coverage policy one artifact can satisfy",
            ));
        }
        self.validate_facts(device)
    }

    /// Refuse every stated metric whose price needs a fact this device withheld.
    fn validate_facts(&self, device: DeviceFacts) -> Result<(), CompileError> {
        let stated = self
            .ordering_metrics()
            .into_iter()
            .chain(self.bounds.stated().into_iter().map(|(metric, _)| metric));
        for metric in stated {
            let Some(fact) = metric.required_fact() else {
                continue;
            };
            if fact_is_reported(fact, device) {
                continue;
            }
            return Err(failure(
                CompilerFailureKind::MissingCalibratedFact,
                "request.objective",
                format!(
                    "metric `{}` is priced by the `{}` target fact and this device reports none",
                    metric.name(),
                    fact.name()
                ),
                &format!(
                    "supply the fact through `{}`, or state an objective this device can price",
                    fact.supplied_by()
                ),
            ));
        }
        Ok(())
    }

    /// Aggregate `per_class` figures of one candidate into one figure per
    /// metric.
    ///
    /// A metric no class reported stays absent: an absent figure is not zero,
    /// and a bound checked against zero would admit a candidate nothing
    /// measured.
    #[must_use]
    pub fn aggregate(&self, per_class: &[MetricFigures]) -> MetricFigures {
        let mut aggregated = MetricFigures::empty();
        for metric in ObjectiveMetric::ALL {
            let mut values = Vec::with_capacity(per_class.len());
            for figures in per_class {
                match figures.get(*metric) {
                    Some(value) => values.push(value),
                    None => {
                        values.clear();
                        break;
                    }
                }
            }
            if values.is_empty() {
                continue;
            }
            aggregated = aggregated.with(*metric, self.workload.aggregate(&values));
        }
        aggregated
    }
}

/// Whether `device` reported the calibrated fact `fact` names.
fn fact_is_reported(fact: RequiredFact, device: DeviceFacts) -> bool {
    match fact {
        RequiredFact::PersistentSetup => device.persistent_setup_overhead_ns() > 0,
        // No target fact prices device energy. The variant exists so an energy
        // objective fails with the fact named instead of being ranked against a
        // guess, and so adding an energy fact has one place to report it.
        RequiredFact::EnergyRate => false,
    }
}
