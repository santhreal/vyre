//! Hard service limits one compile refuses to exceed.
//!
//! A bound is not a preference. A candidate that violates one is not ranked
//! last, it is rejected, and a compile whose whole legal candidate set violates
//! one fails with the bound and the best achieved figure named. Bounds are
//! stated per metric and stored positionally, so adding a metric adds a bound
//! slot instead of a second table that can disagree with the metric list.

use serde::{Deserialize, Serialize};

use super::metric::ObjectiveMetric;

/// Hard limit per metric, in that metric's unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectiveBounds {
    limits: [Option<u64>; ObjectiveMetric::ALL.len()],
}

impl ObjectiveBounds {
    /// No bound on any metric.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            limits: [None; ObjectiveMetric::ALL.len()],
        }
    }

    /// Bound `metric` at `limit`, replacing any previous limit on it.
    #[must_use]
    pub const fn with_bound(mut self, metric: ObjectiveMetric, limit: u64) -> Self {
        self.limits[metric.index()] = Some(limit);
        self
    }

    /// Limit stated for `metric`, when one is stated.
    #[must_use]
    pub const fn limit(&self, metric: ObjectiveMetric) -> Option<u64> {
        self.limits[metric.index()]
    }

    /// Whether any metric carries a limit.
    #[must_use]
    pub fn is_unbounded(&self) -> bool {
        self.limits.iter().all(Option::is_none)
    }

    /// Every stated bound, in metric declaration order.
    #[must_use]
    pub fn stated(&self) -> Vec<(ObjectiveMetric, u64)> {
        ObjectiveMetric::ALL
            .iter()
            .filter_map(|metric| self.limit(*metric).map(|limit| (*metric, limit)))
            .collect()
    }

    /// The metric `figures` exceeds first, in metric declaration order.
    ///
    /// `figures` reports one figure per metric, indexed by
    /// [`ObjectiveMetric::index`]; a metric with no figure is not checked here
    /// because the stage that owns it checks it where the figure is real.
    #[must_use]
    pub fn first_violation(
        &self,
        figures: &[Option<u64>; ObjectiveMetric::ALL.len()],
    ) -> Option<BoundViolation> {
        ObjectiveMetric::ALL.iter().find_map(|metric| {
            let limit = self.limit(*metric)?;
            let achieved = figures[metric.index()]?;
            (achieved > limit).then_some(BoundViolation {
                metric: *metric,
                limit,
                achieved,
            })
        })
    }
}

impl Default for ObjectiveBounds {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// One hard bound a figure exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoundViolation {
    /// Metric that was bounded.
    pub metric: ObjectiveMetric,
    /// Limit the objective stated.
    pub limit: u64,
    /// Figure the candidate or artifact achieved.
    pub achieved: u64,
}

impl BoundViolation {
    /// Diagnostic sentence naming the bound, the unit, and both figures.
    #[must_use]
    pub fn statement(&self) -> String {
        format!(
            "{} bound is {} {} and the best legal candidate achieves {} {}",
            self.metric.name(),
            self.limit,
            self.metric.unit(),
            self.achieved,
            self.metric.unit()
        )
    }
}
