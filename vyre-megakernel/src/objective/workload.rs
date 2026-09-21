//! The workload one compile optimizes for, and how its classes combine.
//!
//! One graph is submitted under more than one arrangement: a single interactive
//! submission, a batch of a thousand, several concurrent streams sharing the
//! device. Those arrangements do not rank the same candidate the same way, so a
//! compile states the arrangements it cares about with a weight each, and states
//! whether the objective reads their weighted average or their worst case.
//!
//! Weights are permille so the profile stays integral: a weighted average of
//! nanosecond figures must be reproducible on every host that reads the same
//! artifact, and a float average is not.

use serde::{Deserialize, Serialize};

use super::sequence::fixed_capacity_list;

/// How per-class figures combine into the figure the objective orders by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkloadAggregation {
    /// Weighted mean over the stated classes, weights in permille.
    Weighted,
    /// Worst stated class, weights ignored.
    WorstCase,
}

impl WorkloadAggregation {
    /// Every declared aggregation.
    pub const ALL: &'static [Self] = &[Self::Weighted, Self::WorstCase];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::WorstCase => "worst_case",
        }
    }

    /// Whether stated weights change the aggregate.
    #[must_use]
    pub const fn reads_weights(self) -> bool {
        match self {
            Self::Weighted => true,
            Self::WorstCase => false,
        }
    }
}

/// Which statistic of a measured sample set decides a comparison.
///
/// A median hides a tail an interactive submission feels; a worst case pays for
/// one outlier the protocol already trims. The statistic is part of the
/// objective because two compiles that disagree about it can select different
/// winners from identical samples, and an artifact must state which one it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RiskStatistic {
    /// Robust central estimate the measurement protocol already computes.
    TrimmedMean,
    /// Middle retained sample.
    Median,
    /// 95th percentile of retained samples.
    P95,
    /// 99th percentile of retained samples.
    P99,
    /// Slowest retained sample.
    WorstCase,
}

impl RiskStatistic {
    /// Every declared statistic.
    pub const ALL: &'static [Self] = &[
        Self::TrimmedMean,
        Self::Median,
        Self::P95,
        Self::P99,
        Self::WorstCase,
    ];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TrimmedMean => "trimmed_mean",
            Self::Median => "median",
            Self::P95 => "p95",
            Self::P99 => "p99",
            Self::WorstCase => "worst_case",
        }
    }

    /// Permille rank the statistic reads from an ordered retained sample set.
    ///
    /// A trimmed mean is not a rank and reports `None`: it is the estimate the
    /// protocol computes over the whole retained set.
    #[must_use]
    pub const fn permille_rank(self) -> Option<u16> {
        match self {
            Self::TrimmedMean => None,
            Self::Median => Some(500),
            Self::P95 => Some(950),
            Self::P99 => Some(990),
            Self::WorstCase => Some(1000),
        }
    }

    /// Statistic of `ordered` this risk policy reads.
    ///
    /// `ordered` is ascending and already trimmed by the measurement protocol.
    /// `trimmed_mean` is the estimate the protocol computed over that same set,
    /// so a caller passes it rather than recomputing a second central estimate
    /// from the samples.
    #[must_use]
    pub fn read(self, ordered: &[u64], trimmed_mean: u64) -> u64 {
        let Some(rank) = self.permille_rank() else {
            return trimmed_mean;
        };
        if ordered.is_empty() {
            return trimmed_mean;
        }
        let last = ordered.len() - 1;
        // Nearest-rank on the retained set: an interpolated percentile would
        // report a duration no launch took.
        let index = (last as u64 * u64::from(rank) + 500) / 1000;
        ordered[index as usize]
    }
}

/// One submission arrangement the objective optimizes for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkloadClass {
    /// Launches one submission of this class issues.
    pub launch_batch: u32,
    /// Independent streams sharing the device while this class runs.
    pub concurrent_streams: u32,
    /// Share of the workload this class represents, in permille.
    pub weight_permille: u16,
}

impl WorkloadClass {
    /// One submission of one launch on an otherwise idle device.
    #[must_use]
    pub const fn single() -> Self {
        Self {
            launch_batch: 1,
            concurrent_streams: 1,
            weight_permille: 1_000,
        }
    }

    /// Construct an explicit class.
    #[must_use]
    pub const fn new(launch_batch: u32, concurrent_streams: u32, weight_permille: u16) -> Self {
        Self {
            launch_batch,
            concurrent_streams,
            weight_permille,
        }
    }
}

/// Every arrangement one compile optimizes for, and how they combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkloadProfile {
    classes: [WorkloadClass; Self::CAPACITY],
    len: u8,
    aggregation: WorkloadAggregation,
}

impl WorkloadProfile {
    /// Classes one profile holds.
    pub const CAPACITY: usize = 4;

    /// One single-launch class on an idle device.
    #[must_use]
    pub const fn single() -> Self {
        Self {
            classes: [WorkloadClass::single(); Self::CAPACITY],
            len: 1,
            aggregation: WorkloadAggregation::Weighted,
        }
    }

    /// Replace the aggregation policy.
    #[must_use]
    pub const fn with_aggregation(mut self, aggregation: WorkloadAggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Replace the class set with `first` alone.
    #[must_use]
    pub const fn of(first: WorkloadClass) -> Self {
        Self {
            classes: [first; Self::CAPACITY],
            len: 1,
            aggregation: WorkloadAggregation::Weighted,
        }
    }

    /// Aggregation policy.
    #[must_use]
    pub const fn aggregation(&self) -> WorkloadAggregation {
        self.aggregation
    }

    /// Sum of stated weights, in permille.
    #[must_use]
    pub fn weight_permille(&self) -> u32 {
        self.as_slice()
            .iter()
            .fold(0, |total, class| total + u32::from(class.weight_permille))
    }

    /// Combine one figure per stated class into the aggregate the objective
    /// orders by.
    ///
    /// `per_class` is indexed like [`Self::as_slice`]. A weighted aggregate
    /// rounds up, so an aggregate never reports a figure below every class it
    /// combines.
    #[must_use]
    pub fn aggregate(&self, per_class: &[u64]) -> u64 {
        let classes = self.as_slice();
        let paired = classes.iter().zip(per_class.iter());
        match self.aggregation {
            WorkloadAggregation::WorstCase => {
                paired.fold(0, |worst: u64, (_, value)| worst.max(*value))
            }
            WorkloadAggregation::Weighted => {
                let weighted = paired.fold(0_u128, |total, (class, value)| {
                    total + u128::from(*value) * u128::from(class.weight_permille)
                });
                let total_weight = u128::from(self.weight_permille());
                if total_weight == 0 {
                    return 0;
                }
                let rounded = weighted.div_ceil(total_weight);
                u64::try_from(rounded).unwrap_or(u64::MAX)
            }
        }
    }
}

fixed_capacity_list!(WorkloadProfile, classes, WorkloadClass);

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self::single()
    }
}
