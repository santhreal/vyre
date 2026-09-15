//! What a compile optimizes, and the calibrated fact that prices it.
//!
//! A metric is a whole-artifact figure the compiler can either rank candidates
//! by or bound them against. The two are not the same set: an artifact byte
//! length is not known until emission, so it is a bound and never an ordering
//! key, and a metric whose price needs a device fact the target never reported
//! is refused with the fact named rather than ranked against a guess.

use serde::{Deserialize, Serialize};

use super::sequence::fixed_capacity_list;

/// Calibrated target fact a metric needs before it can be priced.
///
/// Every variant is earned by a metric whose figure reads that fact. A fact no
/// metric needs would report a check that never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RequiredFact {
    /// One-time cost of entering a persistent execution mode.
    PersistentSetup,
    /// Energy per unit of device work. No target reports one yet.
    EnergyRate,
}

impl RequiredFact {
    /// Every declared calibrated fact.
    pub const ALL: &'static [Self] = &[Self::PersistentSetup, Self::EnergyRate];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PersistentSetup => "persistent_setup",
            Self::EnergyRate => "energy_rate",
        }
    }

    /// The `DeviceFacts` builder a caller uses to supply this fact.
    #[must_use]
    pub const fn supplied_by(self) -> &'static str {
        match self {
            Self::PersistentSetup => "DeviceFacts::with_launch_costs",
            Self::EnergyRate => "no builder: this compiler records no energy fact",
        }
    }
}

/// A figure one compile can rank candidates by or bound them against.
///
/// Every metric is stated so that lower is better, including throughput: a
/// throughput objective ranks by steady-state time per submitted launch, which
/// is the same order as bytes per second for one fixed graph and removes the
/// need for a second comparison direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ObjectiveMetric {
    /// End-to-end device nanoseconds of one submission, cold start included.
    Latency,
    /// Steady-state device nanoseconds per launch over the amortization horizon.
    Throughput,
    /// Device nanoseconds the first submission pays before steady state.
    ColdStart,
    /// Peak resident bytes one launch unit holds.
    PeakMemory,
    /// Device energy in microjoules.
    Energy,
    /// Serialized artifact byte length.
    ArtifactBytes,
    /// Number of artifacts a portfolio retains.
    VariantCount,
    /// Deterministic CPU work units the search charged.
    CompileWork,
    /// On-device measurements the search spent.
    MeasurementWork,
}

impl ObjectiveMetric {
    /// Every declared metric.
    pub const ALL: &'static [Self] = &[
        Self::Latency,
        Self::Throughput,
        Self::ColdStart,
        Self::PeakMemory,
        Self::Energy,
        Self::ArtifactBytes,
        Self::VariantCount,
        Self::CompileWork,
        Self::MeasurementWork,
    ];

    /// Stable identifier used in diagnostics, identity, and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::Throughput => "throughput",
            Self::ColdStart => "cold_start",
            Self::PeakMemory => "peak_memory",
            Self::Energy => "energy",
            Self::ArtifactBytes => "artifact_bytes",
            Self::VariantCount => "variant_count",
            Self::CompileWork => "compile_work",
            Self::MeasurementWork => "measurement_work",
        }
    }

    /// Whether candidate ranking can read this metric from a scored candidate.
    ///
    /// A metric that only exists once an artifact is serialized, or once a whole
    /// portfolio is assembled, cannot order a candidate: ranking would have to
    /// invent the figure. Such a metric is admissible as a hard bound, which is
    /// checked where the figure is real.
    #[must_use]
    pub const fn is_orderable(self) -> bool {
        match self {
            Self::Latency
            | Self::Throughput
            | Self::ColdStart
            | Self::PeakMemory
            | Self::Energy => true,
            Self::ArtifactBytes
            | Self::VariantCount
            | Self::CompileWork
            | Self::MeasurementWork => false,
        }
    }

    /// Calibrated target fact this metric needs, when it needs one.
    #[must_use]
    pub const fn required_fact(self) -> Option<RequiredFact> {
        match self {
            Self::Throughput | Self::ColdStart => Some(RequiredFact::PersistentSetup),
            Self::Energy => Some(RequiredFact::EnergyRate),
            Self::Latency
            | Self::PeakMemory
            | Self::ArtifactBytes
            | Self::VariantCount
            | Self::CompileWork
            | Self::MeasurementWork => None,
        }
    }

    /// Positional index of this metric, used by every per-metric array.
    ///
    /// The index is the declaration order of the variant, so a bound slot and a
    /// figure slot cannot disagree about which metric they hold. A closure test
    /// proves `ALL` lists every variant at its own index.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Unit the recorded figure is stated in.
    #[must_use]
    pub const fn unit(self) -> &'static str {
        match self {
            Self::Latency | Self::Throughput | Self::ColdStart => "nanoseconds",
            Self::PeakMemory | Self::ArtifactBytes => "bytes",
            Self::Energy => "microjoules",
            Self::VariantCount => "artifacts",
            Self::CompileWork => "work units",
            Self::MeasurementWork => "measurements",
        }
    }
}

/// Ordered metric sequence with a fixed capacity.
///
/// A tie-break chain is short by construction: past the metrics a device can
/// price, further keys never change an order. The fixed capacity keeps the whole
/// objective `Copy`, which is what lets request identity hash it by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetricSequence {
    metrics: [ObjectiveMetric; Self::CAPACITY],
    len: u8,
}

impl MetricSequence {
    /// Metrics one sequence holds.
    pub const CAPACITY: usize = 4;

    /// The empty sequence.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            metrics: [ObjectiveMetric::Latency; Self::CAPACITY],
            len: 0,
        }
    }

    /// Whether `metric` is already stated.
    #[must_use]
    pub fn contains(&self, metric: ObjectiveMetric) -> bool {
        self.as_slice().contains(&metric)
    }
}

fixed_capacity_list!(MetricSequence, metrics, ObjectiveMetric);

impl Default for MetricSequence {
    fn default() -> Self {
        Self::empty()
    }
}
