//! One figure per metric for one candidate under one workload class.
//!
//! Every figure here is derived from the open cost model and authenticated
//! target facts, never from a metric name. A metric whose figure only exists
//! later in the pipeline reports `None` at ranking time and is filled in by the
//! stage that owns it, so a bound on artifact bytes is checked against the
//! serialized artifact and never against an estimate of one.

use crate::cost::CostBreakdown;
use crate::DeviceFacts;

use super::metric::ObjectiveMetric;
use super::workload::WorkloadClass;

/// One figure per declared metric, indexed by [`ObjectiveMetric::index`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetricFigures {
    figures: [Option<u64>; ObjectiveMetric::ALL.len()],
}

impl MetricFigures {
    /// No figure for any metric.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            figures: [None; ObjectiveMetric::ALL.len()],
        }
    }

    /// Record `value` for `metric`.
    #[must_use]
    pub const fn with(mut self, metric: ObjectiveMetric, value: u64) -> Self {
        self.figures[metric.index()] = Some(value);
        self
    }

    /// Figure recorded for `metric`, when one is recorded.
    #[must_use]
    pub const fn get(&self, metric: ObjectiveMetric) -> Option<u64> {
        self.figures[metric.index()]
    }

    /// Positional figures, for bound checking.
    #[must_use]
    pub const fn as_array(&self) -> &[Option<u64>; ObjectiveMetric::ALL.len()] {
        &self.figures
    }

    /// Derive every ranking figure of `cost` for one workload class.
    ///
    /// `amortization_launches` is the horizon one-time device cost is spread
    /// over, and is at least one because a horizon of zero would price a cold
    /// start as infinite rather than as itself.
    ///
    /// - latency is the wall time of one submission of the class: the plan's
    ///   modeled device time, once per launch in the batch, multiplied by the
    ///   streams that share the device;
    /// - cold start is the fixed device cost the first submission pays, which is
    ///   the persistent setup the target reported plus the plan's launch term;
    /// - throughput is stated as steady-state nanoseconds per launch, so lower
    ///   is better for it too, and it carries the cold start amortized over the
    ///   horizon;
    /// - peak memory is the resident bytes one stream holds: the bytes the
    ///   allocation plan holds at once, plus the workgroup scratch the selected
    ///   programs declare, which the plan does not place.
    ///
    /// Energy has no figure: no target fact prices device energy, so an energy
    /// objective is refused during validation instead of ranked against a guess.
    #[must_use]
    pub fn derive(
        cost: &CostBreakdown,
        device: DeviceFacts,
        class: WorkloadClass,
        amortization_launches: u32,
    ) -> Self {
        let streams = u64::from(class.concurrent_streams.max(1));
        let batch = u64::from(class.launch_batch.max(1));
        let horizon = u64::from(amortization_launches.max(1));
        let shared = cost.total.saturating_mul(streams);
        let latency = shared.saturating_mul(batch);
        let cold_start = device
            .persistent_setup_overhead_ns()
            .saturating_add(cost.launch_ns)
            .saturating_mul(streams);
        let throughput = shared.saturating_add(cold_start.div_ceil(horizon));
        let peak_memory = cost
            .planned_peak_bytes
            .saturating_add(cost.shared_scratch_bytes)
            .saturating_mul(streams);
        Self::empty()
            .with(ObjectiveMetric::Latency, latency)
            .with(ObjectiveMetric::ColdStart, cold_start)
            .with(ObjectiveMetric::Throughput, throughput)
            .with(ObjectiveMetric::PeakMemory, peak_memory)
    }

    /// Figures a candidate cannot get below, from the launch time `unavoidable_ns`
    /// it is proved to spend.
    ///
    /// Search prunes against this, so every term left out must be non-negative:
    /// latency and throughput carry the proved launch time alone, and a metric
    /// with no derivable lower bound reports none rather than zero, which stops
    /// the search pruning on a figure it cannot prove.
    #[must_use]
    pub fn unavoidable(unavoidable_ns: u64, class: WorkloadClass) -> Self {
        let streams = u64::from(class.concurrent_streams.max(1));
        let batch = u64::from(class.launch_batch.max(1));
        let shared = unavoidable_ns.saturating_mul(streams);
        Self::empty()
            .with(ObjectiveMetric::Latency, shared.saturating_mul(batch))
            .with(ObjectiveMetric::Throughput, shared)
    }
}
