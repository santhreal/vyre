//! Runtime-side paired speculation races for megakernel dispatch.
//!
//! The driver crate owns the backend-neutral decision math. This module
//! owns the megakernel/runtime bridge: every candidate rewrite is measured
//! as a conservative/speculative pair, the faster side is recorded in the
//! shared autotune store, and the accumulated sample window is converted
//! into the N2 adoption verdict.

use vyre_driver::autotune_store::{AutotuneRecord, AutotuneStore};
use vyre_driver::speculation_verdict::{
    decide_speculation, SpeculationObservation, SpeculationVerdict,
};
use vyre_driver::{
    record_speculative_variant_race, SpeculativeVariantDecision, SpeculativeVariantKeys,
    SpeculativeVariantRace,
};

/// One measured conservative/speculative dispatch pair.
#[derive(Debug, Clone)]
pub struct PairedSpeculationSample {
    /// Conservative dispatch elapsed time, excluding compile/cache miss.
    pub conservative_dispatch_ns: u64,
    /// Speculative dispatch elapsed time, excluding compile/cache miss.
    pub speculative_dispatch_ns: u64,
    /// Conservative compile/cache-miss time for this pair.
    pub conservative_compile_ns: u64,
    /// Speculative compile/cache-miss time for this pair.
    pub speculative_compile_ns: u64,
    /// Autotune record attached to the conservative variant.
    pub conservative_record: AutotuneRecord,
    /// Autotune record attached to the speculative variant.
    pub speculative_record: AutotuneRecord,
}

/// Result of recording one paired race.
#[derive(Debug, Clone)]
pub struct PairedSpeculationUpdate {
    /// Winning per-sample cache/autotune decision.
    pub race_decision: SpeculativeVariantDecision,
    /// Accumulated N2 verdict for the shape.
    pub verdict: SpeculationVerdict,
    /// Observation fed into the verdict.
    pub observation: SpeculationObservation,
}

/// Accumulated paired-race window for one rewrite candidate and shape.
#[derive(Debug, Default, Clone)]
pub struct PairedSpeculationWindow {
    conservative: RunningMean,
    speculative: RunningMean,
    side_compile_cost_ns: u64,
}

impl PairedSpeculationWindow {
    /// Empty paired-race window.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            conservative: RunningMean::new(),
            speculative: RunningMean::new(),
            side_compile_cost_ns: 0,
        }
    }

    /// Number of paired samples recorded.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.conservative.count.min(self.speculative.count)
    }

    /// True when no paired samples were recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current observation for the N2 speculation policy.
    #[must_use]
    pub fn observation(&self) -> SpeculationObservation {
        SpeculationObservation {
            baseline_dispatches: self.conservative.count,
            baseline_mean_ns: self.conservative.mean_ns(),
            speculative_dispatches: self.speculative.count,
            speculative_mean_ns: self.speculative.mean_ns(),
            side_compile_cost_ns: self.side_compile_cost_ns,
        }
    }

    /// Record one paired sample, update the autotune store with the
    /// per-sample winner, and return the accumulated adoption verdict.
    pub fn record_sample(
        &mut self,
        store: &mut AutotuneStore,
        keys: SpeculativeVariantKeys<'_>,
        sample: PairedSpeculationSample,
    ) -> PairedSpeculationUpdate {
        self.conservative.record(sample.conservative_dispatch_ns);
        self.speculative.record(sample.speculative_dispatch_ns);
        self.side_compile_cost_ns = self
            .side_compile_cost_ns
            .saturating_add(sample.speculative_compile_ns);

        let race_decision = record_speculative_variant_race(
            store,
            keys,
            SpeculativeVariantRace {
                conservative_dispatch_ns: sample.conservative_dispatch_ns,
                speculative_dispatch_ns: sample.speculative_dispatch_ns,
                conservative_compile_ns: sample.conservative_compile_ns,
                speculative_compile_ns: sample.speculative_compile_ns,
                conservative_record: sample.conservative_record,
                speculative_record: sample.speculative_record,
            },
        );
        let observation = self.observation();
        let verdict = decide_speculation(observation);
        PairedSpeculationUpdate {
            race_decision,
            verdict,
            observation,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct RunningMean {
    count: u32,
    total_ns: u128,
}

impl RunningMean {
    const fn new() -> Self {
        Self {
            count: 0,
            total_ns: 0,
        }
    }

    fn record(&mut self, value_ns: u64) {
        self.count = self.count.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(u128::from(value_ns));
    }

    fn mean_ns(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let mean = self.total_ns / u128::from(self.count);
        match u64::try_from(mean) {
            Ok(mean) => mean,
            Err(_) => u64::MAX,
        }
    }
}
