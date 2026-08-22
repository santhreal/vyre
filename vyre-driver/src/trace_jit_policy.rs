//! I2 substrate: trace-based JIT specialization decision policy.
//!
//! When the runtime sees the SAME `SpecCacheKey` repeatedly miss the
//! AutotuneStore (I3), the dispatcher should pre-emptively
//! specialize for the next-most-likely shape variation: same
//! `shader_hash` + slightly-different `spec_hash` (different
//! literal values, dtype tag, etc.).
//!
//! Pure decision: given a hit/miss histogram for a `(shader_hash,
//! adapter_id)` pair, should the runtime fire a speculative
//! pre-spec on a related shape?
//!
//! The trade-off: a speculative spec costs one full compile cycle
//! up-front but eliminates the cache miss when the predicted shape
//! arrives. Worth it iff the hit rate on the predicted shape is
//! high enough to amortise the speculative cost.

/// Inputs to the speculative-spec decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceJitInputs {
    /// How many times the dispatcher has seen the SAME `shader_hash`
    /// in the recent window. The trace JIT only considers shapes
    /// that have already proven hot.
    pub shader_hit_count: u32,
    /// Confidence  -  out of 10000  -  that the next miss will be on
    /// the predicted shape. Computed by the runtime from a sliding
    /// window over recent miss patterns.
    pub prediction_confidence_bps: u32,
    /// Cost of one speculative spec in nanoseconds (pipeline compile
    /// + storage). The runtime measures this on the last full
    /// compile pass.
    pub speculative_spec_cost_ns: u64,
    /// Cost of a missed dispatch (cache miss + compile path) in
    /// nanoseconds. Same source as the autotune sample.
    pub miss_cost_ns: u64,
}

/// Verdict from [`decide_trace_jit_speculation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceJitDecision {
    /// Don't speculate  -  either the shape isn't hot enough,
    /// confidence is too low, or the speculative cost won't amortise.
    HoldSteady,
    /// Fire a speculative spec on the predicted shape now. The
    /// `expected_savings_ns` is the predicted miss cost weighted by
    /// confidence, minus the speculative spec cost.
    Speculate {
        /// Predicted savings (nanoseconds) from avoiding the next
        /// miss, after netting out the speculative spec cost.
        /// Positive by construction.
        expected_savings_ns: u128,
    },
}

/// Hit count below which speculation is never worth it. Below this,
/// the shape isn't hot enough to justify a pre-emptive compile.
pub const TRACE_JIT_HOT_SHAPE_THRESHOLD: u32 = 8;

/// Confidence (basis points) below which the prediction isn't
/// reliable enough to justify the speculative spec cost.
pub const TRACE_JIT_MIN_CONFIDENCE_BPS: u32 = 6_000; // 60%

/// Decide whether to speculatively pre-specialize for a predicted
/// shape on the basis of recent hit/miss patterns.
///
/// Predicted savings = `(prediction_confidence / 10000) * miss_cost`.
/// Speculate iff predicted savings exceed the speculative spec cost.
#[must_use]
pub fn decide_trace_jit_speculation(inputs: TraceJitInputs) -> TraceJitDecision {
    if inputs.shader_hit_count < TRACE_JIT_HOT_SHAPE_THRESHOLD {
        return TraceJitDecision::HoldSteady;
    }
    if inputs.prediction_confidence_bps < TRACE_JIT_MIN_CONFIDENCE_BPS {
        return TraceJitDecision::HoldSteady;
    }
    if inputs.miss_cost_ns == 0 {
        return TraceJitDecision::HoldSteady;
    }
    let weighted = crate::numeric::weighted_u64_by_basis_points_u128(
        inputs.miss_cost_ns,
        inputs.prediction_confidence_bps,
    );
    let speculative_spec_cost_ns = u128::from(inputs.speculative_spec_cost_ns);
    if weighted <= speculative_spec_cost_ns {
        return TraceJitDecision::HoldSteady;
    }
    let expected_savings_ns = weighted - speculative_spec_cost_ns;
    TraceJitDecision::Speculate {
        expected_savings_ns,
    }
}
