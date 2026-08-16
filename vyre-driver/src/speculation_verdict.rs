//! N2 substrate (foundation half): per-rewrite speculation-as-substrate
//! decision policy.
//!
//! Generalizes I2's trace-JIT speculation to ANY "probably profitable"
//! rewrite (vec_pack, shared_promote, async_load_promote, ...). For each
//! candidate rewrite the runtime keeps two compiled variants  -  a
//! conservative baseline and a speculative variant  -  and races them
//! against the autotune DB's recorded winner.
//!
//! This module owns the pure *decision*: given the speculative variant's
//! observed cost vs the baseline (recorded by I3 [`crate::autotune_store`]),
//! return [`crate::speculation_verdict::SpeculationVerdict::Adopt`] (replace baseline with speculative
//! in the cache) or [`crate::speculation_verdict::SpeculationVerdict::Reject`] (drop speculative,
//! stop racing). Pure arithmetic; no I/O, no allocation.
//!
//! The runtime side (compiling both variants on a side pipeline cache
//! key, dispatching them in alternation, recording observations to
//! [`crate::autotune_store`]) lives in `runtime_resident_work_queue` and is
//! Codex's lane. This module is the half that's safe to land before
//! that wiring exists  -  every consumer reads the same decision contract.

/// Per-shape observation feeding the speculation decision.
#[derive(Debug, Clone, Copy)]
pub struct SpeculationObservation {
    /// Number of times the baseline variant was dispatched. Used to
    /// gate how confident we are in `baseline_mean_ns`.
    pub baseline_dispatches: u32,
    /// Mean wall-clock dispatch latency of the baseline variant in
    /// nanoseconds.
    pub baseline_mean_ns: u64,
    /// Number of times the speculative variant was dispatched.
    pub speculative_dispatches: u32,
    /// Mean wall-clock dispatch latency of the speculative variant.
    pub speculative_mean_ns: u64,
    /// Side-compile cost (one-time amortized over future dispatches).
    /// Treated as overhead the speculative variant must pay back.
    pub side_compile_cost_ns: u64,
}

/// Verdict returned by [`decide_speculation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationVerdict {
    /// Speculative variant wins  -  replace the baseline in the cache.
    /// Future dispatches use the speculative variant directly.
    Adopt,
    /// Speculative variant loses or is statistically inconclusive  -
    /// drop it from the cache and stop racing on this shape.
    Reject,
    /// Not enough samples yet  -  keep racing.
    KeepRacing,
}

/// Minimum number of dispatches per variant before a verdict can be
/// rendered. Below this threshold the variance dominates and the
/// decision is unreliable; the runtime keeps racing both variants.
pub const MIN_DISPATCHES_FOR_VERDICT: u32 = 8;

/// Minimum savings in basis points (1 bp = 0.01%) the speculative
/// variant must show over the baseline to be adopted, after side-compile
/// cost amortization. 1500 bps = 15%  -  tuned conservative so adopting
/// is rare but high-confidence.
pub const MIN_ADOPT_SAVINGS_BPS: u64 = 1500;

/// Decide whether to adopt the speculative variant, reject it, or keep
/// racing. Pure arithmetic; widened throughout so adversarial inputs cannot
/// panic or silently clamp a release-path adoption decision.
#[must_use]
pub fn decide_speculation(obs: SpeculationObservation) -> SpeculationVerdict {
    if obs.baseline_dispatches < MIN_DISPATCHES_FOR_VERDICT
        || obs.speculative_dispatches < MIN_DISPATCHES_FOR_VERDICT
    {
        return SpeculationVerdict::KeepRacing;
    }
    if obs.baseline_mean_ns == 0 {
        // Degenerate baseline  -  keep racing rather than divide-by-zero.
        return SpeculationVerdict::KeepRacing;
    }

    // Amortized speculative cost: per-dispatch latency plus
    // side-compile-cost / dispatches-so-far. The further we go, the
    // less the side-compile bites.
    let amortized_overhead_ns = obs
        .side_compile_cost_ns
        .checked_div(u64::from(obs.speculative_dispatches.max(1)))
        .unwrap_or(u64::MAX);
    let effective_speculative_ns =
        u128::from(obs.speculative_mean_ns) + u128::from(amortized_overhead_ns);
    let baseline_mean_ns = u128::from(obs.baseline_mean_ns);

    if effective_speculative_ns >= baseline_mean_ns {
        return SpeculationVerdict::Reject;
    }
    let savings_ns = u64::try_from(baseline_mean_ns - effective_speculative_ns).unwrap_or(u64::MAX);
    let savings_bps = crate::numeric::ratio_basis_points_u64_wide(
        savings_ns,
        obs.baseline_mean_ns,
        0,
        "speculation savings",
        "driver",
    );
    if savings_bps >= MIN_ADOPT_SAVINGS_BPS {
        SpeculationVerdict::Adopt
    } else {
        // Speculative wins but by less than the threshold  -  keep
        // racing in case the gap widens with more samples.
        SpeculationVerdict::KeepRacing
    }
}
