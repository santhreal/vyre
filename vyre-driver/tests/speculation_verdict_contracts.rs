//! Contracts for `vyre_driver::speculation_verdict`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::speculation_verdict::{
    decide_speculation, SpeculationObservation, SpeculationVerdict,
};

fn obs(b_n: u32, b_ns: u64, s_n: u32, s_ns: u64, sc_ns: u64) -> SpeculationObservation {
    SpeculationObservation {
        baseline_dispatches: b_n,
        baseline_mean_ns: b_ns,
        speculative_dispatches: s_n,
        speculative_mean_ns: s_ns,
        side_compile_cost_ns: sc_ns,
    }
}

#[test]
fn under_threshold_keeps_racing() {
    // baseline only sampled 3 times  -  too few to verdict.
    let v = decide_speculation(obs(3, 100_000, 100, 50_000, 0));
    assert_eq!(v, SpeculationVerdict::KeepRacing);
}

#[test]
fn speculative_clearly_faster_adopts() {
    // baseline 100us, speculative 50us, no side-compile cost.
    // savings = 50%, well over 15% threshold.
    let v = decide_speculation(obs(50, 100_000, 50, 50_000, 0));
    assert_eq!(v, SpeculationVerdict::Adopt);
}

#[test]
fn speculative_slower_rejects() {
    let v = decide_speculation(obs(50, 50_000, 50, 100_000, 0));
    assert_eq!(v, SpeculationVerdict::Reject);
}

#[test]
fn speculative_marginally_faster_keeps_racing() {
    // baseline 100us, speculative 95us → 5% savings, under 15%.
    let v = decide_speculation(obs(50, 100_000, 50, 95_000, 0));
    assert_eq!(v, SpeculationVerdict::KeepRacing);
}

#[test]
fn side_compile_cost_amortizes_into_decision() {
    // baseline 100us, speculative 50us, but side-compile = 1ms.
    // After 50 dispatches, amortized overhead = 1ms/50 = 20us.
    // Effective speculative = 50us + 20us = 70us → 30% savings.
    let v = decide_speculation(obs(50, 100_000, 50, 50_000, 1_000_000));
    assert_eq!(v, SpeculationVerdict::Adopt);
}

#[test]
fn side_compile_cost_can_dominate_early() {
    // Same shape but only 8 speculative dispatches.
    // Amortized overhead = 1ms/8 = 125us. Effective = 50us + 125us = 175us
    // > baseline 100us → reject.
    let v = decide_speculation(obs(50, 100_000, 8, 50_000, 1_000_000));
    assert_eq!(v, SpeculationVerdict::Reject);
}

#[test]
fn zero_baseline_keeps_racing_rather_than_dividing_by_zero() {
    let v = decide_speculation(obs(50, 0, 50, 50_000, 0));
    assert_eq!(v, SpeculationVerdict::KeepRacing);
}

#[test]
fn extreme_inputs_do_not_panic() {
    assert_eq!(
        decide_speculation(obs(u32::MAX, u64::MAX, u32::MAX, u64::MAX, u64::MAX)),
        SpeculationVerdict::Reject
    );
    assert_eq!(
        decide_speculation(obs(u32::MAX, 1, u32::MAX, u64::MAX, 0)),
        SpeculationVerdict::Reject
    );
}

#[test]
fn huge_savings_use_widened_arithmetic_not_saturation() {
    assert_eq!(
        decide_speculation(obs(u32::MAX, u64::MAX, u32::MAX, 1, 0)),
        SpeculationVerdict::Adopt
    );
}
