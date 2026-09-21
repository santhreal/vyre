//! Contracts for `vyre_driver::trace_jit_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::trace_jit_policy::{
    decide_trace_jit_speculation, TraceJitDecision, TraceJitInputs, TRACE_JIT_HOT_SHAPE_THRESHOLD,
    TRACE_JIT_MIN_CONFIDENCE_BPS,
};

fn inp(hit: u32, conf: u32, spec_cost: u64, miss_cost: u64) -> TraceJitInputs {
    TraceJitInputs {
        shader_hit_count: hit,
        prediction_confidence_bps: conf,
        speculative_spec_cost_ns: spec_cost,
        miss_cost_ns: miss_cost,
    }
}

#[test]
fn cold_shape_holds_steady() {
    // hit_count below threshold → HoldSteady regardless of others.
    assert_eq!(
        decide_trace_jit_speculation(inp(7, 9_000, 1_000, 100_000)),
        TraceJitDecision::HoldSteady
    );
}

#[test]
fn low_confidence_holds_steady() {
    // 5999 < 6000 → HoldSteady.
    assert_eq!(
        decide_trace_jit_speculation(inp(100, 5_999, 1_000, 1_000_000)),
        TraceJitDecision::HoldSteady
    );
}

#[test]
fn zero_miss_cost_holds_steady() {
    // No miss to avoid.
    assert_eq!(
        decide_trace_jit_speculation(inp(100, 9_000, 1_000, 0)),
        TraceJitDecision::HoldSteady
    );
}

#[test]
fn positive_savings_speculates() {
    // 100% confidence × 100us miss cost = 100us weighted savings.
    // Speculative spec costs 10us → net savings 90us.
    let dec = decide_trace_jit_speculation(inp(100, 10_000, 10_000, 100_000));
    assert_eq!(
        dec,
        TraceJitDecision::Speculate {
            expected_savings_ns: 90_000
        }
    );
}

#[test]
fn confidence_weights_predicted_savings() {
    // 60% × 100us = 60us weighted; spec cost 50us → savings 10us.
    let dec = decide_trace_jit_speculation(inp(100, 6_000, 50_000, 100_000));
    assert_eq!(
        dec,
        TraceJitDecision::Speculate {
            expected_savings_ns: 10_000
        }
    );
}

#[test]
fn spec_cost_above_weighted_savings_holds_steady() {
    // 60% × 100us = 60us; spec cost 60us → no net savings.
    assert_eq!(
        decide_trace_jit_speculation(inp(100, 6_000, 60_000, 100_000)),
        TraceJitDecision::HoldSteady
    );
}

#[test]
fn at_threshold_speculates_when_other_inputs_pass() {
    // Hit count exactly at threshold (8) is the minimum that
    // qualifies  -  strict `<` for cold check.
    let dec = decide_trace_jit_speculation(inp(8, 10_000, 1_000, 100_000));
    match dec {
        TraceJitDecision::Speculate { .. } => {}
        other => panic!("expected Speculate; got {:?}", other),
    }
}

#[test]
fn confidence_at_threshold_speculates() {
    // Confidence exactly at threshold (6000 = 60%) is the minimum
    // that qualifies.
    let dec = decide_trace_jit_speculation(inp(100, 6_000, 1_000, 100_000));
    match dec {
        TraceJitDecision::Speculate { .. } => {}
        other => panic!("expected Speculate; got {:?}", other),
    }
}

#[test]
fn extreme_inputs_use_widened_arithmetic() {
    // u64::MAX miss_cost × 10000 confidence shouldn't panic.
    let dec = decide_trace_jit_speculation(inp(100, 10_000, 1_000, u64::MAX));
    match dec {
        TraceJitDecision::Speculate {
            expected_savings_ns,
        } => assert_eq!(expected_savings_ns, u128::from(u64::MAX) - 1_000),
        other => panic!("expected Speculate; got {:?}", other),
    }
}

#[test]
fn calibration_constants_pinned() {
    assert_eq!(TRACE_JIT_HOT_SHAPE_THRESHOLD, 8);
    assert_eq!(TRACE_JIT_MIN_CONFIDENCE_BPS, 6_000);
}
