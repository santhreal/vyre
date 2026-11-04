//! Contracts for `vyre_driver::command_reuse_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::command_reuse_policy::{
    decide_command_reuse, CommandReuseDecision, CommandReuseInputs,
};

fn inp(rep: u32, launch: u64, record: u64, replay: u64) -> CommandReuseInputs {
    CommandReuseInputs {
        repeat_count: rep,
        per_launch_overhead_ns: launch,
        record_overhead_ns: record,
        replay_overhead_ns: replay,
    }
}

#[test]
fn single_dispatch_is_plain() {
    // No repetition → recording wastes work.
    assert_eq!(
        decide_command_reuse(inp(1, 5_000, 25_000, 500)),
        CommandReuseDecision::PlainLaunches
    );
}

#[test]
fn zero_repeat_is_plain() {
    assert_eq!(
        decide_command_reuse(inp(0, 5_000, 25_000, 500)),
        CommandReuseDecision::PlainLaunches
    );
}

#[test]
fn replay_no_cheaper_than_launch_is_plain() {
    // Graph replay = per-launch overhead → no savings possible.
    assert_eq!(
        decide_command_reuse(inp(1000, 5_000, 25_000, 5_000)),
        CommandReuseDecision::PlainLaunches
    );
}

#[test]
fn small_repeat_under_amortisation_is_plain() {
    // 5 repeats × (5000 - 500) savings = 22_500; record costs 25_000.
    assert_eq!(
        decide_command_reuse(inp(5, 5_000, 25_000, 500)),
        CommandReuseDecision::PlainLaunches
    );
}

#[test]
fn large_repeat_above_amortisation_picks_record_and_replay() {
    // 100 repeats × 4_500 savings = 450_000; record 25_000.
    // Net savings = 425_000.
    assert_eq!(
        decide_command_reuse(inp(100, 5_000, 25_000, 500)),
        CommandReuseDecision::RecordAndReplay {
            savings_ns: 425_000
        }
    );
}

#[test]
fn savings_strictly_positive_when_record_and_replay() {
    let dec = decide_command_reuse(inp(1000, 5_000, 25_000, 500));
    match dec {
        CommandReuseDecision::RecordAndReplay { savings_ns } => assert!(savings_ns > 0),
        other => panic!("expected RecordAndReplay; got {:?}", other),
    }
}

#[test]
fn widened_arithmetic_preserves_extreme_savings() {
    // u32::MAX repeats × u64-near-max savings shouldn't panic.
    let dec = decide_command_reuse(inp(u32::MAX, u64::MAX / 2, 25_000, 1));
    match dec {
        CommandReuseDecision::RecordAndReplay { savings_ns } => {
            assert_eq!(
                savings_ns,
                u128::from(u32::MAX) * (u128::from(u64::MAX / 2) - 1) - 25_000
            );
        }
        other => panic!("expected RecordAndReplay; got {:?}", other),
    }
}
