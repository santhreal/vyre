//! Contracts for `vyre_driver::shape_prediction`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::shape_prediction::{ShapeFingerprint, ShapeHistory, MAX_HISTORY};

fn fp(seed: u32) -> ShapeFingerprint {
    let mut a = [0u32; 8];
    for (i, slot) in a.iter_mut().enumerate() {
        *slot = seed.wrapping_mul(31).wrapping_add(i as u32);
    }
    a
}

#[test]
fn empty_history_predicts_nothing() {
    let h = ShapeHistory::new();
    assert!(h.predict_next().is_none());
}

#[test]
fn single_entry_history_cannot_predict() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    assert!(h.predict_next().is_none());
}

#[test]
fn repeated_fingerprint_predicts_repeat() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    h.record(fp(1));
    assert_eq!(h.predict_next(), Some(fp(1)));
}

#[test]
fn two_step_cycle_is_predicted() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    h.record(fp(2));
    h.record(fp(1));
    h.record(fp(2));
    assert_eq!(h.predict_next(), Some(fp(1)));
}

#[test]
fn three_step_cycle_is_predicted() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    h.record(fp(2));
    h.record(fp(3));
    h.record(fp(1));
    h.record(fp(2));
    h.record(fp(3));
    assert_eq!(h.predict_next(), Some(fp(1)));
}

#[test]
fn partial_three_step_cycle_is_predicted_before_second_cycle_completes() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    h.record(fp(2));
    h.record(fp(3));
    h.record(fp(1));
    h.record(fp(2));
    assert_eq!(h.predict_next(), Some(fp(3)));
}

#[test]
fn partial_long_cycle_prefetches_next_phase() {
    let mut h = ShapeHistory::new();
    for seed in [10, 20, 30, 40, 10, 20, 30] {
        h.record(fp(seed));
    }
    assert_eq!(h.predict_next(), Some(fp(40)));
}

#[test]
fn no_pattern_means_no_prediction() {
    let mut h = ShapeHistory::new();
    h.record(fp(1));
    h.record(fp(2));
    h.record(fp(3));
    h.record(fp(4));
    assert!(h.predict_next().is_none());
}

#[test]
fn history_caps_at_max_entries() {
    let mut h = ShapeHistory::new();
    for i in 0..(MAX_HISTORY + 5) {
        h.record(fp(i as u32));
    }
    assert_eq!(h.len(), MAX_HISTORY);
    // Earliest entry is fp(5), latest is fp(MAX_HISTORY+4).
    assert_eq!(h.latest(), Some(&fp((MAX_HISTORY + 4) as u32)));
    assert!(!h.contains(&fp(0)));
    assert!(h.contains(&fp(5)));
    assert!(h.contains(&fp((MAX_HISTORY + 4) as u32)));
}
