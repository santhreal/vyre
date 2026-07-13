//! GPU-IR parity for `fixpoint::bitset_fixpoint::bitset_fixpoint_warm_start`.
//!
//! One warm-started fixpoint step (had no parity test, found by the
//! registry-coverage closure gate). Per word `w` (lane `t < words`):
//!   current[w] = current[w] | seed[w]              // warm-start with prior state
//!   if original_current[w] != next[w] { changed[0] |= 1 }   // convergence flag
//!
//! The convergence comparison uses the ORIGINAL pre-warm-start `c0`, NOT the
//! seed-warmed `c1`: the documented AUDIT_2026-04-24 F-BF-01 CRITICAL fix (a
//! `c1`-comparison falsely signalled convergence when the seed masked the delta).
//! This test LOCKS that distinction: case A has `c1 != next` everywhere yet
//! `c0 == next`, so a correct implementation reports NOT-changed (0); a regressed
//! `c1`-comparison would wrongly report changed (1).
#![cfg(feature = "fixpoint")]

use vyre_primitives::fixpoint::bitset_fixpoint::bitset_fixpoint_warm_start;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Returns (current_out, changed). Buffers: current(0,RW), next(1,RO),
/// changed(2,RW), seed(3,RO) → writable outputs are [current, changed].
fn eval(current: &[u32], next: &[u32], seed: &[u32]) -> (Vec<u32>, u32) {
    let words = current.len() as u32;
    let program = bitset_fixpoint_warm_start("current", "next", "changed", "seed", words);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(current)),
            Value::from(pack(next)),
            Value::from(pack(&[0u32])),
            Value::from(pack(seed)),
        ],
    )
    .expect("bitset_fixpoint_warm_start reference evaluation must succeed");
    // Locate each writable output BY NAME (not fixed position) via the interpreter's own
    // `output_index`, so a buffer reorder or a future fused intermediate can't silently
    // shift the reads (the drift class the multi-block prefix-scan harness hit).
    let current_idx = vyre_reference::output_index(&program, "current")
        .expect("Fix: bitset_fixpoint_warm_start must declare the `current` output");
    let changed_idx = vyre_reference::output_index(&program, "changed")
        .expect("Fix: bitset_fixpoint_warm_start must declare the `changed` output");
    let current_out = unpack(&outputs[current_idx].to_bytes());
    let changed = unpack(&outputs[changed_idx].to_bytes())[0];
    (current_out, changed)
}

#[test]
fn warm_start_ors_seed_into_current() {
    let current = [0b0001u32, 0b0000];
    let seed = [0b0010u32, 0b0100];
    let next = [0b0001u32, 0b0000];
    let (current_out, _) = eval(&current, &next, &seed);
    assert_eq!(
        current_out,
        vec![0b0011, 0b0100],
        "current must be OR-ed with the seed (warm start)"
    );
}

#[test]
fn convergence_compares_pre_warm_start_state_not_warmed() {
    // c0 == next in EVERY word, but c1 (= c0 | seed) != next everywhere. The
    // F-BF-01-correct implementation compares c0 → NOT changed (0). A regressed
    // c1-comparison would see c1 != next and wrongly flag changed (1).
    let current = [0b0001u32, 0b0000];
    let seed = [0b0010u32, 0b0100];
    let next = [0b0001u32, 0b0000]; // == original current, != warmed current
    let (current_out, changed) = eval(&current, &next, &seed);
    assert_eq!(
        current_out,
        vec![0b0011, 0b0100],
        "warm start still applied"
    );
    assert_eq!(
        changed, 0,
        "convergence must compare the ORIGINAL current (c0) vs next, not the seed-warmed c1 (F-BF-01)"
    );
}

#[test]
fn changed_flag_set_when_original_differs_from_next() {
    let current = [0b0001u32, 0b0000];
    let seed = [0b0010u32, 0b0100];
    // w0: original c0=0b0001 != next 0b1001 → changed must be 1.
    let next = [0b1001u32, 0b0000];
    let (current_out, changed) = eval(&current, &next, &seed);
    assert_eq!(
        current_out,
        vec![0b0011, 0b0100],
        "warm start unaffected by next"
    );
    assert_eq!(
        changed, 1,
        "changed must flag when any original word differs from next"
    );
}
