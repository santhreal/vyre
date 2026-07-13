//! GPU-IR vs CPU-ref parity for `nfa::subgroup_nfa` (one Thompson-NFA byte step:
//! transition gather + epsilon closure, simulated across one 32-lane subgroup).
//!
//! The kernel is the substrate's only user of `subgroup_shuffle`: each lane owns
//! a 32-bit slice of the 1024-state set, gathers every peer lane's active bits
//! via shuffle, ORs in the transition-table row for the input byte, then runs a
//! bounded epsilon closure (also shuffle-gathered). `reference_eval` derives its
//! grid from the OUTPUT buffer (32 lanes), so it dispatches EXACTLY one 32-lane
//! subgroup, matching the real dispatch and giving `subgroup_shuffle` correct
//! peer semantics. Every shipped test is `cpu_step`-vs-oracle or Program-shape;
//! the actual shuffle-gather IR was never executed. A wrong peer index, a
//! transition/epsilon row miscomputation, or a lane-major/state-major swap all
//! diverge here.
//!
//! Convergence: for `num_states <= 32` the emitted epsilon loop runs `num_states`
//! rounds >= the closure diameter, so it reaches the SAME fixpoint the CPU
//! reference computes with its 1024-round cap. For the cross-lane cases
//! (`num_states > 32`, where the loop caps at 32 rounds) we drive transitions
//! only and leave epsilon empty (a no-op closure), so convergence is trivial and
//! the shuffle-gather across lanes is what is under test. A dedicated anchor
//! exercises a shallow cross-lane epsilon.
#![forbid(unsafe_code)]
#![cfg(all(feature = "nfa", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::nfa::subgroup_nfa::{cpu_step, nfa_step, LANES_PER_SUBGROUP};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

const LANES: usize = LANES_PER_SUBGROUP;

/// Pack a set of active states into the lane-major 32-word state bitset:
/// state `s` lives in word `s / 32`, bit `s % 32`.
fn state_bitset(active: &[usize]) -> Vec<u32> {
    let mut words = vec![0u32; LANES];
    for &s in active {
        words[s / 32] |= 1u32 << (s % 32);
    }
    words
}

/// Build an empty transition table of the right shape for `num_states`.
fn empty_transition(num_states: usize) -> Vec<u32> {
    vec![0u32; num_states * 256 * LANES]
}

/// Set a transition edge `from -(byte)-> to` in the lane-major table:
/// destination state `to` contributes bit `to % 32` to output lane `to / 32`.
fn add_transition(table: &mut [u32], from: usize, byte: u8, to: usize) {
    let idx = from * 256 * LANES + (byte as usize) * LANES + (to / 32);
    table[idx] |= 1u32 << (to % 32);
}

/// Set an epsilon edge `from -> to` in the lane-major epsilon table.
fn add_epsilon(table: &mut [u32], from: usize, to: usize) {
    let idx = from * LANES + (to / 32);
    table[idx] |= 1u32 << (to % 32);
}

/// Drive the real NFA-step IR through `reference_eval` and return the 32-word
/// output state bitset. Buffer binding order: state(0), input(1), transition(2),
/// epsilon(3), out(4, the only ReadWrite buffer).
fn gpu_step(
    state: &[u32],
    byte: u8,
    transition: &[u32],
    epsilon: &[u32],
    num_states: u32,
) -> Vec<u32> {
    let program = nfa_step(
        "nfa_state",
        "nfa_input",
        "nfa_transition",
        "nfa_epsilon",
        "nfa_out",
        num_states,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(state)),
            Value::from(pack(&[byte as u32])),
            Value::from(pack(transition)),
            Value::from(pack(epsilon)),
            Value::from(pack(&vec![0u32; LANES])),
        ],
    )
    .expect("subgroup_nfa reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

/// Random single-lane NFA (num_states <= 32) with both transitions and epsilon,
/// so the emitted epsilon loop (num_states rounds) reaches the CPU fixpoint.
fn generated_single_lane(seed: u64) -> (Vec<u32>, u8, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let num_states = 1 + (next() % 14) as usize; // 1..=14 -> all in word 0
    let byte = (next() % 8) as u8; // small alphabet so edges collide on the byte
    let mut transition = empty_transition(num_states);
    let mut epsilon = vec![0u32; num_states * LANES];
    // Random transitions on a few bytes (only `byte` matters for this step, but
    // populating other bytes proves the byte offset is honored).
    for from in 0..num_states {
        for _ in 0..(next() % 3) {
            let b = (next() % 8) as u8;
            let to = (next() as usize) % num_states;
            add_transition(&mut transition, from, b, to);
        }
        for _ in 0..(next() % 2) {
            let to = (next() as usize) % num_states;
            add_epsilon(&mut epsilon, from, to);
        }
    }
    // Random initial active set.
    let mut active = Vec::new();
    for s in 0..num_states {
        if next() & 1 == 0 {
            active.push(s);
        }
    }
    (
        state_bitset(&active),
        byte,
        transition,
        epsilon,
        num_states as u32,
    )
}

/// Random cross-lane NFA (num_states in 33..=64, spanning 2 state words) with
/// transitions only and EMPTY epsilon, so the closure is a no-op and the
/// shuffle-gather across peer lanes is what is validated.
fn generated_cross_lane(seed: u64) -> (Vec<u32>, u8, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let num_states = 33 + (next() % 2) as usize; // 33..=34 -> spans words 0 and 1
    let byte = (next() % 8) as u8;
    let mut transition = empty_transition(num_states);
    let epsilon = vec![0u32; num_states * LANES];
    for from in 0..num_states {
        for _ in 0..(next() % 3) {
            let b = (next() % 8) as u8;
            let to = (next() as usize) % num_states;
            add_transition(&mut transition, from, b, to);
        }
    }
    let mut active = Vec::new();
    for s in 0..num_states {
        if next() & 1 == 0 {
            active.push(s);
        }
    }
    (
        state_bitset(&active),
        byte,
        transition,
        epsilon,
        num_states as u32,
    )
}

proptest! {
    // Cases kept modest: the emitted kernel is fully unrolled (per-peer x per-bit
    // x epsilon-loop) and runs through the tree-walking interpreter, so each case
    // is heavy. Coverage comes from the input diversity per case, not raw count.
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn ir_matches_cpu_ref_single_lane(seed in any::<u64>()) {
        let (state, byte, transition, epsilon, num_states) = generated_single_lane(seed);
        let expected = cpu_step(&state, byte, &transition, &epsilon, num_states as usize);
        let got = gpu_step(&state, byte, &transition, &epsilon, num_states);
        prop_assert_eq!(got, expected, "single-lane NFA step IR diverged from cpu_step (num_states={})", num_states);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn ir_matches_cpu_ref_cross_lane_transitions(seed in any::<u64>()) {
        let (state, byte, transition, epsilon, num_states) = generated_cross_lane(seed);
        let expected = cpu_step(&state, byte, &transition, &epsilon, num_states as usize);
        let got = gpu_step(&state, byte, &transition, &epsilon, num_states);
        prop_assert_eq!(got, expected, "cross-lane NFA transition IR diverged from cpu_step (num_states={})", num_states);
    }
}

/// Deterministic anchors: cross-lane transition (peer lane 1 -> lane 0), byte
/// selectivity, empty state, and a shallow cross-lane epsilon closure.
#[test]
fn ir_matches_cpu_ref_on_anchor_nfas() {
    let num_states = 40u32; // states 0..40 span words 0 (0..32) and 1 (32..40)

    // Cross-lane transition: active state 35 (word 1, bit 3) transitions on byte 2
    // to state 5 (word 0, bit 5). The gather must shuffle peer lane 1's bit 3.
    let state = state_bitset(&[35]);
    let mut transition = empty_transition(num_states as usize);
    add_transition(&mut transition, 35, 2, 5);
    let epsilon = vec![0u32; num_states as usize * LANES];
    let expected = cpu_step(&state, 2, &transition, &epsilon, num_states as usize);
    assert_eq!(
        expected[0] & (1 << 5),
        1 << 5,
        "cpu_step: state 5 reached in word 0"
    );
    assert_eq!(
        gpu_step(&state, 2, &transition, &epsilon, num_states),
        expected,
        "cross-lane transition gather must match"
    );

    // Byte selectivity: the same edge only fires on byte 2. Stepping byte 3 yields
    // no destination.
    let other = cpu_step(&state, 3, &transition, &epsilon, num_states as usize);
    assert_eq!(other, vec![0u32; LANES], "cpu_step: wrong byte -> empty");
    assert_eq!(
        gpu_step(&state, 3, &transition, &epsilon, num_states),
        other,
        "byte offset must gate the transition in IR too"
    );

    // Empty initial state -> empty output regardless of tables.
    let empty_state = vec![0u32; LANES];
    let empty_out = cpu_step(&empty_state, 2, &transition, &epsilon, num_states as usize);
    assert_eq!(
        empty_out,
        vec![0u32; LANES],
        "cpu_step: no active state -> empty"
    );
    assert_eq!(
        gpu_step(&empty_state, 2, &transition, &epsilon, num_states),
        empty_out,
        "empty state must yield empty output in IR too"
    );

    // Shallow cross-lane epsilon: active state 33 (word 1); transition on byte 0
    // to state 1 (word 0); then epsilon 1 -> 34 (word 1) -> 2 (word 0). Two eps
    // hops, well within the 32-round cap. Exercises epsilon shuffle across words.
    let state2 = state_bitset(&[33]);
    let mut transition2 = empty_transition(num_states as usize);
    add_transition(&mut transition2, 33, 0, 1);
    let mut epsilon2 = vec![0u32; num_states as usize * LANES];
    add_epsilon(&mut epsilon2, 1, 34);
    add_epsilon(&mut epsilon2, 34, 2);
    let closure = cpu_step(&state2, 0, &transition2, &epsilon2, num_states as usize);
    assert_eq!(
        closure[0] & (1 << 1),
        1 << 1,
        "cpu_step: state 1 via transition"
    );
    assert_eq!(
        closure[1] & (1 << (34 - 32)),
        1 << 2,
        "cpu_step: state 34 via eps"
    );
    assert_eq!(
        closure[0] & (1 << 2),
        1 << 2,
        "cpu_step: state 2 via eps chain"
    );
    assert_eq!(
        gpu_step(&state2, 0, &transition2, &epsilon2, num_states),
        closure,
        "cross-lane epsilon closure must match"
    );
}
