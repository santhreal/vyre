//! Tier 3 - Property: proptest over random exchange graphs for `math::matroid_intersection_full`,
//! locking the multi-augmentation FIX (`BUG-matroid-megakernel-static-graph-oscillates-multi-augmentation`).
//!
//! The exchange graph is STATIC, so the augmenting path P is re-found identically every augmentation
//! and `set_x`'s orbit under "toggle P" is the 2-cycle `{s0, s0^P}`. The IR now emits a SINGLE
//! augmentation and, for `max_augmentations >= 2`, keeps the higher-cardinality endpoint of that cycle
//! (reverting the toggle iff it strictly shrank the set), matching the sequential reference's
//! seen-state / max-cardinality termination at every budget instead of oscillating.
//!
//! This suite runs the ACTUAL IR through `reference_eval` at a RANDOM budget for each of 4000 random
//! (exchange_adj, sources, sinks, seed) instances and asserts it equals an INDEPENDENT host oracle:
//! one Edmonds augmentation via `cpu_ref` (a distinct BFS+toggle implementation), then the exact
//! period-2 keep-or-revert rule. Shrinking auto-minimizes any counterexample. Complements the fixed
//! 250-case self-substrate sweep + the grow/shrink/tie hand-checks with randomized breadth.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::matroid_intersection_full::{cpu_ref, matroid_intersection_full};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Run one dispatch of the IR at `budget` and return the updated `set_x`.
fn run_ir(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed: &[u32],
    n: u32,
    budget: u32,
) -> Vec<u32> {
    let program = matroid_intersection_full(
        "exchange_adj",
        "sources",
        "sinks",
        "set_x",
        "parent",
        "frontier",
        "next_frontier",
        "visited",
        "any_change",
        "path_out",
        "path_len",
        n,
        budget,
    );
    let zeros_n = vec![0u32; n as usize];
    let zero1 = vec![0u32];
    // Buffer order: exchange_adj(0) sources(1) sinks(2) set_x(3, seeded) parent(4) frontier(5)
    // next_frontier(6) visited(7) any_change(8) path_out(9) path_len(10) target_node_buf(11).
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(exchange_adj)),
            Value::from(pack(sources)),
            Value::from(pack(sinks)),
            Value::from(pack(seed)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zero1)),
        ],
    )
    .expect("matroid_intersection_full reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "set_x")
        .expect("matroid_intersection_full must declare output set_x");
    unpack(&outputs[index].to_bytes())[..n as usize].to_vec()
}

/// Independent host oracle: one Edmonds augmentation via `cpu_ref` (a separate BFS+toggle impl), then
/// the exact static-graph period-2 termination, budget 1 toggles unconditionally; budget >= 2 keeps
/// the higher-cardinality endpoint of `{seed, cpu_ref(seed)}` (ties -> the toggled state), i.e. reverts
/// the toggle iff it strictly shrank the selected set.
fn expected(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed: &[u32],
    n: u32,
    budget: u32,
) -> Vec<u32> {
    let s1 = cpu_ref(exchange_adj, sources, sinks, seed, n as usize);
    if budget <= 1 {
        return s1;
    }
    let count = |v: &[u32]| v.iter().filter(|&&x| x != 0).count();
    if count(&s1) >= count(seed) {
        s1
    } else {
        seed.to_vec()
    }
}

prop_compose! {
    /// A random in-domain matroid instance: `n` in 2..=7, a dense random exchange adjacency, random
    /// source/sink flags, and a seed with the domain precondition applied (a `source` node is a fresh
    /// item so it cannot also be seeded, zero the seed wherever `sources` is set, matching the
    /// self-substrate sweep's valid domain). `budget` in 1..=5 exercises single- AND multi-augmentation.
    fn arb_instance()(n in 2u32..=7)
        (n in Just(n),
         adj in prop::collection::vec(0u32..=1, (n * n) as usize),
         sources in prop::collection::vec(0u32..=1, n as usize),
         sinks in prop::collection::vec(0u32..=1, n as usize),
         seed_raw in prop::collection::vec(0u32..=1, n as usize),
         budget in 1u32..=5)
        -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
        let seed: Vec<u32> = (0..n as usize)
            .map(|i| if sources[i] != 0 { 0 } else { seed_raw[i] })
            .collect();
        (n, adj, sources, sinks, seed, budget)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    #[test]
    fn matroid_ir_matches_period2_oracle_over_random_graphs(
        (n, adj, sources, sinks, seed, budget) in arb_instance()
    ) {
        let got = run_ir(&adj, &sources, &sinks, &seed, n, budget);
        let want = expected(&adj, &sources, &sinks, &seed, n, budget);
        prop_assert_eq!(
            &got, &want,
            "n={} budget={} adj={:?} sources={:?} sinks={:?} seed={:?}: IR {:?} != period-2 oracle {:?}",
            n, budget, adj, sources, sinks, seed, got, want
        );
        // Output is always a 0/1 membership vector.
        prop_assert!(got.iter().all(|&b| b <= 1), "result must be 0/1, got {:?}", got);
    }
}
