//! Tier 3 - Property: differential proptest driving the ACTUAL `predicate::node_kind_eq` IR (the
//! shared u32-per-node → packed-NodeSet filter kernel) through `reference_eval` vs `cpu_ref`. The op
//! had `reference_eval` = 0 in tests/ (its `sweep_predicate_node_kind_oracle_matrix` peer is
//! cpu-vs-cpu).
//!
//! The kernel is one lane per node: it computes `word_idx = v >> 5`, `bit = 1 << (v & 31)`, and
//! `atomic_or`s the bit into `nodeset_out[word_idx]` iff `nodes[v] == kind` — the packed-bitset
//! scatter whose whole reason for existing (per the module doc) is "keep tag predicates from drifting
//! at WORD BOUNDARIES". A wrong word/bit split, a non-atomic OR that loses concurrent sets, or an
//! off-by-one at a 32-node boundary diverges. The sweep runs random node arrays (count 1..=256, single
//! workgroup) with kinds drawn from a SMALL alphabet so matches are dense and multiple lanes hit the
//! same output word concurrently (the exact case a non-atomic OR corrupts). Asserts the full packed
//! NodeSet bit-exact vs `cpu_ref`, plus deterministic all-match / no-match / word-boundary anchors.
#![cfg(feature = "cpu-parity")]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::predicate::node_kind_eq::{cpu_ref, node_kind_eq};

fn run_ir(nodes: &[u32], kind: u32) -> Vec<u32> {
    let node_count = nodes.len() as u32;
    let words = node_count.div_ceil(32).max(1) as usize;
    let program = node_kind_eq("nodes", "nodeset", node_count, kind);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs =
        vyre_reference::reference_eval(&program, &[pack(nodes), pack(&vec![0u32; words])])
            .expect("node_kind_eq reference evaluation must succeed");
    // Sole RW buffer is `nodeset` (binding 1) → results[0].
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn node_kind_eq_ir_matches_cpu_ref(
        // Kinds from a small alphabet (0..8) so matches are dense across the 32-node output words.
        nodes in prop::collection::vec(0u32..8, 1..=256),
        kind in 0u32..8,
    ) {
        let got = run_ir(&nodes, kind);
        let want = cpu_ref(&nodes, kind);
        prop_assert_eq!(&got, &want, "kind={} node_count={}", kind, nodes.len());
    }
}

#[test]
fn node_kind_eq_ir_word_boundary_and_extremes() {
    // Exactly 32, 33, 64, 65 nodes: the div_ceil word-count seam.
    for &n in &[1usize, 31, 32, 33, 64, 65, 96, 200] {
        let all_seven = vec![7u32; n];
        // All match kind 7 → every bit in [0, n) set.
        let got = run_ir(&all_seven, 7);
        assert_eq!(got, cpu_ref(&all_seven, 7), "all-match n={n}");
        // Reconstruct the expected packed all-ones-prefix independently.
        let words = n.div_ceil(32);
        let mut expected = vec![0u32; words];
        for v in 0..n {
            expected[v / 32] |= 1u32 << (v % 32);
        }
        assert_eq!(got, expected, "all-match bit pattern n={n}");

        // No match → empty NodeSet.
        let no_match = run_ir(&all_seven, 3);
        assert_eq!(no_match, vec![0u32; words], "no-match n={n}");
    }
}
