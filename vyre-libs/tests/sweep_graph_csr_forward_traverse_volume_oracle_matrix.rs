//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;
mod graph_sweep_support;

use vyre_libs::graph::csr_forward_traverse;

const CASES: usize = 16384;

#[test]
fn sweep_graph_csr_forward_traverse_volume_oracle_matrix() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_sweep::tuples(
        "single_source_all_kinds",
        CASES as u64,
        0xF074D4D01,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let expected = csr_sweep::oracle_forward_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = csr_forward_traverse::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_forward_traverse volume case {case} node_count={node_count}"
        );
    }
}
