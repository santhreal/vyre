//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
use vyre_libs::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;
mod graph_sweep_fixtures;

use vyre_libs::graph::persistent_bfs;

const CASES: usize = 16384;

#[test]
fn sweep_graph_persistent_bfs_volume_oracle_matrix() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_sweep::tuples(
        "single_source_all_kinds",
        CASES as u64,
        0xBF5FEFE57,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let max_iters = 1 + (case % 8) as u32;
        let expected = csr_sweep::oracle_persistent_closure(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let actual = persistent_bfs::cpu_ref(
            CsrClosureInputs {
                graph: CsrGraphView {
                    node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                allow_mask,
                max_iters,
            },
            &frontier,
        );
        assert_eq!(actual, expected, "Fix: persistent_bfs volume case {case}");
    }
}
