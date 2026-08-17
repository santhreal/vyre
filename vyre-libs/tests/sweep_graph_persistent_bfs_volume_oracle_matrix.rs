//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "graph")]
use vyre_libs::graph::csr_closure_inputs::CsrClosureInputs;
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;
mod graph_sweep_fixtures;

use vyre_libs::graph::persistent_bfs;

const CASES: usize = 16384;

#[test]
fn sweep_graph_persistent_bfs_volume_oracle_matrix() {
    let mut scratch = persistent_bfs::PersistentBfsResidentScratch::default();
    let mut plan_cache = persistent_bfs::PersistentBfsPlanCacheSnapshot::default();
    for (i, case) in graph_sweep_fixtures::SWEEP_GRAPH_FIXTURES
        .iter()
        .take(CASES)
        .enumerate()
    {
        let inputs = CsrClosureInputs {
            graph: &case.view,
            max_iters: 16,
            allow_mask: 0xFFFF_FFFF,
        };
        let _ = persistent_bfs::persistent_bfs_frontier_via_with_scratch_into(
            inputs,
            &case.seed_frontier,
            &mut scratch,
            &mut plan_cache,
        );
        let _ = i;
    }
}
