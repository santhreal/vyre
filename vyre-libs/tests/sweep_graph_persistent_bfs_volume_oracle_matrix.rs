//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "graph")]
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use vyre_driver_reference::ReferenceEvalDispatcher;
use vyre_libs::graph::dispatch::persistent_bfs::{
    bfs_expand_via_with_scratch_into, PersistentBfsGpuScratch,
};
use vyre_reference::composition_witness::csr_forward_traverse_witness;

const CASES: usize = 16384;

#[test]
fn sweep_graph_persistent_bfs_volume_oracle_matrix() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut output = Vec::new();
    for case_index in 0..CASES {
        let groups = csr_sweep::declared_groups();
        let group = &groups[case_index % groups.len()];
        let case = csr_sweep::generate(group, case_index as u64);
        let max_iterations = 16;
        let mut expected = case.frontier.clone();
        for _ in 0..max_iterations {
            let derived = csr_forward_traverse_witness(
                case.node_count,
                &case.offsets,
                &case.targets,
                &case.masks,
                &expected,
                case.allow_mask,
            );
            let prior = expected.clone();
            for (word, next) in expected.iter_mut().zip(derived) {
                *word |= next;
            }
            if expected == prior {
                break;
            }
        }
        bfs_expand_via_with_scratch_into(
            &dispatcher,
            case.inputs(max_iterations),
            &case.frontier,
            &mut scratch,
            &mut output,
        )
        .expect("persistent BFS reference dispatch must succeed");
        assert_eq!(
            output, expected,
            "persistent BFS case {case_index} group {} diverged",
            group.name
        );
    }
}
