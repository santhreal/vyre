//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "graph-dispatch")]
mod semantic_execution_support;

#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_libs::graph::dispatch::persistent_bfs::{
    bfs_expand_via_with_scratch_into, PersistentBfsGpuScratch,
};
use vyre_reference::composition_witness::csr_forward_or_changed_witness_into;

const CASES: usize = 1024;

#[test]
fn sweep_graph_persistent_bfs_volume_oracle_matrix() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut output = Vec::new();
    for case_index in 0..CASES {
        let groups = csr_sweep::declared_groups();
        let group = &groups[case_index % groups.len()];
        let mut case = csr_sweep::generate(group, case_index as u64);
        let expected_words = case.node_count.div_ceil(32) as usize;
        if case.frontier.len() != expected_words {
            case.frontier = vec![0u32; expected_words];
        }
        let has_valid_seed = (0..case.node_count).any(|node| {
            let word = (node / 32) as usize;
            (case.frontier[word] & (1u32 << (node % 32))) != 0
        });
        if !has_valid_seed {
            let start = (case_index as u32) % case.node_count;
            case.frontier[(start / 32) as usize] |= 1u32 << (start % 32);
        }
        assert_eq!(
            case.frontier.len(),
            expected_words,
            "frontier words must match ceil(node_count / 32)"
        );
        let max_iterations = 16;
        let mut expected = case.frontier.clone();
        let mut step_buffer = Vec::new();
        for _ in 0..max_iterations {
            let changed = csr_forward_or_changed_witness_into(
                case.node_count,
                &case.offsets,
                &case.targets,
                &case.masks,
                &expected,
                case.allow_mask,
                &mut step_buffer,
            );
            expected = step_buffer.clone();
            if changed == 0 {
                break;
            }
        }
        bfs_expand_via_with_scratch_into(
            &dispatcher,
            &semantic_execution_support::policy(),
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
