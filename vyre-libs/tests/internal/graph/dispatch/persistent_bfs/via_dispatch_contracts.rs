use super::super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
use crate::test_parity_oracles::{policy, NeverDispatches, StaticOutputs};
use vyre_megakernel::SemanticExecutionError;

const BFS_CONTRACT: &str = "persistent BFS semantic execution";

fn linear_graph<'a>(
    offsets: &'a [u32],
    targets: &'a [u32],
    kinds: &'a [u32],
    max_iters: u32,
) -> CsrClosureInputs<'a> {
    CsrClosureInputs::allow_all(
        CsrGraphView {
            node_count: 4,
            edge_offsets: offsets,
            edge_targets: targets,
            edge_kind_mask: kinds,
        },
        max_iters,
    )
}

#[test]
fn semantic_execution_decodes_frontier_and_status_words() {
    let executor = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_inputs(&[9]);
    let offsets = [0, 1, 2, 3, 3];
    let targets = [1, 2, 3];
    let kinds = [u32::MAX; 3];

    let (frontier, changed, converged) = bfs_expand_via(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 4),
        &[0b0001],
    )
    .expect("Fix: valid persistent BFS semantic execution must succeed");

    assert_eq!(frontier, vec![0b1111]);
    assert_eq!((changed, converged), (1, 1));
}

#[test]
fn semantic_scratch_reuses_buffers_and_refreshes_frontier() {
    let executor = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .recording_input(5);
    let offsets = [0, 1, 2, 3, 3];
    let targets = [1, 2, 3];
    let kinds = [u32::MAX; 3];
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut frontier = Vec::new();

    bfs_expand_via_with_scratch_into(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 4),
        &[0b0001],
        &mut scratch,
        &mut frontier,
    )
    .expect("Fix: first semantic BFS execution must succeed");
    let capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();

    bfs_expand_via_with_scratch_into(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 4),
        &[0b0010],
        &mut scratch,
        &mut frontier,
    )
    .expect("Fix: changed frontier must refresh the dynamic semantic input");

    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        capacities
    );
    assert_eq!(executor.recorded(), vec![vec![0b0001], vec![0b0010]]);
}

#[test]
fn zero_iteration_budget_returns_seed_without_execution() {
    let executor = NeverDispatches("Fix: zero-iteration persistent BFS must not execute");
    let offsets = [0, 1, 2, 3, 3];
    let targets = [1, 2, 3];
    let kinds = [u32::MAX; 3];

    let (frontier, changed, converged) = bfs_expand_via(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 0),
        &[0b0101],
    )
    .expect("Fix: zero-iteration BFS must return the validated seed");

    assert_eq!(frontier, vec![0b0101]);
    assert_eq!((changed, converged), (0, 0));
}

#[test]
fn empty_graph_is_a_converged_no_work_execution() {
    let executor = NeverDispatches("Fix: empty persistent BFS must not execute");
    let inputs = CsrClosureInputs::allow_all(
        CsrGraphView {
            node_count: 0,
            edge_offsets: &[0],
            edge_targets: &[],
            edge_kind_mask: &[],
        },
        4,
    );

    let (frontier, changed, converged) = bfs_expand_via(&executor, &policy(), inputs, &[])
        .expect("Fix: empty persistent BFS must report its trivial fixpoint");
    assert!(frontier.is_empty());
    assert_eq!((changed, converged), (0, 1));
}

#[test]
fn invalid_seed_is_rejected_before_execution() {
    let executor = NeverDispatches("Fix: malformed persistent BFS seed must not execute");
    let offsets = [0, 1, 2, 3, 3];
    let targets = [1, 2, 3];
    let kinds = [u32::MAX; 3];
    let error = bfs_expand_via(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 4),
        &[],
    )
    .expect_err("a short frontier seed must be rejected");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));
}

#[test]
fn malformed_semantic_status_is_rejected() {
    let executor = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[2]),
            u32_slice_to_le_bytes(&[1]),
        ],
    );
    let offsets = [0, 1, 2, 3, 3];
    let targets = [1, 2, 3];
    let kinds = [u32::MAX; 3];
    let error = bfs_expand_via(
        &executor,
        &policy(),
        linear_graph(&offsets, &targets, &kinds, 4),
        &[1],
    )
    .expect_err("a non-boolean changed word must be rejected");
    assert!(matches!(error, SemanticExecutionError::Backend(_)));
}
