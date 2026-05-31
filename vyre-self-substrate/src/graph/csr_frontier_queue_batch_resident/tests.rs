use super::*;
use crate::csr_frontier_queue_resident::upload_resident_csr_queue_graph;
use crate::graph::csr_frontier_queue_scratch::ResidentCsrQueueMaterializer;
use crate::optimizer::dispatcher::{
    DispatchError, OptimizerDispatcher, ResidentDispatchStep, ResidentReadRange,
};
use std::cell::{Cell, RefCell};
use vyre_foundation::ir::Program;

#[derive(Default)]
struct RecordingBatchDispatcher {
    next_handle: Cell<u64>,
    upload_handles: RefCell<Vec<Vec<u64>>>,
    step_handles: RefCell<Vec<Vec<Vec<u64>>>>,
    freed: RefCell<Vec<u64>>,
}

impl OptimizerDispatcher for RecordingBatchDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: batch resident queue tests should not use non-resident dispatch.".to_string(),
        ))
    }

    fn alloc_resident(&self, _byte_len: usize) -> Result<u64, DispatchError> {
        let handle = self.next_handle.get() + 1;
        self.next_handle.set(handle);
        Ok(handle)
    }

    fn upload_resident_many(&self, _uploads: &[(u64, &[u8])]) -> Result<(), DispatchError> {
        Ok(())
    }

    fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        self.upload_handles
            .borrow_mut()
            .push(uploads.iter().map(|(handle, _)| *handle).collect());
        self.step_handles
            .borrow_mut()
            .push(steps.iter().map(|step| step.handle_ids.to_vec()).collect());
        outputs.clear();
        outputs.extend(read_ranges.iter().map(|range| vec![0u8; range.byte_len]));
        Ok(())
    }

    fn free_resident(&self, handle: u64) -> Result<(), DispatchError> {
        self.freed.borrow_mut().push(handle);
        Ok(())
    }
}

#[test]
fn batch_queries_initialize_queue_len_on_device() {
    let dispatcher = RecordingBatchDispatcher::default();
    let graph = upload_resident_csr_queue_graph(&dispatcher, 2, &[0, 0, 0], &[], &[])
        .expect("Fix: zero-edge resident CSR graph is valid");
    let mut scratch = ResidentCsrQueueBatchScratch::default();
    let first = [1u32];
    let second = [2u32];
    let frontiers: [&[u32]; 2] = [&first, &second];
    let mut outputs = Vec::new();

    run_resident_csr_queue_batch_into(
        &dispatcher,
        &graph,
        &mut scratch,
        &frontiers,
        2,
        u32::MAX,
        &mut outputs,
    )
    .expect("Fix: recording dispatcher should complete resident CSR queue batch");

    let expected_uploads: Vec<u64> = scratch
        .handles
        .iter()
        .map(|handles| handles.frontier)
        .collect();
    assert_eq!(
        dispatcher
            .upload_handles
            .borrow()
            .last()
            .cloned()
            .expect("Fix: expected one resident upload sequence"),
        expected_uploads,
        "batch CSR queue traversal must only upload per-query frontier bytes; queue_len and output clear must stay device-side"
    );
    let steps = dispatcher
        .step_handles
        .borrow()
        .last()
        .cloned()
        .expect("Fix: expected one resident step sequence");
    assert_eq!(
        steps.len(),
        6,
        "atomic resident CSR queue batches should clear, compact, then traverse per query; frontier_to_queue clears queue_len itself"
    );
    assert_eq!(steps[0], vec![scratch.handles[0].frontier_out]);
    assert_eq!(
        steps[1],
        vec![
            scratch.handles[0].frontier,
            scratch.handles[0].active_queue,
            scratch.handles[0].queue_len,
        ]
    );
    assert_eq!(steps[3], vec![scratch.handles[1].frontier_out]);
    assert_eq!(
        steps[4],
        vec![
            scratch.handles[1].frontier,
            scratch.handles[1].active_queue,
            scratch.handles[1].queue_len,
        ]
    );
    assert_eq!(outputs, vec![vec![0; 4], vec![0; 4]]);
}

#[test]
fn large_batch_queries_use_word_prefix_queue_materializer() {
    let dispatcher = RecordingBatchDispatcher::default();
    let node_count = 8_193u32;
    let edge_offsets = vec![0u32; node_count as usize + 1];
    let graph = upload_resident_csr_queue_graph(&dispatcher, node_count, &edge_offsets, &[], &[])
        .expect("Fix: zero-edge large resident CSR graph is valid");
    let words = vyre_primitives::bitset::bitset_words(node_count) as usize;
    let first = vec![1u32; words];
    let second = vec![0u32; words];
    let frontiers: [&[u32]; 2] = [&first, &second];
    let mut scratch = ResidentCsrQueueBatchScratch::default();
    let mut outputs = Vec::new();

    run_resident_csr_queue_batch_into(
        &dispatcher,
        &graph,
        &mut scratch,
        &frontiers,
        8,
        u32::MAX,
        &mut outputs,
    )
    .expect("Fix: recording dispatcher should complete large resident CSR queue batch");

    assert_eq!(
        scratch
            .shape
            .expect("Fix: batch scratch shape should be retained")
            .materializer,
        ResidentCsrQueueMaterializer::DeterministicWordPrefix
    );
    assert_eq!(scratch.word_count_handle_sets.len(), frontiers.len());
    assert_eq!(scratch.word_prefix_queue_handle_sets.len(), frontiers.len());
    assert_eq!(scratch.queue_handle_sets.len(), frontiers.len());
    let steps = dispatcher
        .step_handles
        .borrow()
        .last()
        .cloned()
        .expect("Fix: expected one resident step sequence");
    assert_eq!(steps.len(), 8);
    assert_eq!(steps[0], vec![scratch.handles[0].frontier_out]);
    assert_eq!(
        steps[1],
        scratch.word_count_handle_sets[0].as_slice(),
        "large batch query must run word popcount scan before queue scatter"
    );
    assert_eq!(
        steps[2],
        scratch.word_prefix_queue_handle_sets[0].as_slice(),
        "large batch query must run deterministic word-prefix scatter"
    );
    assert_eq!(steps[4], vec![scratch.handles[1].frontier_out]);
    assert_eq!(steps[5], scratch.word_count_handle_sets[1].as_slice());
    assert_eq!(
        steps[6],
        scratch.word_prefix_queue_handle_sets[1].as_slice()
    );
    assert_eq!(
        outputs,
        vec![
            vec![0; words * std::mem::size_of::<u32>()],
            vec![0; words * std::mem::size_of::<u32>()],
        ]
    );
}

#[test]
fn multiblock_batch_queries_scan_block_offsets_once_per_query() {
    let dispatcher = RecordingBatchDispatcher::default();
    let node_count = 32_897u32;
    let edge_offsets = vec![0u32; node_count as usize + 1];
    let graph = upload_resident_csr_queue_graph(&dispatcher, node_count, &edge_offsets, &[], &[])
        .expect("Fix: zero-edge multiblock resident CSR graph is valid");
    let words = vyre_primitives::bitset::bitset_words(node_count) as usize;
    let mut first = vec![0u32; words];
    first[0] = 1;
    first[1028] = 1;
    let second = vec![0u32; words];
    let frontiers: [&[u32]; 2] = [&first, &second];
    let mut scratch = ResidentCsrQueueBatchScratch::default();
    let mut outputs = Vec::new();

    run_resident_csr_queue_batch_into(
        &dispatcher,
        &graph,
        &mut scratch,
        &frontiers,
        8,
        u32::MAX,
        &mut outputs,
    )
    .expect("Fix: recording dispatcher should complete multiblock resident CSR queue batch");

    assert_eq!(scratch.word_count_handle_sets.len(), frontiers.len());
    assert_eq!(
        scratch.word_block_offsets_handle_sets.len(),
        frontiers.len()
    );
    assert_eq!(scratch.word_prefix_queue_handle_sets.len(), frontiers.len());
    let steps = dispatcher
        .step_handles
        .borrow()
        .last()
        .cloned()
        .expect("Fix: expected one resident step sequence");
    assert_eq!(steps.len(), 10);
    assert_eq!(steps[0], vec![scratch.handles[0].frontier_out]);
    assert_eq!(steps[1], scratch.word_count_handle_sets[0].as_slice());
    assert_eq!(
        steps[2],
        scratch.word_block_offsets_handle_sets[0].as_slice(),
        "multiblock batch query must scan block offsets before scatter"
    );
    assert_eq!(
        steps[3],
        scratch.word_prefix_queue_handle_sets[0].as_slice()
    );
    assert_eq!(steps[5], vec![scratch.handles[1].frontier_out]);
    assert_eq!(steps[6], scratch.word_count_handle_sets[1].as_slice());
    assert_eq!(
        steps[7],
        scratch.word_block_offsets_handle_sets[1].as_slice()
    );
    assert_eq!(
        steps[8],
        scratch.word_prefix_queue_handle_sets[1].as_slice()
    );
    assert_eq!(
        outputs,
        vec![
            vec![0; words * std::mem::size_of::<u32>()],
            vec![0; words * std::mem::size_of::<u32>()],
        ]
    );
}

#[test]
fn generated_batch_dispatch_tables_reuse_capacity_across_calls() {
    let dispatcher = RecordingBatchDispatcher::default();
    let graph = upload_resident_csr_queue_graph(&dispatcher, 4, &[0, 0, 0, 0, 0], &[], &[])
        .expect("Fix: zero-edge resident CSR graph is valid");
    let mut scratch = ResidentCsrQueueBatchScratch::default();
    let first = [1_u32];
    let second = [2_u32];
    let frontiers: [&[u32]; 2] = [&first, &second];
    let mut outputs = Vec::new();

    run_resident_csr_queue_batch_into(
        &dispatcher,
        &graph,
        &mut scratch,
        &frontiers,
        4,
        u32::MAX,
        &mut outputs,
    )
    .expect("Fix: first resident CSR queue batch dispatch should succeed");
    let retained_capacities = (
        scratch.clear_handle_sets.capacity(),
        scratch.word_count_handle_sets.capacity(),
        scratch.word_block_offsets_handle_sets.capacity(),
        scratch.queue_handle_sets.capacity(),
        scratch.word_prefix_queue_handle_sets.capacity(),
        scratch.traverse_handle_sets.capacity(),
        scratch.read_ranges.capacity(),
    );

    run_resident_csr_queue_batch_into(
        &dispatcher,
        &graph,
        &mut scratch,
        &frontiers,
        4,
        u32::MAX,
        &mut outputs,
    )
    .expect("Fix: second resident CSR queue batch dispatch should reuse prepared scratch");

    assert_eq!(
        (
            scratch.clear_handle_sets.capacity(),
            scratch.word_count_handle_sets.capacity(),
            scratch.word_block_offsets_handle_sets.capacity(),
            scratch.queue_handle_sets.capacity(),
            scratch.word_prefix_queue_handle_sets.capacity(),
            scratch.traverse_handle_sets.capacity(),
            scratch.read_ranges.capacity(),
        ),
        retained_capacities,
        "resident batch sequence tables must retain allocation capacity across repeated dispatches"
    );
    assert_eq!(scratch.clear_handle_sets.len(), frontiers.len());
    assert_eq!(scratch.word_count_handle_sets.len(), 0);
    assert_eq!(scratch.word_block_offsets_handle_sets.len(), 0);
    assert_eq!(scratch.queue_handle_sets.len(), frontiers.len());
    assert_eq!(scratch.word_prefix_queue_handle_sets.len(), 0);
    assert_eq!(scratch.traverse_handle_sets.len(), frontiers.len());
    assert_eq!(scratch.read_ranges.len(), frontiers.len());

    scratch
        .free(&dispatcher)
        .expect("Fix: resident CSR batch scratch free should release query handles");
    assert!(scratch.clear_handle_sets.is_empty());
    assert!(scratch.word_count_handle_sets.is_empty());
    assert!(scratch.word_block_offsets_handle_sets.is_empty());
    assert!(scratch.queue_handle_sets.is_empty());
    assert!(scratch.word_prefix_queue_handle_sets.is_empty());
    assert!(scratch.traverse_handle_sets.is_empty());
    assert!(scratch.read_ranges.is_empty());
}

#[test]
fn generated_batch_scratch_free_releases_each_handle_once_in_first_seen_order() {
    for seed in 0..4096_u64 {
        let dispatcher = RecordingBatchDispatcher::default();
        let base = 40_000 + seed * 16;
        let mut scratch = ResidentCsrQueueBatchScratch::default();
        scratch.handles.push(ResidentCsrQueueBatchQueryHandles {
            frontier: base,
            active_queue: base + 1,
            queue_len: base,
            frontier_out: base + 2,
            word_partials: None,
            block_totals: None,
        });
        scratch.handles.push(ResidentCsrQueueBatchQueryHandles {
            frontier: base + 2,
            active_queue: base + 3,
            queue_len: base + 3,
            frontier_out: base + 4,
            word_partials: Some(base + 5),
            block_totals: Some(base + 5),
        });
        scratch
            .free(&dispatcher)
            .expect("Fix: batch scratch free dedup");
        assert_eq!(
            dispatcher.freed.borrow().as_slice(),
            &[base, base + 1, base + 2, base + 3, base + 4, base + 5]
        );
    }
}
