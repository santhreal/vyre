use std::cell::{Cell, RefCell};
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{
    DispatchError, ProgramDispatcher, ResidentDispatchStep, ResidentReadRange,
};

use super::super::{
    AdaptiveTraversalPlanCacheSnapshot, AdaptiveTraversalResidentScratch,
    ResidentAdaptiveTraversalGraph,
};

#[derive(Default)]
pub(super) struct RecordingResidentDispatcher {
    pub(super) next_handle: Cell<u64>,
    pub(super) alloc_count: Cell<usize>,
    pub(super) alloc_lengths: RefCell<Vec<usize>>,
    pub(super) resident_uploads: RefCell<Vec<(u64, usize)>>,
    pub(super) upload_handles: RefCell<Vec<Vec<u64>>>,
    pub(super) step_handles: RefCell<Vec<Vec<Vec<u64>>>>,
    pub(super) step_grids: RefCell<Vec<Vec<Option<[u32; 3]>>>>,
    pub(super) step_programs: RefCell<Vec<Vec<[u8; 32]>>>,
    pub(super) freed: RefCell<Vec<u64>>,
}

/// The shared resident traversal graph these tests dispatch against.
///
/// Seventeen copies of this literal used to sit across the module's test
/// files, differing in two or three fields each, so a new field on
/// `ResidentAdaptiveTraversalGraph` meant seventeen edits. Each test now
/// states only what it varies: `ResidentAdaptiveTraversalGraph { node_count:
/// 5, ..traversal_graph() }`.
pub(super) fn traversal_graph() -> ResidentAdaptiveTraversalGraph {
    ResidentAdaptiveTraversalGraph {
        node_count: 33,
        edge_count: 8,
        max_row_degree: 2,
        high_degree_source_count: 0,
        words: 2,
        layout_hash: 7,
        handles: [101, 102, 103, 104],
    }
}

impl RecordingResidentDispatcher {
    pub(super) fn last_upload_handles(&self) -> Vec<u64> {
        self.upload_handles
            .borrow()
            .last()
            .cloned()
            .expect("Fix: test dispatcher expected at least one resident upload sequence")
    }

    pub(super) fn last_step_handles(&self) -> Vec<Vec<u64>> {
        self.step_handles
            .borrow()
            .last()
            .cloned()
            .expect("Fix: test dispatcher expected at least one resident dispatch sequence")
    }

    pub(super) fn last_step_grids(&self) -> Vec<Option<[u32; 3]>> {
        self.step_grids
            .borrow()
            .last()
            .cloned()
            .expect("Fix: test dispatcher expected at least one resident dispatch sequence")
    }

    /// Wire fingerprints of the Programs the last dispatch sequence launched,
    /// in launch order.
    pub(super) fn last_step_programs(&self) -> Vec<[u8; 32]> {
        self.step_programs
            .borrow()
            .last()
            .cloned()
            .expect("Fix: test dispatcher expected at least one resident dispatch sequence")
    }

    pub(super) fn resident_alloc_lengths(&self) -> Vec<usize> {
        self.alloc_lengths.borrow().clone()
    }

    pub(super) fn resident_upload_lengths(&self) -> Vec<usize> {
        self.resident_uploads
            .borrow()
            .iter()
            .map(|(_, bytes)| *bytes)
            .collect()
    }

    pub(super) fn assert_no_resident_work(&self) {
        assert_eq!(
            self.alloc_count.get(),
            0,
            "zero-frontier fast paths must not allocate resident scratch"
        );
        assert!(
            self.upload_handles.borrow().is_empty(),
            "zero-frontier fast paths must not upload resident inputs"
        );
        assert!(
            self.step_handles.borrow().is_empty(),
            "zero-frontier fast paths must not dispatch resident kernels"
        );
    }
}

impl ProgramDispatcher for RecordingResidentDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        Err(DispatchError::Rejected(
            "Fix: recording dispatcher only supports resident sequence tests.".to_string(),
        ))
    }

    fn supports_persistent(&self) -> bool {
        true
    }

    fn alloc_resident(&self, byte_len: usize) -> Result<u64, DispatchError> {
        self.alloc_count.set(self.alloc_count.get() + 1);
        self.alloc_lengths.borrow_mut().push(byte_len);
        let handle = self.next_handle.get() + 1;
        self.next_handle.set(handle);
        Ok(handle)
    }

    fn free_resident(&self, handle: u64) -> Result<(), DispatchError> {
        self.freed.borrow_mut().push(handle);
        Ok(())
    }

    fn upload_resident(&self, handle: u64, bytes: &[u8]) -> Result<(), DispatchError> {
        self.resident_uploads
            .borrow_mut()
            .push((handle, bytes.len()));
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
        self.step_grids
            .borrow_mut()
            .push(steps.iter().map(|step| step.grid_override).collect());
        self.step_programs.borrow_mut().push(
            steps
                .iter()
                .map(|step| step.program.fingerprint())
                .collect(),
        );
        outputs.clear();
        outputs.extend(read_ranges.iter().map(|range| vec![0u8; range.byte_len]));
        Ok(())
    }
}

/// Which packed frontier words a sparse-queue run starts from.
///
/// Both sparse-queue suites built these fills by hand, once per case, from the
/// word count the graph implies. The fill is what selects the queue
/// materializer, so it is data the case declares rather than a loop it spells.
pub(super) enum Frontier {
    /// Node 0 only: one active source, one nonzero packed word.
    SingleSource,
    /// Every bit of every packed word: the densest frontier the graph holds.
    AllWords,
    /// The lowest `n` node ids.
    LowNodes(u32),
}

impl Frontier {
    fn packed(&self, words: usize) -> Vec<u32> {
        match *self {
            Frontier::SingleSource => {
                let mut frontier = vec![0; words];
                frontier[0] = 1;
                frontier
            }
            Frontier::AllWords => vec![u32::MAX; words],
            Frontier::LowNodes(count) => {
                let mut frontier = vec![0; words];
                for node in 0..count {
                    frontier[(node / 32) as usize] |= 1 << (node % 32);
                }
                frontier
            }
        }
    }
}

/// One resident sparse-queue dispatch, and everything it produced.
///
/// Every sparse-queue case built the same five values by hand before it could
/// assert anything: a recording dispatcher, a graph, default scratch, a packed
/// frontier, and an output buffer. It then unwrapped the same scratch handles
/// with the same diagnostics. That scaffold is here once; a case declares the
/// graph and the frontier and asserts the contract.
pub(super) struct SparseQueueRun {
    dispatcher: RecordingResidentDispatcher,
    pub(super) scratch: AdaptiveTraversalResidentScratch,
    pub(super) frontier_out: Vec<u32>,
    pub(super) words: usize,
}

impl SparseQueueRun {
    /// Run one step over a synthetic resident graph whose buffers are the
    /// handles the graph literal names.
    pub(super) fn over_graph(
        graph: &ResidentAdaptiveTraversalGraph,
        frontier: &Frontier,
    ) -> Result<Self, DispatchError> {
        let dispatcher = RecordingResidentDispatcher::default();
        let mut scratch = AdaptiveTraversalResidentScratch::default();
        let mut frontier_out = Vec::new();
        super::super::adaptive_traverse_resident_graph_sparse_queue_step_with_scratch_into(
            &dispatcher,
            graph,
            &frontier.packed(graph.words),
            u32::MAX,
            &mut scratch,
            &mut frontier_out,
        )?;
        Ok(Self {
            dispatcher,
            scratch,
            frontier_out,
            words: graph.words,
        })
    }

    /// `[frontier_in, frontier_out, queue_len]`, in the order the resident step
    /// allocates them.
    pub(super) fn frontier_scratch(&self) -> [u64; 3] {
        self.scratch
            .handles
            .expect("Fix: sparse-queue resident step must allocate frontier/queue-len handles")
    }

    pub(super) fn active_queue(&self) -> u64 {
        self.scratch
            .queue_handle
            .expect("Fix: sparse-queue resident step must allocate an active queue")
    }

    /// `(word_partials, block_totals)`, allocated only by the deterministic
    /// word-prefix materializer.
    pub(super) fn word_prefix(&self) -> (u64, u64) {
        (
            self.scratch.word_partials_handle.expect(
                "Fix: deterministic word-prefix sparse-queue step must allocate word partials",
            ),
            self.scratch.word_block_totals_handle.expect(
                "Fix: deterministic word-prefix sparse-queue step must allocate block totals",
            ),
        )
    }

    pub(super) fn allocated_word_prefix(&self) -> bool {
        self.scratch.word_partials_handle.is_some()
            || self.scratch.word_block_totals_handle.is_some()
    }

    pub(super) fn steps(&self) -> Vec<Vec<u64>> {
        self.dispatcher.last_step_handles()
    }

    pub(super) fn uploads(&self) -> Vec<u64> {
        self.dispatcher.last_upload_handles()
    }

    pub(super) fn plan_cache(&self) -> AdaptiveTraversalPlanCacheSnapshot {
        self.scratch.plan_cache_snapshot()
    }

    pub(super) fn high_degree_queue(&self) -> (u64, u64) {
        (
            self.scratch
                .high_queue_handle
                .expect("Fix: mixed split traversal must allocate a high-degree queue"),
            self.scratch
                .high_len_handle
                .expect("Fix: mixed split traversal must allocate a high-degree queue length"),
        )
    }

    pub(super) fn grids(&self) -> Vec<Option<[u32; 3]>> {
        self.dispatcher.last_step_grids()
    }

    pub(super) fn alloc_lengths(&self) -> Vec<usize> {
        self.dispatcher.resident_alloc_lengths()
    }

    pub(super) fn alloc_count(&self) -> usize {
        self.dispatcher.alloc_count.get()
    }

    pub(super) fn freed_handles(&self) -> Vec<u64> {
        self.dispatcher.freed.borrow().clone()
    }

    /// Run another step through the same dispatcher and scratch, which is how
    /// the reuse contracts observe what the previous run left behind.
    pub(super) fn step_again(
        &mut self,
        graph: &ResidentAdaptiveTraversalGraph,
        frontier: &Frontier,
    ) -> Result<(), DispatchError> {
        super::super::adaptive_traverse_resident_graph_sparse_queue_step_with_scratch_into(
            &self.dispatcher,
            graph,
            &frontier.packed(graph.words),
            u32::MAX,
            &mut self.scratch,
            &mut self.frontier_out,
        )
    }
}

/// A synthetic resident traversal graph `node_count` nodes wide, with no edges
/// and the packed frontier width that node count implies.
///
/// Every sparse-queue case recomputed that width by hand before it could write
/// the graph literal. A case that varies degree or layout spreads this:
/// `ResidentAdaptiveTraversalGraph { max_row_degree: 64, ..graph_of(2048) }`.
pub(super) fn graph_of(node_count: u32) -> ResidentAdaptiveTraversalGraph {
    ResidentAdaptiveTraversalGraph {
        node_count,
        edge_count: 0,
        max_row_degree: 0,
        words: vyre_primitives::bitset::bitset_words(node_count) as usize,
        ..traversal_graph()
    }
}
