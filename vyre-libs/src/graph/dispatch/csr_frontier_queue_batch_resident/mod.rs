//! Batched resident CSR frontier-queue execution.
//!
//! This module owns multi-query sparse traversal over one resident CSR graph:
//! each query gets resident scratch slots, all frontiers are uploaded together,
//! all queue/traverse kernels are submitted as one resident sequence, and all
//! frontier outputs are compactly read back at the end.

mod dispatch;

#[cfg(test)]
mod tests;

pub use dispatch::{run_resident_csr_queue_batch_budgeted_into, run_resident_csr_queue_batch_into};

use crate::graph::dispatch::csr_frontier_queue_programs::ResidentCsrQueuePrograms;
use crate::graph::dispatch::csr_frontier_queue_scratch::{
    ResidentCsrQueueMaterializer, ResidentCsrQueueSlots,
};
use crate::graph::dispatch::resident_handles::free_unique_resident_handles;
use crate::scratch::reserve_vec as reserve_graph_vec;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentReadRange};

/// Reusable resident scratch for batched CSR queue traversal queries.
#[derive(Debug, Default)]
pub struct ResidentCsrQueueBatchScratch {
    slots: Vec<ResidentCsrQueueSlots>,
    shape: Option<ResidentCsrQueueBatchShape>,
    programs: ResidentCsrQueuePrograms,
    frontier_payloads: Vec<Vec<u8>>,
    readbacks: Vec<Vec<u8>>,
    clear_handle_sets: Vec<[u64; 1]>,
    queue_len_handle_sets: Vec<[u64; 1]>,
    word_count_handle_sets: Vec<[u64; 3]>,
    word_block_offsets_handle_sets: Vec<[u64; 1]>,
    queue_handle_sets: Vec<[u64; 3]>,
    atomic_word_queue_handle_sets: Vec<[u64; 4]>,
    word_prefix_queue_handle_sets: Vec<[u64; 5]>,
    traverse_handle_sets: Vec<[u64; 6]>,
    high_len_handle_sets: Vec<[u64; 1]>,
    split_low_handle_sets: Vec<[u64; 8]>,
    high_traverse_handle_sets: Vec<[u64; 6]>,
    read_ranges: Vec<ResidentReadRange>,
}

impl ResidentCsrQueueBatchScratch {
    /// Number of resident per-query scratch slots currently retained.
    #[must_use]
    pub fn resident_query_slots(&self) -> usize {
        self.slots.len()
    }

    /// Total host staging capacity retained for frontier uploads.
    #[must_use]
    pub fn frontier_payload_capacity(&self) -> usize {
        self.frontier_payloads.iter().map(Vec::capacity).sum()
    }

    /// Free all batch scratch resident buffers.
    pub fn free(&mut self, dispatcher: &dyn ProgramDispatcher) -> Result<(), DispatchError> {
        let handle_slots = self.slots.len().checked_mul(8).ok_or_else(|| {
            DispatchError::BackendError(
                "Fix: resident CSR queue batch scratch free handle count overflowed.".to_string(),
            )
        })?;
        let mut handles_to_free = Vec::new();
        reserve_graph_vec(
            &mut handles_to_free,
            handle_slots,
            "resident CSR queue batch scratch free handles",
        )?;
        for slots in self.slots.drain(..) {
            slots.extend_handles(&mut handles_to_free);
        }
        let free_result = free_unique_resident_handles(
            dispatcher,
            &handles_to_free,
            "resident CSR queue batch scratch",
        );
        self.shape = None;
        self.programs.clear();
        self.frontier_payloads.clear();
        self.readbacks.clear();
        self.clear_handle_sets.clear();
        self.queue_len_handle_sets.clear();
        self.word_count_handle_sets.clear();
        self.word_block_offsets_handle_sets.clear();
        self.queue_handle_sets.clear();
        self.atomic_word_queue_handle_sets.clear();
        self.word_prefix_queue_handle_sets.clear();
        self.traverse_handle_sets.clear();
        self.high_len_handle_sets.clear();
        self.split_low_handle_sets.clear();
        self.high_traverse_handle_sets.clear();
        self.read_ranges.clear();
        free_result
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentCsrQueueBatchShape {
    batch_len: usize,
    frontier_bytes: usize,
    queue_capacity: u32,
    high_queue_capacity: u32,
    node_count: u32,
    materializer: ResidentCsrQueueMaterializer,
}
