//! Resident CSR frontier-queue execution.
//!
//! This module owns the reusable device-resident graph and scratch protocol for
//! sparse dataflow-dependent traversal: upload CSR graph buffers once, then run
//! repeated frontier queries by refreshing only frontier/scratch/output state.

mod query;
mod upload;

#[cfg(test)]
mod tests;

pub use query::run_resident_csr_queue_query_into;
pub use upload::upload_resident_csr_queue_graph;

use crate::graph::dispatch::csr_frontier_queue_programs::ResidentCsrQueuePrograms;
use crate::graph::dispatch::csr_frontier_queue_scratch::{
    ResidentCsrQueueMaterializer, ResidentCsrQueueSlots,
};
use crate::graph::dispatch::resident_handles::free_unique_resident_handles;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Device-resident CSR graph for queue-driven sparse traversal.
#[derive(Debug, Clone)]
pub struct ResidentCsrQueueGraph {
    node_count: u32,
    edge_count: u32,
    max_row_degree: u32,
    high_degree_source_count: u32,
    words: usize,
    edge_offsets_handle: u64,
    edge_targets_handle: u64,
    edge_kind_mask_handle: u64,
}

impl ResidentCsrQueueGraph {
    /// Number of graph nodes.
    #[must_use]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Number of physical CSR edges.
    #[must_use]
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }

    /// Largest CSR row degree.
    #[must_use]
    pub fn max_row_degree(&self) -> u32 {
        self.max_row_degree
    }

    /// Number of rows at or above the resident mixed-split high-degree threshold.
    #[must_use]
    pub fn high_degree_source_count(&self) -> u32 {
        self.high_degree_source_count
    }

    /// Number of u32 words in each frontier bitset.
    #[must_use]
    pub fn words(&self) -> usize {
        self.words
    }

    /// Resident edge-offset buffer handle.
    #[must_use]
    pub fn edge_offsets_handle(&self) -> u64 {
        self.edge_offsets_handle
    }

    /// Resident edge-target buffer handle.
    #[must_use]
    pub fn edge_targets_handle(&self) -> u64 {
        self.edge_targets_handle
    }

    /// Resident edge-kind-mask buffer handle.
    #[must_use]
    pub fn edge_kind_mask_handle(&self) -> u64 {
        self.edge_kind_mask_handle
    }

    /// Free graph-resident buffers.
    pub fn free(self, dispatcher: &dyn ProgramDispatcher) -> Result<(), DispatchError> {
        free_unique_resident_handles(
            dispatcher,
            &[
                self.edge_offsets_handle,
                self.edge_targets_handle,
                self.edge_kind_mask_handle,
            ],
            "resident CSR queue graph",
        )
    }
}

/// Reusable resident scratch for CSR queue traversal queries.
#[derive(Debug, Default)]
pub struct ResidentCsrQueueScratch {
    slots: Option<ResidentCsrQueueSlots>,
    shape: Option<ResidentCsrQueueScratchShape>,
    frontier_bytes: Vec<u8>,
    readbacks: Vec<Vec<u8>>,
    programs: ResidentCsrQueuePrograms,
}

impl ResidentCsrQueueScratch {
    /// Free scratch-resident buffers.
    pub fn free(&mut self, dispatcher: &dyn ProgramDispatcher) -> Result<(), DispatchError> {
        let Some(slots) = self.slots.take() else {
            return Ok(());
        };
        self.shape = None;
        self.frontier_bytes.clear();
        self.readbacks.clear();
        self.programs.clear();
        let mut handles_to_free = Vec::new();
        slots.extend_handles(&mut handles_to_free);
        free_unique_resident_handles(dispatcher, &handles_to_free, "resident CSR queue scratch")
    }
}

/// Allocation shape the retained scratch slots satisfy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentCsrQueueScratchShape {
    queue_capacity: u32,
    high_queue_capacity: u32,
    frontier_bytes: usize,
    materializer: ResidentCsrQueueMaterializer,
}
