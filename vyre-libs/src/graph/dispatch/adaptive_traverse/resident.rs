use crate::graph::dispatch::resident_handles::{
    impl_resident_graph_accessors, impl_resident_graph_free,
};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Device-resident graph layouts for adaptive sparse/dense traversal.
#[derive(Debug, Clone)]
pub struct ResidentAdaptiveTraversalGraph {
    pub(crate) node_count: u32,
    pub(crate) edge_count: u32,
    pub(crate) max_row_degree: u32,
    pub(crate) high_degree_source_count: u32,
    pub(crate) words: usize,
    pub(crate) layout_hash: u64,
    pub(crate) handles: [u64; 4],
}

/// Device-resident CSR graph for adaptive sparse-queue traversal.
#[derive(Debug, Clone)]
pub struct ResidentAdaptiveSparseQueueGraph {
    pub(crate) node_count: u32,
    pub(crate) edge_count: u32,
    pub(crate) max_row_degree: u32,
    pub(crate) high_degree_source_count: u32,
    pub(crate) words: usize,
    pub(crate) layout_hash: u64,
    pub(crate) handles: [u64; 3],
}

impl_resident_graph_free!(
    ResidentAdaptiveSparseQueueGraph,
    "resident adaptive sparse-queue graph"
);
impl_resident_graph_free!(
    ResidentAdaptiveTraversalGraph,
    "resident adaptive traversal graph"
);

impl_resident_graph_accessors!(ResidentAdaptiveSparseQueueGraph);
impl_resident_graph_accessors!(ResidentAdaptiveTraversalGraph);

/// Device-resident Four-Russians dense traversal LUT for adaptive graph waves.
#[derive(Debug, Clone)]
pub struct ResidentAdaptiveFourRussiansDenseGraph {
    pub(crate) node_count: u32,
    pub(crate) words: usize,
    pub(crate) layout_hash: u64,
    pub(crate) lut_handle: u64,
}

impl ResidentAdaptiveFourRussiansDenseGraph {
    /// Number of graph nodes.
    #[must_use]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Number of u32 words per frontier bitset.
    #[must_use]
    pub fn words(&self) -> usize {
        self.words
    }

    /// Stable in-session hash of the dense LUT source layout.
    #[must_use]
    pub fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Resident handle for the dense byte-tile LUT.
    #[must_use]
    pub fn lut_handle(&self) -> u64 {
        self.lut_handle
    }

    /// Free graph-resident LUT buffer.
    ///
    /// # Errors
    ///
    /// Returns the backend free failure, if any.
    pub fn free(self, dispatcher: &dyn ProgramDispatcher) -> Result<(), DispatchError> {
        dispatcher.free_resident(self.lut_handle)
    }
}

impl ResidentAdaptiveSparseQueueGraph {
    /// Stable in-session hash of CSR graph layout.
    #[must_use]
    pub fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Resident handles in adaptive sparse-queue order:
    /// edge_offsets, edge_targets, edge_kind_mask.
    #[must_use]
    pub fn handles(&self) -> [u64; 3] {
        self.handles
    }
}

impl ResidentAdaptiveTraversalGraph {
    /// Stable in-session hash of CSR and dense graph layouts.
    #[must_use]
    pub fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Resident handles in adaptive traversal order:
    /// edge_offsets, edge_targets, edge_kind_mask, adj_rows_dense.
    #[must_use]
    pub fn handles(&self) -> [u64; 4] {
        self.handles
    }
}
