//! Caller-owned buffers reused by semantic persistent-BFS execution.

use crate::graph::persistent_bfs::{
    copy_persistent_bfs_seed_frontier_into, PersistentBfsStaticInputKey,
};
use vyre_megakernel::SemanticExecutionError;

/// Caller-owned semantic execution scratch for persistent BFS expansion.
#[derive(Debug, Default)]
pub struct PersistentBfsGpuScratch {
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) changed: Vec<u32>,
    pub(super) converged: Vec<u32>,
    pub(super) static_input_key: Option<PersistentBfsStaticInputKey>,
}

pub(super) fn copy_frontier_seed_into(
    frontier_out: &mut Vec<u32>,
    frontier_in: &[u32],
    context: &'static str,
) -> Result<(), SemanticExecutionError> {
    copy_persistent_bfs_seed_frontier_into(
        frontier_out,
        frontier_in,
        context,
        SemanticExecutionError::Backend,
    )
}
