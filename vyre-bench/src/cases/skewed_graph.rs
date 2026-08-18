//! Shared deterministic topology and sparse-queue sizing for irregular graph benchmarks.

use crate::api::case::BenchError;
use crate::cases::mix32;

/// Return the shared heavy-tailed degree used by IFDS and CSR fixtures.
pub(crate) fn skewed_degree(source: u32, ugly_hub_degree: u32) -> u32 {
    if source % 4096 == 0 {
        ugly_hub_degree
    } else if source % 257 == 0 {
        24
    } else if source % 31 == 0 {
        8
    } else if source % 7 == 0 {
        3
    } else {
        1
    }
}

/// Select a deterministic target in a power-of-two node space.
pub(crate) fn skewed_target(node_count: u32, source: u32, edge: u32) -> u32 {
    let mask = node_count - 1;
    match edge & 7 {
        0 => source.wrapping_add((edge + 1).wrapping_mul(17)) & mask,
        1 => source.wrapping_sub((edge + 3).wrapping_mul(11)) & mask,
        _ => {
            let salt = edge.wrapping_mul(0x9E37_79B9).rotate_left((edge & 15) + 1);
            mix32(source ^ salt ^ source.rotate_left(edge & 15)) & mask
        }
    }
}

/// Convert an observed active-source count into a queue capacity with contextual errors.
pub(crate) fn sparse_queue_capacity(
    active_sources: u64,
    empty_error: &str,
    overflow_context: &str,
) -> Result<u32, BenchError> {
    if active_sources == 0 {
        return Err(BenchError::EnvironmentInvalid(empty_error.to_string()));
    }
    u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{overflow_context} active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })
}

/// Materialize an active source queue from a packed frontier bitset.
pub(crate) fn materialize_active_frontier_queue(
    frontier_in: &[u32],
    node_count: u32,
    active_sources: u64,
    queue_capacity: usize,
    context: &str,
) -> Result<Vec<u32>, BenchError> {
    let mut active_queue = Vec::new();
    let expected = u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{context} active source count {active_sources} exceeds u32 indexing. Fix: split the frontier."
        ))
    })?;
    let seen = vyre_reference::composition_witness::frontier_to_queue_witness_into(
        frontier_in,
        node_count,
        queue_capacity,
        &mut active_queue,
    );
    if seen != expected || active_queue.len() < expected as usize {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} queue_capacity {queue_capacity} cannot hold {expected} active sources. Fix: increase queue capacity."
        )));
    }
    Ok(active_queue)
}
