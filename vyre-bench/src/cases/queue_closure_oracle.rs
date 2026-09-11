//! The CPU reference closure over a queued CSR frontier.
//!
//! Both queue-closure benchmarks need the same answer from the host: propagate a
//! seed queue wave by wave until no destination bit is new, recording the wave
//! lengths the GPU sequence will be measured against. Only the edge-kind mask,
//! the label in each error, and the hint for raising the wave bound differ
//! between the families.

use crate::api::case::BenchError;

/// One CPU closure, plus the wave profile the GPU run is judged against.
pub(crate) struct QueueClosureOracle {
    pub(crate) output: Vec<u32>,
    pub(crate) changed: u32,
    pub(crate) iterations: u32,
    pub(crate) total_queue_pops: u64,
    pub(crate) max_wave_queue_len: u32,
    pub(crate) wave_queue_lengths: Vec<u32>,
}

/// A queued CSR closure the host can run.
pub(crate) struct QueueClosureGraph<'a> {
    pub(crate) node_count: u32,
    pub(crate) edge_offsets: &'a [u32],
    pub(crate) edge_targets: &'a [u32],
    pub(crate) edge_kind_mask: &'a [u32],
    /// Packed seed frontier the accumulator starts from.
    pub(crate) frontier_in: &'a [u32],
    /// Sources the first wave pops, materialized by the caller so the error it
    /// raises names the caller's own fixture.
    pub(crate) seed_queue: Vec<u32>,
    pub(crate) allow_mask: u32,
}

/// Close `graph` under `allow_mask`, one queue wave at a time.
///
/// `label` names the family in every error, e.g. `"IFDS"`. `wave_bound_hint`
/// completes the non-convergence fix, e.g. `"raise CLOSURE_MAX_ITERS"`, because
/// the families raise the bound in different places.
pub(crate) fn queue_closure_oracle(
    graph: QueueClosureGraph<'_>,
    max_iters: u32,
    queue_capacity: u32,
    label: &str,
    wave_bound_hint: &str,
) -> Result<QueueClosureOracle, BenchError> {
    let QueueClosureGraph {
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        seed_queue,
        allow_mask,
    } = graph;

    let capacity = queue_capacity as usize;
    let mut accumulator = frontier_in.to_vec();
    let mut current = seed_queue;
    let mut next = Vec::with_capacity(capacity.min(node_count as usize));
    let mut iterations = 0_u32;
    let mut total_queue_pops = 0_u64;
    let mut max_wave_queue_len = current.len() as u32;
    let mut wave_queue_lengths = Vec::new();

    while !current.is_empty() && iterations < max_iters {
        wave_queue_lengths.push(current.len() as u32);
        max_wave_queue_len = max_wave_queue_len.max(current.len() as u32);
        total_queue_pops = total_queue_pops.saturating_add(current.len() as u64);
        next.clear();
        for &src in &current {
            if src >= node_count {
                continue;
            }
            let start = edge_offsets[src as usize] as usize;
            let end = edge_offsets[src as usize + 1] as usize;
            for edge in start..end {
                if edge_kind_mask[edge] & allow_mask == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                if dst >= node_count {
                    continue;
                }
                let dst_word = dst as usize / 32;
                let dst_bit = 1_u32 << (dst % 32);
                if accumulator[dst_word] & dst_bit != 0 {
                    continue;
                }
                accumulator[dst_word] |= dst_bit;
                if next.len() >= capacity {
                    return Err(BenchError::EnvironmentInvalid(format!(
                        "{label} queue closure next wave exceeded queue_capacity={queue_capacity}. Fix: increase queue capacity or shard closure waves."
                    )));
                }
                next.push(dst);
            }
        }
        iterations = iterations.saturating_add(1);
        std::mem::swap(&mut current, &mut next);
    }

    if !current.is_empty() {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{label} queue closure did not converge within {max_iters} queue waves. Fix: {wave_bound_hint} or use a smaller fixture diameter."
        )));
    }

    Ok(QueueClosureOracle {
        changed: u32::from(accumulator != frontier_in),
        output: accumulator,
        iterations,
        total_queue_pops,
        max_wave_queue_len,
        wave_queue_lengths,
    })
}
