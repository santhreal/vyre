//! Sequential mathematical witnesses for CSR graph structures and frontier operations.

use std::collections::VecDeque;

/// Sequential mathematical witness for CSR (Compressed Sparse Row) graph breadth-first traversal.
///
/// Computes shortest distances in unweighted graphs from `source` node to all reachable nodes.
/// Unreachable nodes receive `u32::MAX`.
#[must_use]
pub fn csr_bfs_witness(
    node_count: usize,
    row_offsets: &[u32],
    col_indices: &[u32],
    source: usize,
) -> Vec<u32> {
    assert!(
        row_offsets.len() >= node_count + 1,
        "row_offsets must have at least node_count + 1 entries"
    );

    let mut distances = vec![u32::MAX; node_count];
    if source >= node_count {
        return distances;
    }

    distances[source] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let dist_u = distances[u];
        let start = row_offsets[u] as usize;
        let end = row_offsets[u + 1] as usize;

        for &v in &col_indices[start..end] {
            let v = v as usize;
            if v < node_count && distances[v] == u32::MAX {
                distances[v] = dist_u + 1;
                queue.push_back(v);
            }
        }
    }

    distances
}

/// Sequential mathematical witness for one-step forward traversal over CSR graph.
#[must_use]
pub fn csr_forward_traverse_witness(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let node_count = node_count as usize;
    assert!(row_offsets.len() > node_count, "complete CSR offset table");
    let edge_count = row_offsets[node_count] as usize;
    assert!(
        col_indices.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "complete CSR edge buffers"
    );
    let mut next_frontier = vec![0_u32; node_count.div_ceil(32)];
    for source in 0..node_count {
        let source_active = frontier
            .get(source / 32)
            .is_some_and(|word| word & (1 << (source % 32)) != 0);
        if !source_active {
            continue;
        }
        let start = row_offsets[source] as usize;
        let end = row_offsets[source + 1] as usize;
        for edge in start..end {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let destination = col_indices[edge] as usize;
            if destination < node_count {
                next_frontier[destination / 32] |= 1 << (destination % 32);
            }
        }
    }
    next_frontier
}

/// Sequential mathematical witness for one reverse CSR frontier step.
#[must_use]
pub fn csr_backward_traverse_witness(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let node_count = node_count as usize;
    assert!(row_offsets.len() > node_count, "complete CSR offset table");
    let edge_count = row_offsets[node_count] as usize;
    assert!(
        col_indices.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "complete CSR edge buffers"
    );
    let mut previous_frontier = vec![0_u32; node_count.div_ceil(32)];
    for source in 0..node_count {
        let start = row_offsets[source] as usize;
        let end = row_offsets[source + 1] as usize;
        let reaches_active = (start..end).any(|edge| {
            if edge_kind_mask[edge] & allow_mask == 0 {
                return false;
            }
            let destination = col_indices[edge] as usize;
            destination < node_count
                && frontier
                    .get(destination / 32)
                    .is_some_and(|word| word & (1 << (destination % 32)) != 0)
        });
        if reaches_active {
            previous_frontier[source / 32] |= 1 << (source % 32);
        }
    }
    previous_frontier
}

/// Sequential mathematical witness for persistent fixed-point iteration.
pub fn persistent_fixpoint_witness<F>(
    seed: &[u32],
    max_iterations: u32,
    mut step: F,
) -> (Vec<u32>, u32)
where
    F: FnMut(&[u32]) -> Vec<u32>,
{
    let mut current = seed.to_vec();
    let mut iters = 0;
    for i in 0..max_iterations {
        let next = step(&current);
        iters = i + 1;
        if next == current {
            break;
        }
        current = next;
    }
    (current, iters)
}

/// Sequential mathematical witness for resolve family / nodeset filtering.
#[must_use]
pub fn resolve_family_witness(node_tags: &[u32], family_mask: u32) -> Vec<u32> {
    let words = (node_tags.len() + 31) / 32;
    let mut bitset = vec![0u32; words];
    for (idx, &tag) in node_tags.iter().enumerate() {
        if (tag & family_mask) != 0 {
            bitset[idx / 32] |= 1 << (idx % 32);
        }
    }
    bitset
}

/// Sequential mathematical witness for CSR transitive closure with step hook.
pub fn csr_closure_with_step_hook_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    allow_mask: u32,
    max_iters: u32,
    seed_frontier: &[u32],
    mut step_hook: impl FnMut(&[u32]),
) -> Vec<u32> {
    let mut current = seed_frontier.to_vec();
    let num_words = (node_count as usize).div_ceil(32);
    if current.len() < num_words {
        current.resize(num_words, 0);
    }
    for _ in 0..max_iters {
        let mut next = current.clone();
        let mut changed = false;
        for src in 0..node_count {
            let src_word = (src / 32) as usize;
            let src_bit = 1u32 << (src % 32);
            if (current[src_word] & src_bit) == 0 {
                continue;
            }
            let start = edge_offsets[src as usize] as usize;
            let end = edge_offsets[src as usize + 1] as usize;
            for edge in start..end {
                if (edge_kind_mask[edge] & allow_mask) == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                if dst >= node_count {
                    continue;
                }
                let dst_word = (dst / 32) as usize;
                let dst_bit = 1u32 << (dst % 32);
                let old = next[dst_word];
                next[dst_word] |= dst_bit;
                if next[dst_word] != old {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        step_hook(&next);
        current = next;
    }
    current
}

/// Materialize active packed-frontier bits into an ascending queue.
///
/// The queue is truncated to `queue_capacity`; the returned count records every
/// active node below `node_count`, including entries beyond that capacity.
#[must_use]
pub fn frontier_to_queue_witness(
    frontier: &[u32],
    node_count: u32,
    queue_capacity: usize,
) -> (Vec<u32>, u32) {
    let mut queue = Vec::with_capacity(queue_capacity.min(node_count as usize));
    let mut active_count = 0_u32;
    for node in 0..node_count {
        let word = (node / 32) as usize;
        let bit = node % 32;
        if frontier.get(word).is_some_and(|value| value & (1 << bit) != 0) {
            if queue.len() < queue_capacity {
                queue.push(node);
            }
            active_count = active_count.saturating_add(1);
        }
    }
    (queue, active_count)
}

/// Sequential in-place CSR forward-closure step with a changed flag.
#[must_use]
pub fn csr_forward_or_changed_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let node_count = node_count as usize;
    assert!(edge_offsets.len() > node_count, "complete CSR offset table");
    let edge_count = edge_offsets[node_count] as usize;
    assert!(
        edge_targets.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "complete CSR edge buffers"
    );
    let mut output = frontier.to_vec();
    output.resize(node_count.div_ceil(32), 0);
    let mut changed = 0;
    for source in 0..node_count {
        if output[source / 32] & (1 << (source % 32)) == 0 {
            continue;
        }
        for edge in edge_offsets[source] as usize..edge_offsets[source + 1] as usize {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let destination = edge_targets[edge] as usize;
            if destination < node_count {
                let bit = 1 << (destination % 32);
                let word = &mut output[destination / 32];
                if *word & bit == 0 {
                    *word |= bit;
                    changed = 1;
                }
            }
        }
    }
    (output, changed)
}

/// Sum the degrees of active CSR frontier nodes with wrapping arithmetic.
#[must_use]
pub fn csr_frontier_degree_sum_witness(
    frontier: &[u32],
    edge_offsets: &[u32],
    node_count: u32,
) -> u32 {
    (0..node_count).fold(0_u32, |sum, node| {
        let active = frontier
            .get((node / 32) as usize)
            .is_some_and(|word| word & (1_u32 << (node % 32)) != 0);
        if !active {
            return sum;
        }
        let start = edge_offsets.get(node as usize).copied().unwrap_or(0);
        let end = edge_offsets
            .get(node as usize + 1)
            .copied()
            .unwrap_or(start);
        sum.wrapping_add(end.wrapping_sub(start))
    })
}

/// Sequential queue-driven CSR expansion witness.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_strided_forward_witness(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    let mut frontier = vec![0_u32; node_count.div_ceil(32) as usize];
    for &source in active_queue.iter().take(queue_len as usize) {
        if source >= node_count {
            continue;
        }
        let start = edge_offsets.get(source as usize).copied().unwrap_or(0) as usize;
        let end = edge_offsets
            .get(source as usize + 1)
            .copied()
            .unwrap_or(start as u32) as usize;
        for edge in start..end {
            let Some((&destination, &kind)) =
                edge_targets.get(edge).zip(edge_kind_mask.get(edge))
            else {
                break;
            };
            if destination < node_count && kind & allow_mask != 0 {
                frontier[(destination / 32) as usize] |= 1_u32 << (destination % 32);
            }
        }
    }
    frontier
}

/// Dense boolean-semiring matvec witness over byte-indexed Four-Russians tiles.
#[must_use]
pub fn dense_boolean_matvec_witness(
    frontier: &[u32],
    lut: &[u32],
    tile_count: u32,
    destination_words: u32,
) -> Vec<u32> {
    let mut output = vec![0_u32; destination_words as usize];
    for tile in 0..tile_count as usize {
        let byte = frontier
            .get(tile / 4)
            .map_or(0, |word| ((word >> ((tile % 4) * 8)) & 0xff) as usize);
        let base = (tile * 256 + byte) * destination_words as usize;
        for (destination, output_word) in output.iter_mut().enumerate() {
            if let Some(&value) = lut.get(base + destination) {
                *output_word |= value;
            }
        }
    }
    output
}
