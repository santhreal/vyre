//! Sequential mathematical witnesses for CSR frontier steps, closure, and sharding.

use super::csr::{
    csr_backward_traverse_witness, for_each_active_edge, prepare_copied_buffer,
    prepare_output_buffer, validate_csr_inputs,
};

/// Sequential in-place CSR forward-closure step with a changed flag writing into caller storage.
pub fn csr_forward_or_changed_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    output: &mut Vec<u32>,
) -> u32 {
    let (_, words) = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask);
    prepare_copied_buffer(output, words.max(1), frontier);
    if edge_offsets.is_empty() {
        return 0;
    }
    let mut changed = 0;
    for_each_active_edge(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        allow_mask,
        |source, dst| {
            if output[source / 32] & (1 << (source % 32)) != 0 {
                let bit = 1 << (dst % 32);
                let word = &mut output[dst / 32];
                if *word & bit == 0 {
                    *word |= bit;
                    changed = 1;
                }
            }
        },
    );
    changed
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
    let mut output = Vec::new();
    let changed = csr_forward_or_changed_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        &mut output,
    );
    (output, changed)
}

/// Iterate the in-place forward step with an iteration callback into caller storage.
#[allow(clippy::too_many_arguments)]
pub fn csr_forward_or_changed_closure_with_step_hook_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iterations: u32,
    mut on_step: impl FnMut(u32),
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    let words = ((node_count as usize).div_ceil(32)).max(1);
    prepare_copied_buffer(current, words, seed);
    prepare_output_buffer(next, words);
    if edge_offsets.is_empty() || max_iterations == 0 {
        return;
    }
    for iteration in 0..max_iterations {
        on_step(iteration);
        let changed = csr_forward_or_changed_witness_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            current,
            allow_mask,
            next,
        );
        std::mem::swap(current, next);
        if changed == 0 {
            break;
        }
    }
}

/// Iterate the in-place forward step until stability or the iteration bound into caller storage.
#[allow(clippy::too_many_arguments)]
pub fn csr_forward_or_changed_closure_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iterations: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    csr_forward_or_changed_closure_with_step_hook_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        seed,
        allow_mask,
        max_iterations,
        |_| {},
        current,
        next,
    );
}

/// Iterate the in-place forward step until stability or the iteration bound.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_forward_or_changed_closure_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iterations: u32,
) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    csr_forward_or_changed_closure_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        seed,
        allow_mask,
        max_iterations,
        &mut current,
        &mut next,
    );
    current
}

/// Iterate the in-place forward step with an iteration callback.
#[allow(clippy::too_many_arguments)]
pub fn csr_forward_or_changed_closure_with_step_hook_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iterations: u32,
    on_step: impl FnMut(u32),
) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    csr_forward_or_changed_closure_with_step_hook_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        seed,
        allow_mask,
        max_iterations,
        on_step,
        &mut current,
        &mut next,
    );
    current
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

/// Sequential queue-driven CSR expansion witness into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_strided_forward_witness_into(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
    frontier: &mut Vec<u32>,
) {
    let words = (node_count as usize).div_ceil(32);
    prepare_output_buffer(frontier, words);
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
            let Some((&destination, &kind)) = edge_targets.get(edge).zip(edge_kind_mask.get(edge))
            else {
                break;
            };
            if destination < node_count && kind & allow_mask != 0 {
                frontier[(destination / 32) as usize] |= 1_u32 << (destination % 32);
            }
        }
    }
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
    let mut frontier = Vec::with_capacity((node_count as usize).div_ceil(32));
    csr_queue_strided_forward_witness_into(
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_count,
        allow_mask,
        &mut frontier,
    );
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
        if let Some(base) = (tile * 256 + byte).checked_mul(destination_words as usize) {
            for (destination, output_word) in output.iter_mut().enumerate() {
                if let Some(&value) = lut.get(base + destination) {
                    *output_word |= value;
                }
            }
        }
    }
    output
}

/// Sequential dense row-scan bitmatrix Boolean step writing into caller storage.
///
/// `frontier_in` is a packed bitset over `node_count` nodes; `adj_rows_dense` is the reverse
/// adjacency matrix where destination `d` has a bitmask row of length `div_ceil(node_count, 32)` words.
pub fn dense_bitmatrix_step_witness_into(
    frontier_in: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
    out: &mut Vec<u32>,
) {
    let words = (node_count as usize).div_ceil(32);
    out.clear();
    out.resize(words, 0);
    for d in 0..node_count as usize {
        let row_start = d * words;
        let mut hit: u32 = 0;
        for w in 0..words {
            let adj = adj_rows_dense.get(row_start + w).copied().unwrap_or(0);
            let frontier = frontier_in.get(w).copied().unwrap_or(0);
            hit |= adj & frontier;
        }
        if hit != 0 {
            out[d / 32] |= 1 << (d % 32);
        }
    }
}

/// Sequential dense row-scan bitmatrix Boolean step.
#[must_use]
pub fn dense_bitmatrix_step_witness(
    frontier_in: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
) -> Vec<u32> {
    let words = (node_count as usize).div_ceil(32);
    let mut out = Vec::with_capacity(words);
    dense_bitmatrix_step_witness_into(frontier_in, adj_rows_dense, node_count, &mut out);
    out
}

/// One forward CSR step that retains the input frontier and reports growth.
#[must_use]
pub fn csr_forward_step_with_change_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    csr_forward_or_changed_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_masks,
        frontier,
        allow_mask,
    )
}

/// Sequential union of one forward and one backward CSR frontier step writing into caller storage.
pub fn csr_bidirectional_step_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    let (node_count_usize, words) =
        validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_masks);
    prepare_output_buffer(out, words);
    if edge_offsets.is_empty() {
        return;
    }

    for source in 0..node_count_usize {
        let source_active = frontier
            .get(source / 32)
            .is_some_and(|word| word & (1 << (source % 32)) != 0);
        if !source_active {
            continue;
        }
        let start = edge_offsets[source] as usize;
        let end = edge_offsets[source + 1] as usize;
        for edge in start..end {
            if edge_kind_masks[edge] & allow_mask == 0 {
                continue;
            }
            let destination = edge_targets[edge] as usize;
            if destination < node_count_usize {
                out[destination / 32] |= 1 << (destination % 32);
            }
        }
    }

    for source in 0..node_count_usize {
        let start = edge_offsets[source] as usize;
        let end = edge_offsets[source + 1] as usize;
        let reaches_active = (start..end).any(|edge| {
            if edge_kind_masks[edge] & allow_mask == 0 {
                return false;
            }
            let destination = edge_targets[edge] as usize;
            destination < node_count_usize
                && frontier
                    .get(destination / 32)
                    .is_some_and(|word| word & (1 << (destination % 32)) != 0)
        });
        if reaches_active {
            out[source / 32] |= 1 << (source % 32);
        }
    }
}

/// Union of one forward and one backward CSR frontier step.
#[must_use]
pub fn csr_bidirectional_step_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = (node_count as usize).div_ceil(32);
    let mut out = Vec::with_capacity(words);
    csr_bidirectional_step_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_masks,
        frontier,
        allow_mask,
        &mut out,
    );
    out
}

/// Sequential backward step with change flag witness.
#[must_use]
pub fn csr_backward_step_with_change_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    seed: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let mut output = seed.to_vec();
    let reached = csr_backward_traverse_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_masks,
        seed,
        allow_mask,
    );
    let mut changed = 0;
    for (word, reached) in output.iter_mut().zip(reached) {
        let prior = *word;
        *word |= reached;
        changed |= u32::from(*word != prior);
    }
    (output, changed)
}

/// Sequential transitive backward closure witness.
#[must_use]
pub fn csr_backward_closure_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    seed: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let mut frontier = seed.to_vec();
    loop {
        let previous = frontier.clone();
        let step = csr_backward_traverse_witness(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_masks,
            &frontier,
            allow_mask,
        );
        for (word, reached) in frontier.iter_mut().zip(step) {
            *word |= reached;
        }
        if frontier == previous {
            return frontier;
        }
    }
}

/// Sequential transitive bidirectional closure witness writing into caller storage and scratch.
#[allow(clippy::too_many_arguments)]
pub fn csr_bidirectional_closure_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    let (_, words) = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_masks);
    assert!(seed.len() <= words, "frontier fits the node domain");
    prepare_copied_buffer(current, words, seed);
    prepare_output_buffer(next, words);

    for _ in 0..max_iters {
        csr_bidirectional_step_witness_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_masks,
            current,
            allow_mask,
            next,
        );
        let mut changed = false;
        for (c, s) in current.iter_mut().zip(next.iter()) {
            let old = *c;
            *c |= *s;
            if *c != old {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Sequential transitive bidirectional closure witness.
#[must_use]
pub fn csr_bidirectional_closure_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    csr_bidirectional_closure_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_masks,
        seed,
        allow_mask,
        max_iters,
        &mut current,
        &mut next,
    );
    current
}

/// Sequential mathematical witness for partitioning active vertices across contiguous shard ranges writing into caller storage.
pub fn partition_frontier_by_vertex_witness_into(
    frontier_in: &[u32],
    node_count: u32,
    shard_count: usize,
    shards_out: &mut Vec<Vec<u32>>,
) -> Result<(), String> {
    if shard_count == 0 {
        return Err(
            "csr_frontier_shard: shard_count must be >= 1. Fix: pass at least one device shard to partition the frontier across.".to_string(),
        );
    }
    let words = (node_count as usize).div_ceil(32);
    if frontier_in.len() != words {
        return Err(format!(
            "csr_frontier_shard: frontier_in has {} word(s) but node_count {node_count} needs {words} (bitset_words). Fix: size the frontier bitset to the graph before sharding.",
            frontier_in.len()
        ));
    }
    shards_out.clear();
    shards_out.resize(shard_count, vec![0u32; words]);
    for v in 0..node_count {
        let word = (v >> 5) as usize;
        let bit = 1u32 << (v & 31);
        if frontier_in[word] & bit != 0 {
            let shard = ((u64::from(v) * shard_count as u64) / u64::from(node_count)) as usize;
            let shard = shard.min(shard_count - 1);
            shards_out[shard][word] |= bit;
        }
    }
    Ok(())
}

/// Sequential mathematical witness for partitioning active vertices across contiguous shard ranges.
#[must_use]
pub fn partition_frontier_by_vertex_witness(
    frontier_in: &[u32],
    node_count: u32,
    shard_count: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let mut shards = Vec::new();
    partition_frontier_by_vertex_witness_into(frontier_in, node_count, shard_count, &mut shards)?;
    Ok(shards)
}

/// Sequential mathematical witness for merging per-shard frontier bitsets by bitwise OR writing into caller storage.
pub fn merge_frontier_out_witness_into(
    shards: &[Vec<u32>],
    words: usize,
    merged_out: &mut Vec<u32>,
) -> Result<(), String> {
    for (index, shard) in shards.iter().enumerate() {
        if shard.len() != words {
            return Err(format!(
                "csr_frontier_shard: shard {index} frontier_out has {} word(s), expected {words}. Fix: every shard must expand into a frontier bitset sized to the whole graph.",
                shard.len()
            ));
        }
    }
    merged_out.clear();
    merged_out.resize(words, 0);
    for shard in shards {
        for (slot, value) in merged_out.iter_mut().zip(shard.iter()) {
            *slot |= *value;
        }
    }
    Ok(())
}

/// Sequential mathematical witness for merging per-shard frontier bitsets by bitwise OR.
#[must_use]
pub fn merge_frontier_out_witness(shards: &[Vec<u32>], words: usize) -> Result<Vec<u32>, String> {
    let mut merged = Vec::new();
    merge_frontier_out_witness_into(shards, words, &mut merged)?;
    Ok(merged)
}

/// Sequential mathematical witness for running one forward frontier-expansion level sharded across device shards.
pub fn frontier_step_sharded_witness(
    frontier_in: &[u32],
    node_count: u32,
    shard_count: usize,
    mut expand: impl FnMut(usize, &[u32]) -> Result<Vec<u32>, String>,
) -> Result<Vec<u32>, String> {
    let words = (node_count as usize).div_ceil(32);
    let partitions = partition_frontier_by_vertex_witness(frontier_in, node_count, shard_count)?;
    let mut outputs = Vec::with_capacity(shard_count);
    for (shard_index, masked_frontier_in) in partitions.iter().enumerate() {
        let out = expand(shard_index, masked_frontier_in)?;
        if out.len() != words {
            return Err(format!(
                "csr_frontier_shard: shard {shard_index} expand returned a {}-word frontier_out, expected {words} for node_count {node_count}. Fix: the per-shard expansion must write a full-graph frontier bitset.",
                out.len()
            ));
        }
        outputs.push(out);
    }
    merge_frontier_out_witness(&outputs, words)
}
