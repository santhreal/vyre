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
        if start <= end && end <= col_indices.len() {
            for &v in &col_indices[start..end] {
                let v = v as usize;
                if v < node_count && distances[v] == u32::MAX {
                    distances[v] = dist_u.saturating_add(1);
                    queue.push_back(v);
                }
            }
        }
    }

    distances
}

fn validate_csr_inputs(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
) -> (usize, usize) {
    let node_count = node_count as usize;
    assert!(row_offsets.len() > node_count, "complete CSR offset table");
    let edge_count = row_offsets[node_count] as usize;
    assert!(
        col_indices.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "complete CSR edge buffers"
    );
    assert!(
        row_offsets[..=node_count].windows(2).all(|row| {
            row[0] <= row[1] && usize::try_from(row[1]).is_ok_and(|end| end <= edge_count)
        }),
        "non-monotonic CSR offsets within edge buffers"
    );
    (node_count, node_count.div_ceil(32))
}

/// Sequential mathematical witness for one-step forward traversal over CSR graph writing into caller storage.
pub fn csr_forward_traverse_witness_into(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    let (node_count, words) =
        validate_csr_inputs(node_count, row_offsets, col_indices, edge_kind_mask);
    if out.capacity() < words {
        out.reserve(words.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(words, 0_u32);
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
                out[destination / 32] |= 1 << (destination % 32);
            }
        }
    }
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
    let mut out = Vec::new();
    csr_forward_traverse_witness_into(
        node_count, row_offsets, col_indices, edge_kind_mask, frontier, allow_mask, &mut out,
    );
    out
}

/// Sequential mathematical witness for one reverse CSR frontier step writing into caller storage.
pub fn csr_backward_traverse_witness_into(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    let (node_count, words) =
        validate_csr_inputs(node_count, row_offsets, col_indices, edge_kind_mask);
    if out.capacity() < words {
        out.reserve(words.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(words, 0_u32);
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
            out[source / 32] |= 1 << (source % 32);
        }
    }
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
    let words = (node_count as usize).div_ceil(32);
    let mut out = Vec::with_capacity(words);
    csr_backward_traverse_witness_into(
        node_count,
        row_offsets,
        col_indices,
        edge_kind_mask,
        frontier,
        allow_mask,
        &mut out,
    );
    out
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

/// Sequential mathematical witness for persistent fixed-point iteration into caller-provided ping-pong buffers.
pub fn try_persistent_fixpoint_into_witness<F>(
    seed: &[u32],
    max_iterations: u32,
    mut step: F,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<u32, String>
where
    F: FnMut(&[u32], &mut [u32]),
{
    let additional_current = seed.len().saturating_sub(current.capacity());
    let additional_next = seed.len().saturating_sub(next.capacity());
    current
        .try_reserve_exact(additional_current)
        .map_err(|err| format!("failed to reserve current fixpoint buffer: {err}"))?;
    next.try_reserve_exact(additional_next)
        .map_err(|err| format!("failed to reserve next fixpoint buffer: {err}"))?;
    current.clear();
    current.extend_from_slice(seed);
    next.clear();
    next.resize(seed.len(), 0);
    for iter in 0..max_iterations {
        next.fill(0);
        step(current, next);
        if next == current {
            return Ok(iter + 1);
        }
        std::mem::swap(current, next);
    }
    Ok(max_iterations)
}

/// Sequential mathematical witness for persistent fixed-point iteration into caller-provided ping-pong buffers (panicking wrapper).
///
/// # Panics
///
/// Panics if scratch memory reservation fails for `current` or `next`.
pub fn persistent_fixpoint_into_witness<F>(
    seed: &[u32],
    max_iterations: u32,
    step: F,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> u32
where
    F: FnMut(&[u32], &mut [u32]),
{
    try_persistent_fixpoint_into_witness(seed, max_iterations, step, current, next)
        .expect("persistent_fixpoint_into_witness scratch reservation failed")
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

/// Sequential mathematical witness for filtering nodes matching a specific kind into a bitset.
#[must_use]
pub fn node_kind_eq_witness(nodes: &[u32], kind: u32) -> Vec<u32> {
    let mut out = Vec::new();
    node_kind_eq_witness_into(nodes, kind, &mut out);
    out
}

/// Sequential mathematical witness for filtering nodes matching a specific kind into caller-owned bitset.
pub fn node_kind_eq_witness_into(nodes: &[u32], kind: u32, out: &mut Vec<u32>) {
    let needed = (nodes.len() + 31) / 32;
    out.clear();
    out.resize(needed, 0);
    for (node, &value) in nodes.iter().enumerate() {
        if value == kind {
            out[node / 32] |= 1_u32 << (node % 32);
        }
    }
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
    assert!(
        edge_offsets.len() > node_count as usize,
        "complete CSR offset table"
    );
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
            let end = (edge_offsets[src as usize + 1] as usize)
                .min(edge_targets.len())
                .min(edge_kind_mask.len());
            if start >= end {
                continue;
            }
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

/// Sequential bounded masked CSR closure writing into caller storage and scratch.
#[allow(clippy::too_many_arguments)]
pub fn csr_persistent_closure_witness_with_scratch_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iterations: u32,
    frontier_out: &mut Vec<u32>,
    step_scratch: &mut Vec<u32>,
) -> u32 {
    let (_, words) = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask);
    if frontier_out.capacity() < words {
        frontier_out.reserve(words.saturating_sub(frontier_out.len()));
    }
    frontier_out.clear();
    frontier_out.extend_from_slice(frontier_in);
    frontier_out.resize(words, 0);

    let mut changed = 0;
    if edge_offsets.is_empty() || max_iterations == 0 {
        step_scratch.clear();
        step_scratch.resize(words, 0);
        return 0;
    }

    for _ in 0..max_iterations {
        csr_forward_traverse_witness_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_out,
            allow_mask,
            step_scratch,
        );
        let mut step_changed = false;
        for (word, reached) in frontier_out.iter_mut().zip(step_scratch.iter()) {
            let previous = *word;
            *word |= *reached;
            step_changed |= *word != previous;
        }
        if step_changed {
            changed = 1;
        } else {
            break;
        }
    }
    step_scratch.clear();
    step_scratch.resize(words, 0);
    changed
}

/// Sequential bounded masked CSR closure with a sticky growth flag.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_persistent_closure_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    max_iterations: u32,
) -> (Vec<u32>, u32) {
    let mut frontier_out = Vec::with_capacity((node_count as usize).div_ceil(32));
    let mut step_scratch = Vec::new();
    let changed = csr_persistent_closure_witness_with_scratch_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        max_iterations,
        &mut frontier_out,
        &mut step_scratch,
    );
    (frontier_out, changed)
}

/// Detailed result of a bounded persistent CSR closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrPersistentClosureWitness {
    /// Accumulated frontier after the last attempted step.
    pub frontier: Vec<u32>,
    /// Sticky flag indicating that at least one step added a node.
    pub changed: u32,
    /// Whether a no-growth step proved the fixed point.
    pub converged: bool,
    /// Number of steps attempted.
    pub stop_iteration: u32,
    /// Active-node count after each attempted iteration through convergence.
    pub active_per_iteration: Vec<u32>,
    /// Active-node count per iteration through max_iterations.
    pub active_density: Vec<u32>,
}

/// Compute bounded persistent CSR closure with convergence and density detail.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_persistent_closure_detailed_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iterations: u32,
) -> CsrPersistentClosureWitness {
    let mut frontier = seed.to_vec();
    frontier.resize((node_count as usize).div_ceil(32), 0);
    let mut changed = 0;
    let mut converged = false;
    let mut stop_iteration = 0;
    let mut active_per_iteration = Vec::new();
    let mut step = Vec::new();

    if !edge_offsets.is_empty() {
        for iteration in 0..max_iterations {
            csr_forward_traverse_witness_into(
                node_count,
                edge_offsets,
                edge_targets,
                edge_kind_mask,
                &frontier,
                allow_mask,
                &mut step,
            );
            let mut step_changed = false;
            for (word, reached) in frontier.iter_mut().zip(&step) {
                let previous = *word;
                *word |= *reached;
                step_changed |= *word != previous;
            }
            stop_iteration = iteration + 1;
            active_per_iteration.push(frontier.iter().map(|word| word.count_ones()).sum());
            if step_changed {
                changed = 1;
            } else {
                converged = true;
                break;
            }
        }
    }

    let fill = active_per_iteration
        .last()
        .copied()
        .unwrap_or_else(|| seed.iter().map(|word| word.count_ones()).sum());
    let mut active_density = active_per_iteration.clone();
    active_density.resize(max_iterations as usize, fill);

    CsrPersistentClosureWitness {
        frontier,
        changed,
        converged,
        stop_iteration,
        active_per_iteration,
        active_density,
    }
}

/// Materialize active packed-frontier bits into an ascending queue writing into caller storage.
pub fn frontier_to_queue_witness_into(
    frontier: &[u32],
    node_count: u32,
    queue_capacity: usize,
    queue: &mut Vec<u32>,
) -> u32 {
    let target_cap = queue_capacity.min(node_count as usize);
    if queue.capacity() < target_cap {
        queue.reserve(target_cap.saturating_sub(queue.len()));
    }
    queue.clear();
    let mut active_count = 0_u32;
    for node in 0..node_count {
        let word = (node / 32) as usize;
        let bit = node % 32;
        if frontier
            .get(word)
            .is_some_and(|value| value & (1 << bit) != 0)
        {
            if queue.len() < queue_capacity {
                queue.push(node);
            }
            active_count = active_count.saturating_add(1);
        }
    }
    active_count
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
    let active_count =
        frontier_to_queue_witness_into(frontier, node_count, queue_capacity, &mut queue);
    (queue, active_count)
}

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
    let node_count = node_count as usize;
    let words = (node_count.div_ceil(32)).max(1);
    if output.capacity() < words {
        output.reserve(words.saturating_sub(output.len()));
    }
    output.clear();
    output.extend_from_slice(frontier);
    output.resize(words, 0);
    if edge_offsets.is_empty() {
        return 0;
    }
    assert!(edge_offsets.len() > node_count, "complete CSR offset table");
    let edge_count = edge_offsets[node_count] as usize;
    assert!(
        edge_targets.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "complete CSR edge buffers"
    );
    let mut changed = 0;
    for source in 0..node_count {
        if output[source / 32] & (1 << (source % 32)) == 0 {
            continue;
        }
        let start = edge_offsets[source] as usize;
        let end = (edge_offsets[source + 1] as usize)
            .min(edge_targets.len())
            .min(edge_kind_mask.len());
        if start >= end {
            continue;
        }
        for edge in start..end {
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
        node_count, edge_offsets, edge_targets, edge_kind_mask, frontier, allow_mask, &mut output,
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
    if current.capacity() < words {
        current.reserve(words.saturating_sub(current.len()));
    }
    current.clear();
    current.extend_from_slice(seed);
    if current.len() < words {
        current.resize(words, 0);
    }
    if next.capacity() < words {
        next.reserve(words.saturating_sub(next.len()));
    }
    next.clear();
    next.resize(words, 0);
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
    if frontier.capacity() < words {
        frontier.reserve(words.saturating_sub(frontier.len()));
    }
    frontier.clear();
    frontier.resize(words, 0);
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
    if out.capacity() < words {
        out.reserve(words.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(words, 0_u32);

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
    if current.capacity() < words {
        current.reserve(words.saturating_sub(current.len()));
    }
    current.clear();
    current.extend_from_slice(seed);
    if current.len() < words {
        current.resize(words, 0);
    }

    if next.capacity() < words {
        next.reserve(words.saturating_sub(next.len()));
    }
    next.clear();
    next.resize(words, 0);

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
