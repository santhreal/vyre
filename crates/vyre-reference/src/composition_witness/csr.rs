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
    if row_offsets.is_empty() {
        assert!(
            col_indices.is_empty(),
            "empty CSR offset shorthand requires empty edge buffers"
        );
        let mut distances = vec![u32::MAX; node_count];
        if source < node_count {
            distances[source] = 0;
        }
        return distances;
    }
    assert!(
        row_offsets.len() > node_count,
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

pub(crate) fn validate_csr_inputs(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
) -> (usize, usize) {
    let node_count = node_count as usize;
    if row_offsets.is_empty() {
        assert!(
            col_indices.is_empty() && edge_kind_mask.is_empty(),
            "empty CSR offset shorthand requires empty edge buffers"
        );
        return (node_count, node_count.div_ceil(32));
    }
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
pub(crate) fn prepare_output_buffer(out: &mut Vec<u32>, words: usize) {
    if out.capacity() < words {
        out.reserve(words.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(words, 0);
}

pub(crate) fn prepare_copied_buffer(out: &mut Vec<u32>, words: usize, src: &[u32]) {
    if out.capacity() < words {
        out.reserve(words.saturating_sub(out.len()));
    }
    out.clear();
    out.extend_from_slice(src);
    if out.len() < words {
        out.resize(words, 0);
    }
}
fn accumulate_frontier_step(frontier: &mut [u32], step: &[u32]) -> bool {
    let mut step_changed = false;
    for (word, reached) in frontier.iter_mut().zip(step) {
        let previous = *word;
        *word |= *reached;
        step_changed |= *word != previous;
    }
    step_changed
}

pub(crate) fn for_each_active_edge(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    allow_mask: u32,
    mut f: impl FnMut(usize, usize),
) -> (usize, usize) {
    let (node_count, words) =
        validate_csr_inputs(node_count, row_offsets, col_indices, edge_kind_mask);
    if row_offsets.is_empty() {
        return (node_count, words);
    }
    for source in 0..node_count {
        let start = row_offsets[source] as usize;
        let end = row_offsets[source + 1] as usize;
        for edge in start..end {
            if edge_kind_mask[edge] & allow_mask != 0 {
                let destination = col_indices[edge] as usize;
                if destination < node_count {
                    f(source, destination);
                }
            }
        }
    }
    (node_count, words)
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
    let (_, words) = validate_csr_inputs(node_count, row_offsets, col_indices, edge_kind_mask);
    prepare_output_buffer(out, words);
    for_each_active_edge(
        node_count,
        row_offsets,
        col_indices,
        edge_kind_mask,
        allow_mask,
        |src, dst| {
            if frontier
                .get(src / 32)
                .is_some_and(|w| w & (1 << (src % 32)) != 0)
            {
                out[dst / 32] |= 1 << (dst % 32);
            }
        },
    );
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
    let (_, words) = validate_csr_inputs(node_count, row_offsets, col_indices, edge_kind_mask);
    prepare_output_buffer(out, words);
    for_each_active_edge(
        node_count,
        row_offsets,
        col_indices,
        edge_kind_mask,
        allow_mask,
        |src, dst| {
            if frontier
                .get(dst / 32)
                .is_some_and(|w| w & (1 << (dst % 32)) != 0)
            {
                out[src / 32] |= 1 << (src % 32);
            }
        },
    );
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
        .expect("Fix: reserve at least seed.len() capacity in current and next ping-pong buffers")
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
    let mut next = current.clone();
    for _ in 0..max_iters {
        let mut changed = false;
        for_each_active_edge(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            allow_mask,
            |src, dst| {
                if current[src / 32] & (1 << (src % 32)) != 0 {
                    let old = next[dst / 32];
                    next[dst / 32] |= 1 << (dst % 32);
                    if next[dst / 32] != old {
                        changed = true;
                    }
                }
            },
        );
        if !changed {
            break;
        }
        step_hook(&next);
        current = next.clone();
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
    prepare_copied_buffer(frontier_out, words, frontier_in);

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
        let step_changed = accumulate_frontier_step(frontier_out, step_scratch);
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
            let step_changed = accumulate_frontier_step(&mut frontier, &step);
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

pub use super::csr_step::*;
