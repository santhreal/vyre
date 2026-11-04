//! Sequential mathematical witnesses for tensor flow propagation and vector KNN/graph traversal.

/// Helper: compute tensor bit index in the flat bitset.
#[must_use]
pub const fn tensor_bit_index_witness(
    node: u32,
    ctx: u32,
    fld: u32,
    context_limit: u32,
    field_limit: u32,
) -> u32 {
    node * context_limit * field_limit + ctx * field_limit + fld
}

/// Sequential mathematical witness for 3D tensor flow propagation writing into caller storage.
#[allow(clippy::too_many_arguments)]
pub fn try_tensor_flow_forward_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    tensor_in: &[u32],
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!("Fix: node_count + 1 overflows usize for node_count={node_count}.")
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: edge_offsets.len() must equal node_count + 1, got len={}, node_count={node_count}.",
            edge_offsets.len()
        ));
    }
    if edge_offsets[0] != 0 {
        return Err(format!(
            "Fix: edge_offsets[0] must be 0, got {}.",
            edge_offsets[0]
        ));
    }
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: non-monotonic CSR offsets at row {index}: {} > {}.",
                pair[0], pair[1]
            ));
        }
    }
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: edge_targets.len() must equal edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    if edge_targets.len() != edge_count {
        return Err(format!(
            "Fix: final offset declares edge_count={edge_count}, but edge_targets.len() is {}.",
            edge_targets.len()
        ));
    }
    let lanes_per_node = context_limit
        .checked_mul(field_limit)
        .ok_or_else(|| "context_limit * field_limit overflowed u32".to_string())?;
    let total_bits = (node_count as u64)
        .checked_mul(lanes_per_node as u64)
        .ok_or_else(|| "node_count * lanes_per_node overflowed u64".to_string())?;
    let words = usize::try_from(total_bits.div_ceil(32))
        .map_err(|_| "tensor words count exceeds usize limit".to_string())?;
    if tensor_in.len() < words {
        return Err("tensor input buffer shorter than required tensor words".to_owned());
    }
    out.try_reserve(words.saturating_sub(out.len()))
        .map_err(|error| format!("failed to reserve tensor output buffer: {error}"))?;
    out.clear();
    out.resize(words, 0);

    for src in 0..node_count as usize {
        let (start, end) = (edge_offsets[src] as usize, edge_offsets[src + 1] as usize);
        for edge in start..end {
            let dst = edge_targets[edge];
            if (edge_kind_mask[edge] & allow_mask) != 0 && dst < node_count {
                for ctx in 0..context_limit {
                    for fld in 0..field_limit {
                        let s_bit = tensor_bit_index_witness(
                            src as u32,
                            ctx,
                            fld,
                            context_limit,
                            field_limit,
                        );
                        if (tensor_in[(s_bit / 32) as usize] & (1 << (s_bit % 32))) != 0 {
                            let d_bit =
                                tensor_bit_index_witness(dst, ctx, fld, context_limit, field_limit);
                            out[(d_bit / 32) as usize] |= 1 << (d_bit % 32);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Sequential mathematical witness for 3D tensor flow propagation.
#[allow(clippy::too_many_arguments)]
pub fn try_tensor_flow_forward_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    tensor_in: &[u32],
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    try_tensor_flow_forward_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        tensor_in,
        context_limit,
        field_limit,
        allow_mask,
        &mut out,
    )?;
    Ok(out)
}

/// Sequential mathematical witness for 3D tensor flow propagation (panicking on invalid shapes).
///
/// # Panics
///
/// Panics if graph buffers, masks, or tensor dimensions are invalid or inconsistent.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn tensor_flow_forward_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    tensor_in: &[u32],
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
) -> Vec<u32> {
    try_tensor_flow_forward_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        tensor_in,
        context_limit,
        field_limit,
        allow_mask,
    )
    .unwrap_or_else(|err| panic!("tensor_flow_forward_witness failed: {err}"))
}

/// Helper: compute squared Euclidean (L2) distance between two vector slices.
#[must_use]
pub fn vector_squared_l2_witness(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(l, r)| {
            let d = *l - *r;
            d * d
        })
        .sum()
}

/// Sequential mathematical witness for deterministic K-nearest-neighbor (KNN) CSR graph construction.
#[must_use]
pub fn knn_csr_witness(
    vectors: &[f32],
    dimension: usize,
    neighbor_k: usize,
) -> (Vec<u32>, Vec<u32>) {
    if dimension == 0 || vectors.len() % dimension != 0 {
        return (vec![0], Vec::new());
    }
    let node_count = vectors.len() / dimension;
    let mut offsets = Vec::with_capacity(node_count + 1);
    let mut targets = Vec::with_capacity(node_count * neighbor_k);
    offsets.push(0);
    for src in 0..node_count {
        let src_row = &vectors[src * dimension..(src + 1) * dimension];
        let mut candidates: Vec<(f32, usize)> = (0..node_count)
            .filter(|dst| *dst != src)
            .map(|dst| {
                let dst_row = &vectors[dst * dimension..(dst + 1) * dimension];
                (vector_squared_l2_witness(src_row, dst_row), dst)
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        targets.extend(
            candidates
                .into_iter()
                .take(neighbor_k)
                .map(|(_, dst)| dst as u32),
        );
        offsets.push(targets.len() as u32);
    }
    (offsets, targets)
}

/// Sequential mathematical witness for Top-K ranking of nodes by squared L2 distance to a query vector.
#[must_use]
pub fn vector_top_k_witness<I>(
    vectors: &[f32],
    dimension: usize,
    query: &[f32],
    nodes: I,
    rank_k: usize,
) -> Vec<(usize, f32)>
where
    I: IntoIterator<Item = usize>,
{
    if dimension == 0 || query.len() != dimension {
        return Vec::new();
    }
    let mut scored: Vec<(f32, usize)> = nodes
        .into_iter()
        .filter_map(|node| {
            let start = node.checked_mul(dimension)?;
            let end = start.checked_add(dimension)?;
            if end <= vectors.len() {
                let node_row = &vectors[start..end];
                Some((vector_squared_l2_witness(node_row, query), node))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    scored
        .into_iter()
        .take(rank_k)
        .map(|(dist, node)| (node, dist))
        .collect()
}

/// Sequential mathematical witness for seed reachability traversal over CSR graph.
#[must_use]
pub fn vector_graph_traverse_from_seed_witness(
    seed: usize,
    node_count: usize,
    csr_offsets: &[u32],
    csr_targets: &[u32],
) -> Vec<bool> {
    let mut reached = vec![false; node_count];
    if seed >= node_count || csr_offsets.len() != node_count + 1 {
        return reached;
    }
    let mut queue = std::collections::VecDeque::new();
    reached[seed] = true;
    queue.push_back(seed);
    while let Some(node) = queue.pop_front() {
        let start = csr_offsets[node] as usize;
        let end = csr_offsets[node + 1] as usize;
        if start <= end && end <= csr_targets.len() {
            for &target in &csr_targets[start..end] {
                let target = target as usize;
                if target < node_count && !reached[target] {
                    reached[target] = true;
                    queue.push_back(target);
                }
            }
        }
    }
    reached
}
