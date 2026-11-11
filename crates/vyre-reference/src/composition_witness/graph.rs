//! Sequential mathematical witnesses for graph analysis, dominator trees, homology, and matroids.

/// Sequential mathematical witness for first Betti number calculation on a 1-skeleton graph: `(b0, b1, edges)`.
///
/// Shape: `mask` is row-major `n × n`, symmetric, self-edges ignored.
/// Returns `(b0, b1, edges)`.
#[must_use]
pub fn betti_persistence_witness(mask: &[u32], n: u32) -> (u32, u32, u32) {
    if n == 0 {
        return (0, 0, 0);
    }
    let n_us = n as usize;
    let Some(cells) = n_us.checked_mul(n_us) else {
        return (0, 0, 0);
    };
    if mask.len() < cells {
        return (0, 0, 0);
    }
    let mut parent: Vec<u32> = (0..n).collect();
    let mut rank: Vec<u32> = vec![0; n_us];

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = parent[x as usize];
        }
        x
    }

    fn union(parent: &mut [u32], rank: &mut [u32], a: u32, b: u32) -> bool {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return false;
        }
        let (ra_rank, rb_rank) = (rank[ra as usize], rank[rb as usize]);
        match ra_rank.cmp(&rb_rank) {
            std::cmp::Ordering::Less => parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                parent[rb as usize] = ra;
                rank[ra as usize] = ra_rank + 1;
            }
        }
        true
    }

    let mut edges: u32 = 0;
    let mut tree_edges: u32 = 0;

    for i in 0..n_us {
        for j in (i + 1)..n_us {
            let upper = mask[i * n_us + j];
            let lower = mask[j * n_us + i];
            if upper != lower || upper == 0 {
                continue;
            }
            edges = edges.saturating_add(1);
            if union(&mut parent, &mut rank, i as u32, j as u32) {
                tree_edges = tree_edges.saturating_add(1);
            }
        }
    }

    let mut b0 = 0u32;
    for v in 0..n {
        if find(&mut parent, v) == v {
            b0 = b0.saturating_add(1);
        }
    }
    let b1 = edges.saturating_sub(tree_edges);
    (b0, b1, edges)
}

/// Publish motif endpoints only when every requested typed edge exists into caller-owned storage.
pub fn motif_witness_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    motif_edges: &[(u32, u32, u32)],
    output: &mut Vec<u32>,
) {
    let node_count_usize = node_count as usize;
    if output.capacity() < node_count_usize {
        output.reserve(node_count_usize.saturating_sub(output.len()));
    }
    output.clear();
    output.resize(node_count_usize, 0);
    for &(source, required_mask, destination) in motif_edges {
        if source >= node_count || destination >= node_count {
            return;
        }
        let start = edge_offsets.get(source as usize).copied().unwrap_or(0) as usize;
        let end = edge_offsets
            .get(source as usize + 1)
            .copied()
            .unwrap_or(start as u32) as usize;
        let present = (start..end).any(|edge| {
            edge_targets.get(edge) == Some(&destination)
                && edge_kind_masks
                    .get(edge)
                    .is_some_and(|kind| kind & required_mask != 0)
        });
        if !present {
            return;
        }
    }
    for &(source, _, destination) in motif_edges {
        output[source as usize] = 1;
        output[destination as usize] = 1;
    }
}

/// Publish motif endpoints only when every requested typed edge exists.
#[must_use]
pub fn motif_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    motif_edges: &[(u32, u32, u32)],
) -> Vec<u32> {
    let mut output = Vec::with_capacity(node_count as usize);
    motif_witness_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_masks,
        motif_edges,
        &mut output,
    );
    output
}

/// Split queued CSR rows into scalar work and a compact high-degree queue.
///
/// The returned tuple is `(frontier_out, high_queue, observed_high_count)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_split_low_forward_witness(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_masks: &[u32],
    frontier_seed: &[u32],
    node_count: u32,
    high_queue_capacity: usize,
    high_degree_threshold: u32,
    allow_mask: u32,
) -> (Vec<u32>, Vec<u32>, u32) {
    let active_len = (queue_len as usize).min(active_queue.len());
    let mut frontier_out = frontier_seed.to_vec();
    let num_words = (node_count as usize).div_ceil(32);
    if frontier_out.len() < num_words {
        frontier_out.resize(num_words, 0);
    }
    let mut high_queue = Vec::with_capacity(high_queue_capacity.min(active_len));
    let mut high_count = 0_u32;

    for &source in &active_queue[..active_len] {
        if source >= node_count {
            continue;
        }
        let start = edge_offsets.get(source as usize).copied().unwrap_or(0) as usize;
        let end = edge_offsets
            .get(source as usize + 1)
            .copied()
            .unwrap_or(start as u32) as usize;
        let end = end.min(edge_targets.len()).min(edge_kind_masks.len());
        if end.saturating_sub(start) as u32 >= high_degree_threshold {
            high_count = high_count.saturating_add(1);
            if high_queue.len() < high_queue_capacity {
                high_queue.push(source);
                continue;
            }
        }
        if start < end {
            for edge in start..end {
                if edge_kind_masks[edge] & allow_mask == 0 {
                    continue;
                }
                let destination = edge_targets[edge];
                if destination < node_count {
                    frontier_out[destination as usize / 32] |= 1_u32 << (destination % 32);
                }
            }
        }
    }

    (frontier_out, high_queue, high_count)
}

/// Scratch storage for exploded IFDS CSR construction.
#[must_use]
pub fn scc_decompose_witness(
    node_count: u32,
    forward: &[u32],
    backward: &[u32],
    components: &[u32],
    pivot: u32,
) -> Vec<u32> {
    let mut output = vec![u32::MAX; node_count as usize];
    for (destination, &component) in output.iter_mut().zip(components) {
        *destination = component;
    }
    for node in 0..node_count as usize {
        let bit = 1_u32 << (node % 32);
        let word = node / 32;
        if forward.get(word).is_some_and(|value| value & bit != 0)
            && backward.get(word).is_some_and(|value| value & bit != 0)
            && output[node] == u32::MAX
        {
            output[node] = pivot;
        }
    }
    output
}

fn fill_path_witness(parents: &[u32], target: u32, path: &mut [u32]) -> u32 {
    path.fill(0);
    let mut current = target;
    let mut length = 0_usize;
    while length < path.len() {
        path[length] = current;
        length += 1;
        let parent = parents.get(current as usize).copied().unwrap_or(current);
        if parent == current {
            break;
        }
        current = parent;
    }
    length as u32
}

/// Follow parent pointers from a target into a zero-padded bounded path in caller-owned storage.
pub fn path_reconstruct_witness_into(
    parents: &[u32],
    target: u32,
    max_depth: u32,
    path: &mut Vec<u32>,
) -> u32 {
    let max_depth = max_depth as usize;
    if path.capacity() < max_depth {
        path.reserve(max_depth.saturating_sub(path.len()));
    }
    path.clear();
    path.resize(max_depth, 0);
    fill_path_witness(parents, target, path)
}

/// Follow parent pointers from a target into a zero-padded bounded path.
#[must_use]
pub fn path_reconstruct_witness(parents: &[u32], target: u32, max_depth: u32) -> (Vec<u32>, u32) {
    let mut path = Vec::with_capacity(max_depth as usize);
    let length = path_reconstruct_witness_into(parents, target, max_depth, &mut path);
    (path, length)
}

/// Follow parent pointers for many targets into caller-owned, zero-padded storage.
///
/// # Errors
///
/// Returns an error before mutating either output when the packed output length
/// exceeds addressable storage.
pub fn try_path_reconstruct_batch_witness_into(
    parents: &[u32],
    targets: &[u32],
    max_depth: u32,
    paths: &mut Vec<u32>,
    lengths: &mut Vec<u32>,
) -> Result<(), String> {
    let max_depth = max_depth as usize;
    let output_len = targets.len().checked_mul(max_depth).ok_or_else(|| {
        "Fix: batched_path_reconstruct output length exceeds addressable storage.".to_string()
    })?;
    if paths.capacity() < output_len {
        paths.reserve(output_len.saturating_sub(paths.len()));
    }
    if lengths.capacity() < targets.len() {
        lengths.reserve(targets.len().saturating_sub(lengths.len()));
    }
    paths.clear();
    paths.resize(output_len, 0);
    lengths.clear();
    if max_depth == 0 {
        lengths.resize(targets.len(), 0);
        return Ok(());
    }
    for (&target, path) in targets.iter().zip(paths.chunks_exact_mut(max_depth)) {
        lengths.push(fill_path_witness(parents, target, path));
    }
    Ok(())
}

/// Sequential union-by-min over an initialized parent forest.
#[must_use]
pub fn union_find_alias_witness(parent_init: &[u32], edge_a: &[u32], edge_b: &[u32]) -> Vec<u32> {
    let mut roots = canonicalize_union_find_witness(parent_init);
    for (&left, &right) in edge_a.iter().zip(edge_b) {
        let (left, right) = (left as usize, right as usize);
        if left >= roots.len() || right >= roots.len() {
            continue;
        }
        let low = roots[left].min(roots[right]);
        let high = roots[left].max(roots[right]);
        for root in &mut roots {
            if *root == high {
                *root = low;
            }
        }
    }
    roots
}

/// Resolve each parent pointer to a stable root with cycle and bounds guards.
#[must_use]
pub fn canonicalize_union_find_witness(parents: &[u32]) -> Vec<u32> {
    (0..parents.len())
        .map(|node| {
            let mut current = node as u32;
            for _ in 0..parents.len() {
                let Some(&next) = parents.get(current as usize) else {
                    break;
                };
                if next == current {
                    break;
                }
                current = next;
            }
            current
        })
        .collect()
}

/// Apply a column mapping to one row, with later sources winning duplicates.
#[must_use]
pub fn functor_apply_witness(source_row: &[u32], mapping: &[u32], target_size: u32) -> Vec<u32> {
    let mut output = Vec::new();
    functor_apply_witness_into(source_row, mapping, target_size, &mut output);
    output
}

/// Apply a column mapping to one row into caller-owned storage, with later sources winning duplicates.
pub fn functor_apply_witness_into(
    source_row: &[u32],
    mapping: &[u32],
    target_size: u32,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.resize(target_size as usize, 0);
    for (&value, &target) in source_row.iter().zip(mapping) {
        if let Some(destination) = out.get_mut(target as usize) {
            *destination = value;
        }
    }
}

/// Compute packed forward and backward reachability from one dense-graph pivot.
#[must_use]
pub fn dense_reachability_bitsets_witness(
    adjacency: &[u32],
    pivot: u32,
    node_count: u32,
) -> (Vec<u32>, Vec<u32>) {
    let node_count = node_count as usize;
    assert!(node_count > 0, "node_count must be nonzero");
    assert!((pivot as usize) < node_count, "pivot must be in range");
    assert_eq!(
        adjacency.len(),
        node_count * node_count,
        "dense adjacency must contain node_count squared entries"
    );

    let traverse = |reverse: bool| {
        let mut reached = vec![false; node_count];
        let mut stack = vec![pivot as usize];
        reached[pivot as usize] = true;
        while let Some(source) = stack.pop() {
            for destination in 0..node_count {
                let edge = if reverse {
                    adjacency[destination * node_count + source]
                } else {
                    adjacency[source * node_count + destination]
                };
                if edge != 0 && !reached[destination] {
                    reached[destination] = true;
                    stack.push(destination);
                }
            }
        }
        let mut words = vec![0_u32; node_count.div_ceil(32)];
        for (node, present) in reached.into_iter().enumerate() {
            if present {
                words[node / 32] |= 1 << (node % 32);
            }
        }
        words
    };

    (traverse(false), traverse(true))
}

/// Label strongly connected components of a dense adjacency matrix.
#[must_use]
pub fn dense_scc_components_witness(adjacency: &[u32], node_count: u32) -> Vec<u32> {
    if node_count == 0 {
        assert!(
            adjacency.is_empty(),
            "empty graph must have empty adjacency"
        );
        return Vec::new();
    }
    let mut components = vec![u32::MAX; node_count as usize];
    for pivot in 0..node_count {
        if components[pivot as usize] != u32::MAX {
            continue;
        }
        let (forward, backward) = dense_reachability_bitsets_witness(adjacency, pivot, node_count);
        for node in 0..node_count as usize {
            let bit = 1 << (node % 32);
            if forward[node / 32] & backward[node / 32] & bit != 0 {
                components[node] = pivot;
            }
        }
    }
    components
}

/// Return whether an adjustment candidate intersects descendants of treatment.
#[must_use]
pub fn backdoor_descendants_check_witness(
    candidate_adjustment: &[u32],
    treatment_descendants: &[u32],
) -> bool {
    candidate_adjustment
        .iter()
        .zip(treatment_descendants)
        .any(|(&candidate, &descendant)| candidate != 0 && descendant != 0)
}

/// Sequential mathematical witness for level-wave iteration: visits lanes depth-by-depth in `0..max_depth`.
pub fn level_wave_witness<F>(depths: &[u32], max_depth: u32, mut step_for_lane: F)
where
    F: FnMut(u32, u32),
{
    for current_depth in 0..max_depth {
        for (lane_idx, &lane_depth) in depths.iter().enumerate() {
            if lane_depth == current_depth {
                step_for_lane(lane_idx as u32, current_depth);
            }
        }
    }
}

/// Sequential mathematical witness for Kahn's topological sort over `(from, to)` edges.
///
/// `from` depends on `to`, so `to` comes first in the sort.
pub fn toposort_witness(node_count: u32, edges: &[(u32, u32)]) -> Result<Vec<u32>, String> {
    const NONE: usize = usize::MAX;
    let n = node_count as usize;
    for (edge_idx, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(format!("Unknown node {from} at edge {edge_idx}"));
        }
        if to >= node_count {
            return Err(format!("Unknown node {to} at edge {edge_idx}"));
        }
    }
    let mut indeg = vec![0u32; n];
    let mut outgoing_head = vec![NONE; n];
    let mut outgoing_to = Vec::with_capacity(edges.len());
    let mut outgoing_next = Vec::with_capacity(edges.len());
    let mut depends_head = vec![NONE; n];
    let mut depends_to = Vec::with_capacity(edges.len());
    let mut depends_next = Vec::with_capacity(edges.len());

    for &(from, to) in edges {
        let outgoing_idx = outgoing_to.len();
        outgoing_to.push(from);
        outgoing_next.push(outgoing_head[to as usize]);
        outgoing_head[to as usize] = outgoing_idx;

        let depends_idx = depends_to.len();
        depends_to.push(to);
        depends_next.push(depends_head[from as usize]);
        depends_head[from as usize] = depends_idx;

        let slot = &mut indeg[from as usize];
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| format!("indegree overflow for node {from}"))?;
    }

    let mut queue = Vec::with_capacity(n);
    for v in 0..node_count {
        if indeg[v as usize] == 0 {
            queue.push(v);
        }
    }
    let mut out = Vec::with_capacity(n);
    while let Some(v) = queue.pop() {
        out.push(v);
        let mut edge = outgoing_head[v as usize];
        while edge != NONE {
            let u = outgoing_to[edge];
            let slot = &mut indeg[u as usize];
            *slot = slot
                .checked_sub(1)
                .ok_or_else(|| format!("indegree underflow for node {u}"))?;
            if *slot == 0 {
                queue.push(u);
            }
            edge = outgoing_next[edge];
        }
    }

    if out.len() != n {
        let seed = indeg
            .iter()
            .enumerate()
            .find(|(_, deg)| **deg > 0)
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "toposort could not find positive indegree seed".to_string())?;
        let mut on_stack = vec![false; n];
        let mut cursor = seed;
        let cycle_node = loop {
            if on_stack[cursor as usize] {
                break cursor;
            }
            on_stack[cursor as usize] = true;
            let mut edge = depends_head[cursor as usize];
            let mut next = None;
            while edge != NONE {
                let candidate = depends_to[edge];
                if indeg[candidate as usize] > 0 {
                    next = Some(candidate);
                    break;
                }
                edge = depends_next[edge];
            }
            match next {
                Some(n) => cursor = n,
                None => return Err(format!("cycle diagnosis stuck at node {cursor}")),
            }
        };
        return Err(format!("Cycle detected involving node {cycle_node}"));
    }
    Ok(out)
}

/// Sequential mathematical witness for Kahn's topological sort over CSR graph representation into caller storage.
pub fn toposort_csr_into_witness(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &mut Vec<u32>,
) -> Result<(), String> {
    let mut indegree = Vec::new();
    let mut queue = Vec::new();
    toposort_csr_with_scratch_into_witness(
        node_count,
        offsets,
        targets,
        order,
        &mut indegree,
        &mut queue,
    )
}

/// Sequential CSR topological sort with caller-owned output and work storage.
///
/// Input validation completes before any caller-owned vector is changed.
pub fn toposort_csr_with_scratch_into_witness(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &mut Vec<u32>,
    indegree: &mut Vec<u32>,
    queue: &mut Vec<u32>,
) -> Result<(), String> {
    if node_count == 0 {
        if offsets != [0] || !targets.is_empty() {
            return Err(
                "toposort_csr zero-node graph requires offsets == [0] and empty targets"
                    .to_string(),
            );
        }
        order.clear();
        indegree.clear();
        queue.clear();
        return Ok(());
    }
    let expected_offsets = (node_count as usize) + 1;
    if offsets.len() != expected_offsets {
        return Err(format!(
            "offsets.len() == {expected_offsets} expected, got {}",
            offsets.len()
        ));
    }
    if offsets[0] != 0 {
        return Err("offsets[0] must be 0".to_string());
    }
    for index in 0..node_count as usize {
        if offsets[index] > offsets[index + 1] {
            return Err(format!("offsets not monotonic at index {index}"));
        }
    }
    if offsets[node_count as usize] as usize != targets.len() {
        return Err("last offset does not match targets.len()".to_string());
    }
    for &target in targets {
        if target >= node_count {
            return Err(format!(
                "target node {target} out of range for node_count {node_count}"
            ));
        }
    }

    let node_count = node_count as usize;
    order.clear();
    indegree.clear();
    indegree.resize(node_count, 0);
    queue.clear();
    queue.reserve(node_count);
    for &target in targets {
        indegree[target as usize] = indegree[target as usize]
            .checked_add(1)
            .ok_or_else(|| format!("indegree overflow for target {target}"))?;
    }
    for node in 0..node_count as u32 {
        if indegree[node as usize] == 0 {
            queue.push(node);
        }
    }
    while let Some(node) = queue.pop() {
        order.push(node);
        let start = offsets[node as usize] as usize;
        let end = offsets[node as usize + 1] as usize;
        for &dependent in &targets[start..end] {
            let slot = &mut indegree[dependent as usize];
            *slot = slot
                .checked_sub(1)
                .ok_or_else(|| format!("indegree underflow for dependent {dependent}"))?;
            if *slot == 0 {
                queue.push(dependent);
            }
        }
    }
    if order.len() != node_count {
        return Err(format!(
            "toposort_csr cycle detected: produced {} nodes out of {node_count}",
            order.len()
        ));
    }
    Ok(())
}

/// Sequential mathematical witness for Kahn's topological sort over CSR graph representation returning vector.
#[must_use]
pub fn toposort_csr_witness(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<Vec<u32>, String> {
    let mut order = Vec::new();
    toposort_csr_into_witness(node_count, offsets, targets, &mut order)?;
    Ok(order)
}

/// Sequential mathematical witness for transitive reachability over directed `(from, to)` edges.
pub fn reachable_witness(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
) -> Result<std::collections::HashSet<u32>, String> {
    const NONE: usize = usize::MAX;
    let n = node_count as usize;
    for (index, &(from, to)) in edges.iter().enumerate() {
        if (from as usize) >= n {
            return Err(format!("Unknown node {from} at edge {index}"));
        }
        if (to as usize) >= n {
            return Err(format!("Unknown node {to} at edge {index}"));
        }
    }
    let mut head = vec![NONE; n];
    let mut to_nodes = Vec::with_capacity(edges.len());
    let mut next_edges = Vec::with_capacity(edges.len());
    for &(from, to) in edges {
        let edge_index = to_nodes.len();
        to_nodes.push(to);
        next_edges.push(head[from as usize]);
        head[from as usize] = edge_index;
    }
    let mut visited = vec![false; n];
    let mut out_of_range_sources = Vec::new();
    let mut stack = sources.to_vec();
    while let Some(v) = stack.pop() {
        let idx = v as usize;
        if idx >= n {
            out_of_range_sources.push(v);
            continue;
        }
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        let mut edge = head[idx];
        while edge != NONE {
            let next = to_nodes[edge];
            if !visited[next as usize] {
                stack.push(next);
            }
            edge = next_edges[edge];
        }
    }
    let mut result = std::collections::HashSet::new();
    for (idx, is_visited) in visited.into_iter().enumerate() {
        if is_visited {
            result.insert(idx as u32);
        }
    }
    result.extend(out_of_range_sources);
    Ok(result)
}
pub use super::graph_dataflow::*;
pub use super::graph_dominator::*;
pub use super::graph_matroid::*;
pub use super::graph_sheaf::*;
pub use super::graph_vector::*;
