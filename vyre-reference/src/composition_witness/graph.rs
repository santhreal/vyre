//! Sequential mathematical witnesses for graph analysis, dominator trees, homology, and matroids.

/// Sequential mathematical witness for immediate dominators (Cooper-Harvey-Kennedy).
/// Returns an array `idom` where `idom[u]` is the immediate dominator of `u`, or `u32::MAX` if unreachable.
#[must_use]
pub fn dominator_tree_witness(node_count: u32, root: u32, edges: &[(u32, u32)]) -> Vec<u32> {
    if node_count == 0 {
        return Vec::new();
    }
    let n = node_count as usize;
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if (u as usize) < n && (v as usize) < n {
            preds[v as usize].push(u);
            succs[u as usize].push(v);
        }
    }
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    fn dfs(u: usize, succs: &[Vec<u32>], visited: &mut [bool], postorder: &mut Vec<usize>) {
        visited[u] = true;
        for &v in &succs[u] {
            let v = v as usize;
            if !visited[v] {
                dfs(v, succs, visited, postorder);
            }
        }
        postorder.push(u);
    }
    if (root as usize) < n {
        dfs(root as usize, &succs, &mut visited, &mut postorder);
    }
    let mut postorder_num = vec![usize::MAX; n];
    for (i, &u) in postorder.iter().enumerate() {
        postorder_num[u] = i;
    }
    let mut idom = vec![u32::MAX; n];
    if (root as usize) < n {
        idom[root as usize] = root;
    }
    let intersect =
        |mut b1: usize, mut b2: usize, idom: &[u32], postorder_num: &[usize]| -> usize {
            while b1 != b2 {
                while postorder_num[b1] < postorder_num[b2] {
                    b1 = idom[b1] as usize;
                }
                while postorder_num[b2] < postorder_num[b1] {
                    b2 = idom[b2] as usize;
                }
            }
            b1
        };
    let mut changed = true;
    while changed {
        changed = false;
        for &u in postorder.iter().rev() {
            if u == root as usize {
                continue;
            }
            let mut new_idom: Option<usize> = None;
            for &p in &preds[u] {
                let p = p as usize;
                if idom[p] != u32::MAX {
                    if let Some(curr) = new_idom {
                        new_idom = Some(intersect(p, curr, &idom, &postorder_num));
                    } else {
                        new_idom = Some(p);
                    }
                }
            }
            if let Some(new_idom_val) = new_idom {
                let new_idom_u32 = new_idom_val as u32;
                if idom[u] != new_idom_u32 {
                    idom[u] = new_idom_u32;
                    changed = true;
                }
            }
        }
    }
    idom
}

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

/// One deterministic level-synchronous Edmonds augmentation witness into caller-owned storage.
pub fn matroid_intersection_augmentation_witness_into(
    exchange_adjacency: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
    output: &mut Vec<u32>,
    parent: &mut Vec<u32>,
    visited: &mut Vec<u32>,
    frontier: &mut Vec<u32>,
    next_frontier: &mut Vec<u32>,
) {
    assert_eq!(exchange_adjacency.len(), n * n);
    assert_eq!(sources.len(), n);
    assert_eq!(sinks.len(), n);
    assert_eq!(set_x.len(), n);
    if output.capacity() < n {
        output.reserve(n.saturating_sub(output.len()));
    }
    output.clear();
    output.extend_from_slice(set_x);

    if frontier.capacity() < n {
        frontier.reserve(n.saturating_sub(frontier.len()));
    }
    frontier.clear();
    frontier.extend(sources.iter().map(|&value| u32::from(value != 0)));

    if visited.capacity() < n {
        visited.reserve(n.saturating_sub(visited.len()));
    }
    visited.clear();
    visited.extend_from_slice(frontier);

    if parent.capacity() < n {
        parent.reserve(n.saturating_sub(parent.len()));
    }
    parent.clear();
    parent.resize(n, u32::MAX);

    let mut target = (0..n).find(|&node| frontier[node] != 0 && sinks[node] != 0);
    while target.is_none() && frontier.iter().any(|&value| value != 0) {
        if next_frontier.capacity() < n {
            next_frontier.reserve(n.saturating_sub(next_frontier.len()));
        }
        next_frontier.clear();
        next_frontier.resize(n, 0);
        for destination in 0..n {
            if visited[destination] != 0 {
                continue;
            }
            if let Some(source) = (0..n).find(|&source| {
                frontier[source] != 0 && exchange_adjacency[source * n + destination] != 0
            }) {
                parent[destination] = source as u32;
                next_frontier[destination] = 1;
                visited[destination] = 1;
            }
        }
        target = (0..n).find(|&node| next_frontier[node] != 0 && sinks[node] != 0);
        std::mem::swap(frontier, next_frontier);
    }
    if let Some(mut node) = target {
        loop {
            output[node] = 1_u32.wrapping_sub(output[node]);
            let previous = parent[node];
            if previous == u32::MAX {
                break;
            }
            node = previous as usize;
        }
    }
}

/// One deterministic level-synchronous Edmonds augmentation witness.
#[must_use]
pub fn matroid_intersection_augmentation_witness(
    exchange_adjacency: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
) -> Vec<u32> {
    let mut output = Vec::with_capacity(n);
    let mut parent = Vec::with_capacity(n);
    let mut visited = Vec::with_capacity(n);
    let mut frontier = Vec::with_capacity(n);
    let mut next_frontier = Vec::with_capacity(n);
    matroid_intersection_augmentation_witness_into(
        exchange_adjacency,
        sources,
        sinks,
        set_x,
        n,
        &mut output,
        &mut parent,
        &mut visited,
        &mut frontier,
        &mut next_frontier,
    );
    output
}

/// Sequential bounded Edmonds selector with repeated-state termination.
pub fn matroid_select_optimal_subset_witness(
    exchange_adjacency: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed: &[u32],
    n: usize,
    max_augmentations: u32,
) -> Result<Vec<u32>, String> {
    let adjacency_len = n
        .checked_mul(n)
        .ok_or_else(|| format!("exact matroid solver n*n overflow for n={n}"))?;
    if exchange_adjacency.len() != adjacency_len {
        return Err(format!(
            "exact matroid solver exchange_adj length {} does not match n*n={adjacency_len}",
            exchange_adjacency.len()
        ));
    }
    for (label, actual) in [
        ("sources", sources.len()),
        ("sinks", sinks.len()),
        ("seed", seed.len()),
    ] {
        if actual != n {
            return Err(format!(
                "exact matroid solver {label} length {actual} does not match n={n}"
            ));
        }
    }

    let mut current = seed.to_vec();
    let mut seen = vec![current.clone()];
    for _ in 0..max_augmentations {
        let next = matroid_intersection_augmentation_witness(
            exchange_adjacency,
            sources,
            sinks,
            &current,
            n,
        );
        if next == current {
            break;
        }
        if seen.iter().any(|state| state == &next) {
            if next.iter().filter(|&&value| value != 0).count()
                > current.iter().filter(|&&value| value != 0).count()
            {
                current = next;
            }
            break;
        }
        seen.push(next.clone());
        current = next;
    }
    Ok(current)
}

/// Sequential bounded Edmonds selector writing into caller-owned state buffers.
pub fn matroid_select_optimal_subset_witness_into(
    exchange_adjacency: &[u32],
    sources: &[u32],
    sinks: &[u32],
    seed: &[u32],
    n: usize,
    max_augmentations: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), String> {
    let result = matroid_select_optimal_subset_witness(
        exchange_adjacency,
        sources,
        sinks,
        seed,
        n,
        max_augmentations,
    )?;
    current.clear();
    current.extend_from_slice(&result);
    next.clear();
    next.resize(current.len(), 0);
    Ok(())
}

/// Iterative Cooper-Harvey-Kennedy immediate-dominator witness.
#[must_use]
pub fn dominator_idoms_witness(
    node_count: u32,
    root: u32,
    edges: &[(u32, u32)],
) -> Vec<Option<u32>> {
    let n = node_count as usize;
    if n == 0 {
        return Vec::new();
    }
    if (root as usize) >= n {
        return vec![None; n];
    }

    fn compact_adjacency(n: usize, edges: &[(u32, u32)], reverse: bool) -> (Vec<u32>, Vec<u32>) {
        let mut offsets = vec![0_u32; n + 1];
        for &(source, destination) in edges {
            let (from, to) = if reverse {
                (destination, source)
            } else {
                (source, destination)
            };
            if (from as usize) < n && (to as usize) < n {
                offsets[from as usize + 1] += 1;
            }
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursor = offsets[..n].to_vec();
        let mut targets = vec![0_u32; offsets[n] as usize];
        for &(source, destination) in edges {
            let (from, to) = if reverse {
                (destination, source)
            } else {
                (source, destination)
            };
            if (from as usize) < n && (to as usize) < n {
                let slot = cursor[from as usize] as usize;
                targets[slot] = to;
                cursor[from as usize] += 1;
            }
        }
        (offsets, targets)
    }

    let (successor_offsets, successors) = compact_adjacency(n, edges, false);
    let (predecessor_offsets, predecessors) = compact_adjacency(n, edges, true);
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack = vec![(root, successor_offsets[root as usize])];
    visited[root as usize] = true;
    while let Some((node, next_edge)) = stack.pop() {
        let end = successor_offsets[node as usize + 1];
        if next_edge < end {
            stack.push((node, next_edge + 1));
            let successor = successors[next_edge as usize];
            if !visited[successor as usize] {
                visited[successor as usize] = true;
                stack.push((successor, successor_offsets[successor as usize]));
            }
        } else {
            postorder.push(node);
        }
    }
    postorder.reverse();
    let mut order = vec![u32::MAX; n];
    for (index, &node) in postorder.iter().enumerate() {
        order[node as usize] = index as u32;
    }
    let mut idom = vec![u32::MAX; n];
    idom[root as usize] = root;
    let intersect = |mut left: u32, mut right: u32, idom: &[u32]| {
        while left != right {
            while order[left as usize] > order[right as usize] {
                left = idom[left as usize];
            }
            while order[right as usize] > order[left as usize] {
                right = idom[right as usize];
            }
        }
        left
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &node in postorder.iter().skip(1) {
            let start = predecessor_offsets[node as usize] as usize;
            let end = predecessor_offsets[node as usize + 1] as usize;
            let mut processed = predecessors[start..end]
                .iter()
                .copied()
                .filter(|&predecessor| idom[predecessor as usize] != u32::MAX);
            let Some(mut next_idom) = processed.next() else {
                continue;
            };
            for predecessor in processed {
                next_idom = intersect(predecessor, next_idom, &idom);
            }
            if idom[node as usize] != next_idom {
                idom[node as usize] = next_idom;
                changed = true;
            }
        }
    }
    idom.into_iter()
        .map(|dominator| (dominator != u32::MAX).then_some(dominator))
        .collect()
}

/// Union of dominator frontiers for the dominators selected in `seed` into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn dominator_frontier_witness_into(
    node_count: u32,
    dominator_offsets: &[u32],
    dominator_targets: &[u32],
    predecessor_offsets: &[u32],
    predecessor_targets: &[u32],
    seed: &[u32],
    output: &mut Vec<u32>,
) {
    let node_count = node_count as usize;
    let words = node_count.div_ceil(32);
    if output.capacity() < words {
        output.reserve(words.saturating_sub(output.len()));
    }
    output.clear();
    output.resize(words, 0);
    for dominator in 0..node_count {
        if seed
            .get(dominator / 32)
            .is_none_or(|word| word & (1_u32 << (dominator % 32)) == 0)
        {
            continue;
        }
        let dominated = dominator_offsets.get(dominator).copied().unwrap_or(0) as usize
            ..dominator_offsets.get(dominator + 1).copied().unwrap_or(0) as usize;
        let dominated_nodes = dominator_targets.get(dominated).unwrap_or_default();
        for node in 0..node_count {
            let strictly_dominated = node != dominator && dominated_nodes.contains(&(node as u32));
            if strictly_dominated {
                continue;
            }
            let predecessors = predecessor_offsets.get(node).copied().unwrap_or(0) as usize
                ..predecessor_offsets.get(node + 1).copied().unwrap_or(0) as usize;
            let has_dominated_predecessor = predecessor_targets
                .get(predecessors)
                .unwrap_or_default()
                .iter()
                .any(|predecessor| dominated_nodes.contains(predecessor));
            if has_dominated_predecessor {
                output[node / 32] |= 1_u32 << (node % 32);
            }
        }
    }
}

/// Union of dominator frontiers for the dominators selected in `seed`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn dominator_frontier_witness(
    node_count: u32,
    dominator_offsets: &[u32],
    dominator_targets: &[u32],
    predecessor_offsets: &[u32],
    predecessor_targets: &[u32],
    seed: &[u32],
) -> Vec<u32> {
    let mut output = Vec::with_capacity((node_count as usize).div_ceil(32));
    dominator_frontier_witness_into(
        node_count,
        dominator_offsets,
        dominator_targets,
        predecessor_offsets,
        predecessor_targets,
        seed,
        &mut output,
    );
    output
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExplodedIfdsScratchWitness {
    /// Flattened edge scratch buffer.
    pub edges_flat: Vec<(u32, u32)>,
    /// Dense boolean mask of killed facts.
    pub killed: Vec<bool>,
    /// Generation rule offset scratch buffer.
    pub gen_offsets: Vec<usize>,
    /// Generation rule cursor scratch buffer.
    pub gen_cursor: Vec<usize>,
    /// Generated fact IDs scratch buffer.
    pub gen_facts: Vec<u32>,
    /// Row cursor scratch buffer for CSR materialization.
    pub cursor: Vec<usize>,
}

impl ExplodedIfdsScratchWitness {
    /// Create a new empty scratch workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fallible sequential mathematical witness for building dense-index exploded IFDS CSR into caller storage.
///
/// Validates input bounds before modifying `row_offsets`, `columns`, or `scratch`.
#[allow(clippy::too_many_arguments)]
pub fn try_exploded_ifds_csr_witness_into(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
    row_offsets: &mut Vec<u32>,
    columns: &mut Vec<u32>,
    scratch: &mut ExplodedIfdsScratchWitness,
) -> Result<(), String> {
    if procedure_count == 0 || blocks_per_procedure == 0 || facts_per_procedure == 0 {
        return Err(format!(
            "exploded IFDS CPU reference dimensions must be nonzero, got procs={procedure_count}, blocks={blocks_per_procedure}, facts={facts_per_procedure}. Fix: pass a real exploded-supergraph domain before parity comparison."
        ));
    }
    let Some(slots_per_procedure) =
        (blocks_per_procedure as usize).checked_mul(facts_per_procedure as usize)
    else {
        return Err("exploded IFDS slots_per_procedure overflow".to_string());
    };
    let Some(node_count) = (procedure_count as usize).checked_mul(slots_per_procedure) else {
        return Err("exploded IFDS node_count overflow".to_string());
    };
    if node_count > u32::MAX as usize {
        return Err("exploded IFDS node_count exceeds u32".to_string());
    }

    let index = |procedure: u32, block: u32, fact: u32| {
        procedure as usize * slots_per_procedure
            + block as usize * facts_per_procedure as usize
            + fact as usize
    };
    let in_domain = |procedure: u32, block: u32, fact: u32| {
        procedure < procedure_count && block < blocks_per_procedure && fact < facts_per_procedure
    };

    scratch.killed.clear();
    scratch.killed.resize(node_count, false);
    for &(p, b, f) in killed_facts {
        if in_domain(p, b, f) {
            let idx = index(p, b, f);
            scratch.killed[idx] = true;
        }
    }

    row_offsets.clear();
    row_offsets.resize(node_count + 1, 0);

    for &(procedure, source_block, destination_block) in intra_edges {
        if procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            if !scratch.killed[index(procedure, source_block, fact)] {
                row_offsets[index(procedure, source_block, fact) + 1] += 1;
            }
        }
        for &(generated_procedure, generated_block, fact) in generated_facts {
            if generated_procedure == procedure
                && generated_block == source_block
                && fact < facts_per_procedure
            {
                row_offsets[index(procedure, source_block, 0) + 1] += 1;
            }
        }
    }
    for &(source_procedure, source_block, destination_procedure, destination_block) in inter_edges {
        if source_procedure >= procedure_count
            || destination_procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            row_offsets[index(source_procedure, source_block, fact) + 1] += 1;
        }
    }

    let mut total_edges = 0_usize;
    for i in 0..node_count {
        total_edges += row_offsets[i + 1] as usize;
        row_offsets[i + 1] = total_edges as u32;
    }

    columns.clear();
    columns.resize(total_edges, 0);

    scratch.cursor.clear();
    scratch.cursor.extend(
        row_offsets[..node_count]
            .iter()
            .map(|&offset| offset as usize),
    );
    for &(procedure, source_block, destination_block) in intra_edges {
        if procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            if !scratch.killed[index(procedure, source_block, fact)] {
                let src = index(procedure, source_block, fact);
                let pos = scratch.cursor[src];
                columns[pos] = index(procedure, destination_block, fact) as u32;
                scratch.cursor[src] += 1;
            }
        }
        for &(generated_procedure, generated_block, fact) in generated_facts {
            if generated_procedure == procedure
                && generated_block == source_block
                && fact < facts_per_procedure
            {
                let src = index(procedure, source_block, 0);
                let pos = scratch.cursor[src];
                columns[pos] = index(procedure, destination_block, fact) as u32;
                scratch.cursor[src] += 1;
            }
        }
    }
    for &(source_procedure, source_block, destination_procedure, destination_block) in inter_edges {
        if source_procedure >= procedure_count
            || destination_procedure >= procedure_count
            || source_block >= blocks_per_procedure
            || destination_block >= blocks_per_procedure
        {
            continue;
        }
        for fact in 0..facts_per_procedure {
            let src = index(source_procedure, source_block, fact);
            let pos = scratch.cursor[src];
            columns[pos] = index(destination_procedure, destination_block, fact) as u32;
            scratch.cursor[src] += 1;
        }
    }

    scratch.edges_flat.clear();
    scratch.killed.clear();
    scratch.gen_offsets.clear();
    scratch.gen_cursor.clear();
    scratch.gen_facts.clear();
    scratch.cursor.clear();

    Ok(())
}

/// Build the dense-index exploded IFDS graph as CSR.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn exploded_ifds_csr_witness(
    procedure_count: u32,
    blocks_per_procedure: u32,
    facts_per_procedure: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    generated_facts: &[(u32, u32, u32)],
    killed_facts: &[(u32, u32, u32)],
) -> (Vec<u32>, Vec<u32>) {
    let mut row_offsets = Vec::new();
    let mut columns = Vec::new();
    let mut scratch = ExplodedIfdsScratchWitness::default();
    if try_exploded_ifds_csr_witness_into(
        procedure_count,
        blocks_per_procedure,
        facts_per_procedure,
        intra_edges,
        inter_edges,
        generated_facts,
        killed_facts,
        &mut row_offsets,
        &mut columns,
        &mut scratch,
    )
    .is_err()
    {
        return (vec![0], Vec::new());
    }
    (row_offsets, columns)
}

/// Immediate dominators derived from the textbook iterative dominator-set equation.
#[must_use]
pub fn dominator_sets_idoms_witness(
    node_count: u32,
    root: u32,
    edges: &[(u32, u32)],
) -> Vec<Option<u32>> {
    let n = node_count as usize;
    if n == 0 {
        return Vec::new();
    }
    if (root as usize) >= n {
        return vec![None; n];
    }
    let mut successors = vec![Vec::<usize>::new(); n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for &(source, destination) in edges {
        if (source as usize) < n && (destination as usize) < n {
            successors[source as usize].push(destination as usize);
            predecessors[destination as usize].push(source as usize);
        }
    }
    let mut reachable = vec![false; n];
    let mut stack = vec![root as usize];
    reachable[root as usize] = true;
    while let Some(node) = stack.pop() {
        for &successor in &successors[node] {
            if !reachable[successor] {
                reachable[successor] = true;
                stack.push(successor);
            }
        }
    }
    let reachable_nodes = reachable
        .iter()
        .enumerate()
        .filter_map(|(node, &is_reachable)| is_reachable.then_some(node))
        .collect::<Vec<_>>();
    let mut dominators = vec![vec![false; n]; n];
    for &node in &reachable_nodes {
        if node == root as usize {
            dominators[node][node] = true;
        } else {
            for &candidate in &reachable_nodes {
                dominators[node][candidate] = true;
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &node in &reachable_nodes {
            if node == root as usize {
                continue;
            }
            let mut next = vec![true; n];
            let mut saw_predecessor = false;
            for &predecessor in &predecessors[node] {
                if !reachable[predecessor] {
                    continue;
                }
                saw_predecessor = true;
                for candidate in 0..n {
                    next[candidate] &= dominators[predecessor][candidate];
                }
            }
            if !saw_predecessor {
                next.fill(false);
            }
            next[node] = true;
            if next != dominators[node] {
                dominators[node] = next;
                changed = true;
            }
        }
    }
    let mut idom = vec![None; n];
    idom[root as usize] = Some(root);
    for &node in &reachable_nodes {
        if node == root as usize {
            continue;
        }
        let strict = (0..n)
            .filter(|&candidate| candidate != node && dominators[node][candidate])
            .collect::<Vec<_>>();
        idom[node] = strict
            .iter()
            .copied()
            .find(|&candidate| {
                strict
                    .iter()
                    .all(|&other| other == candidate || dominators[candidate][other])
            })
            .map(|candidate| candidate as u32);
    }
    idom
}

/// Convert immediate dominators into sorted per-node dominator chains.
#[must_use]
pub fn idoms_to_dominator_sets_witness(idoms: &[Option<u32>], node_count: u32) -> Vec<Vec<u32>> {
    let n = node_count as usize;
    let mut sets = vec![Vec::new(); n];
    for node in 0..n {
        let mut current = Some(node as u32);
        let mut seen = vec![false; n];
        while let Some(dominator) = current {
            let index = dominator as usize;
            if index >= n || seen[index] {
                break;
            }
            seen[index] = true;
            sets[node].push(dominator);
            let parent = idoms.get(index).copied().flatten();
            current = parent.filter(|&next| next != dominator);
        }
        sets[node].sort_unstable();
    }
    sets
}

/// Stamp unassigned nodes in the intersection of two reachability bitsets.
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

/// Apply one diagonal sheaf-diffusion step in scalar arithmetic.
#[must_use]
pub fn sheaf_diffusion_step_witness(
    stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    sheaf_diffusion_step_witness_into(stalks, restriction_diagonal, damping, &mut out);
    out
}

/// Apply one diagonal sheaf-diffusion step into caller-owned storage.
pub fn sheaf_diffusion_step_witness_into(
    stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.extend(
        stalks
            .iter()
            .zip(restriction_diagonal)
            .map(|(&stalk, &restriction)| stalk - damping * restriction * stalk),
    );
}
/// Iterate diagonal sheaf diffusion into caller-owned ping-pong storage.
pub fn sheaf_diffusion_equilibrium_witness_into(
    initial_stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
    tolerance: f64,
    max_iterations: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> u32 {
    out.clear();
    out.extend_from_slice(initial_stalks);
    for iteration in 0..max_iterations {
        sheaf_diffusion_step_witness_into(out, restriction_diagonal, damping, scratch);
        let max_change = scratch
            .iter()
            .zip(out.iter())
            .map(|(next, current)| (next - current).abs())
            .fold(0.0_f64, f64::max);
        std::mem::swap(out, scratch);
        if max_change < tolerance {
            return iteration + 1;
        }
    }
    max_iterations
}

/// Mark stalks whose diffusion displacement exceeds the declared threshold.
pub fn sheaf_fusion_incompatible_witness_into(
    initial_stalks: &[f64],
    diffused_stalks: &[f64],
    divergence_threshold: f64,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.extend(
        initial_stalks
            .iter()
            .zip(diffused_stalks)
            .map(|(&initial, &diffused)| {
                u32::from((initial - diffused).abs() > divergence_threshold)
            }),
    );
}

/// Allocate incompatibility flags for one sheaf diffusion result.
#[must_use]
pub fn sheaf_fusion_incompatible_witness(
    initial_stalks: &[f64],
    diffused_stalks: &[f64],
    divergence_threshold: f64,
) -> Vec<u32> {
    let mut out = Vec::new();
    sheaf_fusion_incompatible_witness_into(
        initial_stalks,
        diffused_stalks,
        divergence_threshold,
        &mut out,
    );
    out
}

/// Compute the dominant eigenvalue and eigenvector of a diagonal sheaf Laplacian into caller storage.
///
/// Returns the dominant eigenvalue `max_i r[i]`. The eigenvector `v_out` is resized to `n` and
/// set to zero except `v_out[max_idx] = 1.0`, where `max_idx` is the index of the first maximum element.
/// If `restriction_diag` is empty, returns `0.0` and clears `v_out`.
pub fn sheaf_dominant_spectrum_witness_into(
    restriction_diag: &[f64],
    _iterations: u32,
    v_out: &mut Vec<f64>,
) -> f64 {
    v_out.clear();
    let n = restriction_diag.len();
    if n == 0 {
        return 0.0;
    }
    if v_out.capacity() < n {
        v_out.reserve(n.saturating_sub(v_out.len()));
    }
    v_out.resize(n, 0.0);
    let mut max_val = 0.0_f64;
    let mut max_idx = 0;
    for (i, &r) in restriction_diag.iter().enumerate() {
        if r > max_val {
            max_val = r;
            max_idx = i;
        }
    }
    v_out[max_idx] = 1.0;
    max_val
}

/// Compute the dominant eigenvalue and eigenvector of a diagonal sheaf Laplacian.
#[must_use]
pub fn sheaf_dominant_spectrum_witness(
    restriction_diag: &[f64],
    iterations: u32,
) -> (f64, Vec<f64>) {
    let mut v = Vec::with_capacity(restriction_diag.len());
    let lambda = sheaf_dominant_spectrum_witness_into(restriction_diag, iterations, &mut v);
    (lambda, v)
}

/// Compute the spectral gap signal in `[0, 1]` into caller eigenvector scratch.
pub fn sheaf_spectral_gap_witness_into(
    restriction_diag: &[f64],
    iterations: u32,
    v_scratch: &mut Vec<f64>,
) -> f64 {
    let lambda = sheaf_dominant_spectrum_witness_into(restriction_diag, iterations, v_scratch);
    let max_diag = restriction_diag.iter().cloned().fold(0.0_f64, f64::max);
    if max_diag <= 1e-20 {
        0.0
    } else {
        (lambda / max_diag).clamp(0.0, 1.0)
    }
}

/// Compute the spectral gap signal in `[0, 1]` derived from the dominant eigenvalue.
#[must_use]
pub fn sheaf_spectral_gap_witness(restriction_diag: &[f64], iterations: u32) -> f64 {
    let mut scratch = Vec::with_capacity(restriction_diag.len());
    sheaf_spectral_gap_witness_into(restriction_diag, iterations, &mut scratch)
}

/// Derive a suggested cluster count from the principal eigenvector sign pattern.
///
/// Items whose eigenvector entry has the same sign belong in the same cluster;
/// flips between consecutive items suggest cluster boundaries.
/// Returns the count of distinct sign runs (>= 1 for non-empty eigenvector, 0 for empty).
#[must_use]
pub fn sheaf_suggested_cluster_count_witness(eigenvector: &[f64]) -> u32 {
    if eigenvector.is_empty() {
        return 0;
    }
    let mut count: u32 = 1;
    let mut last_sign = eigenvector[0].signum();
    for &x in eigenvector.iter().skip(1) {
        let sign = x.signum();
        if sign != 0.0 && sign != last_sign && last_sign != 0.0 {
            count = count.saturating_add(1);
            last_sign = sign;
        } else if last_sign == 0.0 && sign != 0.0 {
            last_sign = sign;
        }
    }
    count
}

/// Evaluate a topologically ordered d-DNNF circuit.
///
/// Node tuples contain `(kind, child_offset, child_count)`. Kinds `1` and `2`
/// are positive and negative literals; kinds `3` and `4` are AND and OR.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed node, variable, child, or
/// topological-order indices and for model-count arithmetic overflow.
///
/// Evaluates a topologically ordered d-DNNF circuit into caller-owned storage.
pub fn try_ddnnf_evaluate_witness_into(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
    values: &mut Vec<u32>,
) -> Result<(), String> {
    if node_variables.len() != nodes.len() {
        return Err(format!(
            "d-DNNF node variable count {} does not match node count {}",
            node_variables.len(),
            nodes.len()
        ));
    }
    for &node in topological_order {
        let index = node as usize;
        let &(kind, child_offset, child_count) = nodes
            .get(index)
            .ok_or_else(|| format!("d-DNNF topological node {node} is out of bounds"))?;
        match kind {
            1 | 2 => {
                let variable = node_variables[index] as usize;
                if variable >= variable_assignments.len() {
                    return Err(format!(
                        "d-DNNF literal node {index} variable {variable} is outside assignment_count={}",
                        variable_assignments.len()
                    ));
                }
            }
            3 | 4 => {
                let start = child_offset as usize;
                let end = start
                    .checked_add(child_count as usize)
                    .ok_or_else(|| format!("d-DNNF child range overflows at node {index}"))?;
                let child_ids = children.get(start..end).ok_or_else(|| {
                    format!(
                        "d-DNNF node {index} child range {start}..{end} exceeds child_count={}",
                        children.len()
                    )
                })?;
                for &child in child_ids {
                    if child as usize >= nodes.len() {
                        return Err(format!(
                            "d-DNNF child node {child} is outside node_count={} at node {index}",
                            nodes.len()
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if values.capacity() < nodes.len() {
        values.reserve(nodes.len().saturating_sub(values.len()));
    }
    values.clear();
    values.resize(nodes.len(), 0);
    for &node in topological_order {
        let index = node as usize;
        let &(kind, child_offset, child_count) = &nodes[index];
        values[index] = match kind {
            1 | 2 => {
                let variable = node_variables[index] as usize;
                let assignment = variable_assignments[variable];
                u32::from(if kind == 1 {
                    assignment == 1 || assignment == u32::MAX
                } else {
                    assignment == 0 || assignment == u32::MAX
                })
            }
            3 | 4 => {
                let start = child_offset as usize;
                let end = start + child_count as usize;
                let child_ids = &children[start..end];
                let mut accumulator = u32::from(kind == 3);
                for &child in child_ids {
                    let child = child as usize;
                    let value = values[child];
                    accumulator = if kind == 3 {
                        accumulator.checked_mul(value)
                    } else {
                        accumulator.checked_add(value)
                    }
                    .ok_or_else(|| format!("d-DNNF model count overflows at node {index}"))?;
                }
                accumulator
            }
            _ => 0,
        };
    }
    Ok(())
}

/// Evaluate a topologically ordered d-DNNF circuit.
///
/// Node tuples contain `(kind, child_offset, child_count)`. Kinds `1` and `2`
/// are positive and negative literals; kinds `3` and `4` are AND and OR.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed node, variable, child, or
/// topological-order indices and for model-count arithmetic overflow.
pub fn try_ddnnf_evaluate_witness(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
) -> Result<Vec<u32>, String> {
    let mut values = Vec::with_capacity(nodes.len());
    try_ddnnf_evaluate_witness_into(
        nodes,
        node_variables,
        children,
        variable_assignments,
        topological_order,
        &mut values,
    )?;
    Ok(values)
}

/// Evaluate a valid topologically ordered d-DNNF circuit.
///
/// # Panics
///
/// Panics if the d-DNNF circuit structure, variable assignments, or topological order are invalid.
#[must_use]
pub fn ddnnf_evaluate_witness(
    nodes: &[(u32, u32, u32)],
    node_variables: &[u32],
    children: &[u32],
    variable_assignments: &[u32],
    topological_order: &[u32],
) -> Vec<u32> {
    try_ddnnf_evaluate_witness(
        nodes,
        node_variables,
        children,
        variable_assignments,
        topological_order,
    )
    .unwrap_or_else(|error| panic!("invalid d-DNNF witness input: {error}"))
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

/// Expand one dense matroid-exchange BFS frontier into caller-owned storage.
///
/// # Panics
///
/// Panics if `frontier` or `visited` lengths do not equal `element_count`,
/// if `element_count * element_count` overflows `usize`, or if `exchange_adjacency`
/// length does not match `element_count * element_count`.
pub fn matroid_exchange_bfs_step_witness_into(
    frontier: &[u32],
    exchange_adjacency: &[u32],
    visited: &[u32],
    element_count: usize,
    out: &mut Vec<u32>,
) -> bool {
    assert_eq!(frontier.len(), element_count, "complete matroid frontier");
    assert_eq!(visited.len(), element_count, "complete matroid visited set");
    let expected_adj = element_count
        .checked_mul(element_count)
        .expect("element_count * element_count overflows usize");
    assert_eq!(
        exchange_adjacency.len(),
        expected_adj,
        "complete dense matroid exchange graph"
    );
    if out.capacity() < element_count {
        out.reserve(element_count.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(element_count, 0);
    let mut changed = false;
    for destination in 0..element_count {
        if visited[destination] != 0 {
            continue;
        }
        let reached = (0..element_count).any(|source| {
            frontier[source] != 0 && exchange_adjacency[source * element_count + destination] != 0
        });
        if reached {
            out[destination] = 1;
            changed = true;
        }
    }
    changed
}

/// Expand one dense matroid-exchange BFS frontier.
#[must_use]
pub fn matroid_exchange_bfs_step_witness(
    frontier: &[u32],
    exchange_adjacency: &[u32],
    visited: &[u32],
    element_count: usize,
) -> (Vec<u32>, bool) {
    let mut next = Vec::with_capacity(element_count);
    let changed = matroid_exchange_bfs_step_witness_into(
        frontier,
        exchange_adjacency,
        visited,
        element_count,
        &mut next,
    );
    (next, changed)
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
            .ok_or_else(|| format!("toposort could not find positive indegree seed"))?;
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
    if edge_offsets.len() != (node_count as usize) + 1 {
        return Err("edge offsets length must match node count + 1".to_owned());
    }
    if edge_offsets.is_empty() || edge_offsets[0] != 0 {
        return Err("edge offsets must start at zero".to_owned());
    }
    if edge_offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err("non-monotonic CSR offsets".to_owned());
    }
    let Some(&last_offset) = edge_offsets.last() else {
        return Err("edge offsets must not be empty".to_owned());
    };
    if last_offset as usize != edge_targets.len() {
        return Err("edge offset bound does not match edge targets".to_owned());
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err("edge target count does not match edge kind mask".to_owned());
    }
    let words = {
        let lanes_per_node = context_limit
            .checked_mul(field_limit)
            .ok_or_else(|| "context_limit * field_limit overflowed u32".to_string())?;
        let total_bits = (node_count as u64)
            .checked_mul(lanes_per_node as u64)
            .ok_or_else(|| "node_count * lanes_per_node overflowed u64".to_string())?;
        u32::try_from((total_bits + 31) / 32)
            .map_err(|_| "tensor words count exceeds u32 limit".to_string())?
    };
    if tensor_in.len() < words as usize {
        return Err("tensor input buffer shorter than required tensor words".to_owned());
    }
    let words_len = words as usize;
    out.try_reserve(words_len.saturating_sub(out.len()))
        .map_err(|error| format!("failed to reserve tensor output buffer: {error}"))?;
    out.clear();
    out.resize(words_len, 0);

    for src in 0..node_count as usize {
        let (start, end) = (edge_offsets[src] as usize, edge_offsets[src + 1] as usize);
        for edge in start..end {
            let dst = edge_targets[edge];
            if (edge_kind_mask[edge] & allow_mask) == 0 || dst >= node_count {
                continue;
            }
            for ctx in 0..context_limit {
                for fld in 0..field_limit {
                    let s_bit = tensor_bit_index_witness(src as u32, ctx, fld, context_limit, field_limit);
                    if (tensor_in[(s_bit / 32) as usize] & (1 << (s_bit % 32))) != 0 {
                        let d_bit = tensor_bit_index_witness(dst, ctx, fld, context_limit, field_limit);
                        out[(d_bit / 32) as usize] |= 1 << (d_bit % 32);
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
        node_count, edge_offsets, edge_targets, edge_kind_mask, tensor_in, context_limit, field_limit, allow_mask, &mut out,
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
