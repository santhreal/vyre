//! Sequential mathematical witnesses for graph dominator trees, immediate dominators, and frontiers.

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
    if (root as usize) < n {
        let root_idx = root as usize;
        let mut stack = Vec::with_capacity(64);
        visited[root_idx] = true;
        stack.push((root_idx, 0usize));
        while let Some((u, succ_idx)) = stack.pop() {
            if succ_idx < succs[u].len() {
                let v = succs[u][succ_idx] as usize;
                stack.push((u, succ_idx + 1));
                if !visited[v] {
                    visited[v] = true;
                    stack.push((v, 0usize));
                }
            } else {
                postorder.push(u);
            }
        }
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
    assert_eq!(
        seed.len(),
        words,
        "expected seed length {words} words for {node_count} nodes, got {}",
        seed.len()
    );
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
