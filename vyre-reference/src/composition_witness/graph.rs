//! Sequential mathematical witnesses for graph analysis, dominator trees, homology, and matroids.

/// Sequential mathematical witness for immediate dominators (Cooper-Harvey-Kennedy).
/// Returns an array `idom` where `idom[u]` is the immediate dominator of `u`, or `u32::MAX` if unreachable.
#[must_use]
pub fn dominator_tree_witness(
    node_count: u32,
    root: u32,
    edges: &[(u32, u32)],
) -> Vec<u32> {
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
    let intersect = |mut b1: usize, mut b2: usize, idom: &[u32], postorder_num: &[usize]| -> usize {
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
    if mask.len() < n_us * n_us {
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

/// One deterministic level-synchronous Edmonds augmentation witness.
#[must_use]
pub fn matroid_intersection_augmentation_witness(
    exchange_adjacency: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: usize,
) -> Vec<u32> {
    assert_eq!(exchange_adjacency.len(), n * n);
    assert_eq!(sources.len(), n);
    assert_eq!(sinks.len(), n);
    assert_eq!(set_x.len(), n);
    let mut output = set_x.to_vec();
    let mut frontier = sources.iter().map(|&value| u32::from(value != 0)).collect::<Vec<_>>();
    let mut visited = frontier.clone();
    let mut parent = vec![u32::MAX; n];
    let mut target = (0..n).find(|&node| frontier[node] != 0 && sinks[node] != 0);
    while target.is_none() && frontier.iter().any(|&value| value != 0) {
        let mut next = vec![0_u32; n];
        for destination in 0..n {
            if visited[destination] != 0 {
                continue;
            }
            if let Some(source) = (0..n).find(|&source| {
                frontier[source] != 0
                    && exchange_adjacency[source * n + destination] != 0
            }) {
                parent[destination] = source as u32;
                next[destination] = 1;
                visited[destination] = 1;
            }
        }
        target = (0..n).find(|&node| next[node] != 0 && sinks[node] != 0);
        frontier = next;
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
    output
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
    assert!((root as usize) < n);
    let mut successors = vec![Vec::<u32>::new(); n];
    let mut predecessors = vec![Vec::<u32>::new(); n];
    for &(source, destination) in edges {
        if (source as usize) < n && (destination as usize) < n {
            successors[source as usize].push(destination);
            predecessors[destination as usize].push(source);
        }
    }

    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack = vec![(root, 0_usize)];
    visited[root as usize] = true;
    while let Some((node, edge_index)) = stack.pop() {
        let node_index = node as usize;
        if edge_index < successors[node_index].len() {
            stack.push((node, edge_index + 1));
            let successor = successors[node_index][edge_index];
            if !visited[successor as usize] {
                visited[successor as usize] = true;
                stack.push((successor, 0));
            }
        } else {
            postorder.push(node);
        }
    }
    let reverse_postorder = postorder.into_iter().rev().collect::<Vec<_>>();
    let mut order = vec![usize::MAX; n];
    for (index, &node) in reverse_postorder.iter().enumerate() {
        order[node as usize] = index;
    }
    let mut idom = vec![None; n];
    idom[root as usize] = Some(root);
    let intersect = |mut left: u32, mut right: u32, idom: &[Option<u32>]| {
        while left != right {
            while order[left as usize] > order[right as usize] {
                left = idom[left as usize].expect("processed dominator predecessor");
            }
            while order[right as usize] > order[left as usize] {
                right = idom[right as usize].expect("processed dominator predecessor");
            }
        }
        left
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &node in reverse_postorder.iter().skip(1) {
            let mut processed = predecessors[node as usize]
                .iter()
                .copied()
                .filter(|&predecessor| idom[predecessor as usize].is_some());
            let Some(mut next_idom) = processed.next() else {
                continue;
            };
            for predecessor in processed {
                next_idom = intersect(predecessor, next_idom, &idom);
            }
            if idom[node as usize] != Some(next_idom) {
                idom[node as usize] = Some(next_idom);
                changed = true;
            }
        }
    }
    idom
}
