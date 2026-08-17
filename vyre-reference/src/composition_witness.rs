//! Independent, obviously correct sequential mathematical witnesses for composite operations.
//!
//! Per Section 183.3, reference witnesses use simple sequential mathematical algorithms
//! without Blelloch scheduling, workgroup decomposition, frontier queues, or other GPU optimizations.
//! Composed Programs continue to run through the generic reference interpreter, with independent
//! known-answer cases where interpreter parity alone would compare an implementation with itself.

use vyre_spec::Semiring;

/// Sequential mathematical witness for a generalized semiring matrix multiplication: `C = A ⊗ B`.
///
/// Shape: `A` is `m × k`, `B` is `k × n`, output `C` is `m × n`.
///
/// # Panics
/// Panics if input slice dimensions do not match `m * k` and `k * n`.
#[must_use]
pub fn semiring_gemm_witness(
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    semiring: Semiring,
) -> Vec<u32> {
    assert_eq!(
        a.len(),
        m * k,
        "A dimension mismatch in semiring GEMM witness"
    );
    assert_eq!(
        b.len(),
        k * n,
        "B dimension mismatch in semiring GEMM witness"
    );

    let zero = semiring.identity();
    let mut c = vec![zero; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                let a_val = a[i * k + p];
                let b_val = b[p * n + j];

                let term = match semiring {
                    Semiring::Real => a_val.wrapping_mul(b_val),
                    Semiring::MinPlus | Semiring::MaxPlus => a_val.saturating_add(b_val),
                    Semiring::MaxTimes => a_val.wrapping_mul(b_val),
                    Semiring::BoolOr => a_val & b_val,
                    Semiring::BoolAnd => a_val | b_val,
                    Semiring::Gf2 => a_val & b_val,
                    Semiring::Lineage => a_val | b_val,
                };

                acc = match semiring {
                    Semiring::Real => acc.wrapping_add(term),
                    Semiring::MinPlus => acc.min(term),
                    Semiring::MaxPlus | Semiring::MaxTimes => acc.max(term),
                    Semiring::BoolOr | Semiring::Lineage => acc | term,
                    Semiring::BoolAnd => acc & term,
                    Semiring::Gf2 => acc ^ term,
                };
            }
            c[i * n + j] = acc;
        }
    }

    c
}

/// Sequential mathematical witness for inclusive and exclusive prefix scans.
#[must_use]
pub fn prefix_scan_witness(
    input: &[u32],
    inclusive: bool,
    combine_op: impl Fn(u32, u32) -> u32,
    identity: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = identity;

    for &val in input {
        if inclusive {
            acc = combine_op(acc, val);
            out.push(acc);
        } else {
            out.push(acc);
            acc = combine_op(acc, val);
        }
    }

    out
}

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
    let mut queue = std::collections::VecDeque::new();
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

/// Sequential mathematical witness for bitwise AND on packed bitsets.
#[must_use]
pub fn bitset_and_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & b).collect()
}


/// Sequential mathematical witness for bitwise AND on packed bitsets writing into caller storage.
pub fn bitset_and_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & b));
}

/// Sequential mathematical witness for bitwise OR on packed bitsets writing into caller storage.
pub fn bitset_or_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a | b));
}

/// Sequential mathematical witness for bitwise XOR on packed bitsets writing into caller storage.
pub fn bitset_xor_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a ^ b));
}

/// Sequential mathematical witness for bitwise NOT on packed bitsets writing into caller storage.
pub fn bitset_not_witness_into(input: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.extend(input.iter().map(|&w| !w));
}

/// Sequential mathematical witness for bitwise AND-NOT on packed bitsets writing into caller storage.
pub fn bitset_and_not_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & !b));
}

/// Sequential mathematical witness for bitset popcount writing into caller storage.
pub fn bitset_popcount_witness_into(input: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.extend(input.iter().map(|&w| w.count_ones()));
}
/// Sequential mathematical witness for bitwise OR on packed bitsets.
#[must_use]
pub fn bitset_or_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a | b).collect()
}

/// Sequential mathematical witness for bitwise XOR on packed bitsets.
#[must_use]
pub fn bitset_xor_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a ^ b).collect()
}

/// Sequential mathematical witness for bitwise NOT on packed bitsets.
#[must_use]
pub fn bitset_not_witness(input: &[u32]) -> Vec<u32> {
    input.iter().map(|&w| !w).collect()
}

/// Sequential mathematical witness for bitwise AND-NOT on packed bitsets (`lhs & !rhs`).
#[must_use]
pub fn bitset_and_not_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & !b).collect()
}

/// Sequential mathematical witness for bitset equality.
#[must_use]
pub fn bitset_equal_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    lhs == rhs
}

/// Sequential mathematical witness for bitset subset relation (`lhs ⊆ rhs`).
#[must_use]
pub fn bitset_subset_of_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    lhs.iter().zip(rhs.iter()).all(|(&a, &b)| (a & !b) == 0)
}

/// Sequential mathematical witness for bitset membership test.
#[must_use]
pub fn bitset_contains_witness(input: &[u32], bit_idx: u32) -> bool {
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word < input.len() {
        (input[word] & (1 << bit)) != 0
    } else {
        false
    }
}

/// Sequential mathematical witness for setting a bit in a bitset.
#[must_use]
pub fn bitset_set_bit_witness(target: &[u32], bit_idx: u32) -> Vec<u32> {
    let mut out = target.to_vec();
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word >= out.len() {
        out.resize(word + 1, 0);
    }
    out[word] |= 1 << bit;
    out
}

/// Sequential mathematical witness for clearing a bit in a bitset.
#[must_use]
pub fn bitset_clear_bit_witness(target: &[u32], bit_idx: u32) -> Vec<u32> {
    let mut out = target.to_vec();
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word < out.len() {
        out[word] &= !(1 << bit);
    }
    out
}

/// Sequential mathematical witness for bitset popcount (number of set bits per word).
#[must_use]
pub fn bitset_popcount_witness(input: &[u32]) -> Vec<u32> {
    input.iter().map(|&w| w.count_ones()).collect()
}

/// Sequential mathematical witness for one-step forward traversal over CSR graph.
#[must_use]
pub fn csr_forward_traverse_witness(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    frontier: &[u32],
) -> Vec<u32> {
    let words = (node_count as usize + 31) / 32;
    let mut next_frontier = vec![0u32; words];
    for u in 0..node_count as usize {
        let word = u / 32;
        let bit = u % 32;
        if word < frontier.len() && (frontier[word] & (1 << bit)) != 0 {
            let start = row_offsets[u] as usize;
            let end = row_offsets[u + 1] as usize;
            for &v in &col_indices[start..end] {
                let v = v as usize;
                if v < node_count as usize {
                    let nw = v / 32;
                    let nb = v % 32;
                    next_frontier[nw] |= 1 << nb;
                }
            }
        }
    }
    next_frontier
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

/// Sequential mathematical witness for byte histogram (256 bins).
#[must_use]
pub fn byte_histogram_witness(input: &[u8]) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for &b in input {
        counts[b as usize] += 1;
    }
    counts
}

/// Sequential mathematical witness for character class mapping.
#[must_use]
pub fn char_class_witness(input: &[u8], table: &[u32; 256]) -> Vec<u32> {
    input.iter().map(|&b| table[b as usize]).collect()
}

/// Sequential mathematical witness for line start indexing.
#[must_use]
pub fn line_index_witness(input: &[u8]) -> Vec<u32> {
    let mut indices = vec![0u32];
    for (i, &b) in input.iter().enumerate() {
        if b == b'\n' && i + 1 < input.len() {
            indices.push((i + 1) as u32);
        }
    }
    indices
}

/// Sequential mathematical witness for UTF-8 shape counts (ASCII, 2-byte, 3-byte, 4-byte).
#[must_use]
pub fn utf8_shape_counts_witness(input: &[u8]) -> [u32; 4] {
    let mut counts = [0u32; 4];
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b < 0x80 {
            counts[0] += 1;
            i += 1;
        } else if (b & 0xE0) == 0xC0 {
            counts[1] += 1;
            i += 2;
        } else if (b & 0xF0) == 0xE0 {
            counts[2] += 1;
            i += 3;
        } else if (b & 0xF8) == 0xF0 {
            counts[3] += 1;
            i += 4;
        } else {
            i += 1;
        }
    }
    counts
}

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
