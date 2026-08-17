//! The edge-pair oracle: Kahn over `(from, to)` pairs.

use super::error::{toposort_allocation, ToposortError};

/// CPU reference: Kahn's algorithm over `(node_count, edges)`.
///
/// `edges` is a slice of `(from, to)` u32 pairs  -  `from` depends on
/// `to`, so `to` comes first in the sort. Returns a `Vec<u32>` in
/// topological order on success, or `ToposortError::Cycle` if the
/// graph has a cycle.
///
/// # Errors
///
/// Returns `ToposortError::Cycle` when the input has a cycle, or
/// `ToposortError::UnknownNode` when an edge names a node id
/// outside `0..node_count`.
pub fn toposort(node_count: u32, edges: &[(u32, u32)]) -> Result<Vec<u32>, ToposortError> {
    const NONE: usize = usize::MAX;

    validate_toposort_edge_ids(node_count, edges)?;

    let n = node_count as usize;
    let mut indeg: Vec<u32> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut indeg,
        n,
        "toposort CPU oracle",
        "toposort indegree scratch",
        toposort_allocation,
    )?;
    indeg.resize(n, 0);
    let mut outgoing_head: Vec<usize> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut outgoing_head,
        n,
        "toposort CPU oracle",
        "toposort outgoing heads",
        toposort_allocation,
    )?;
    outgoing_head.resize(n, NONE);
    let mut outgoing_to: Vec<u32> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut outgoing_to,
        edges.len(),
        "toposort CPU oracle",
        "toposort outgoing targets",
        toposort_allocation,
    )?;
    let mut outgoing_next: Vec<usize> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut outgoing_next,
        edges.len(),
        "toposort CPU oracle",
        "toposort outgoing links",
        toposort_allocation,
    )?;
    let mut depends_head: Vec<usize> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut depends_head,
        n,
        "toposort CPU oracle",
        "toposort dependency heads",
        toposort_allocation,
    )?;
    depends_head.resize(n, NONE);
    let mut depends_to: Vec<u32> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut depends_to,
        edges.len(),
        "toposort CPU oracle",
        "toposort dependency targets",
        toposort_allocation,
    )?;
    let mut depends_next: Vec<usize> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut depends_next,
        edges.len(),
        "toposort CPU oracle",
        "toposort dependency links",
        toposort_allocation,
    )?;

    for &(from, to) in edges {
        let outgoing_idx = outgoing_to.len();
        outgoing_to.push(from);
        outgoing_next.push(outgoing_head[to as usize]);
        outgoing_head[to as usize] = outgoing_idx;

        let depends_idx = depends_to.len();
        depends_to.push(to);
        depends_next.push(depends_head[from as usize]);
        depends_head[from as usize] = depends_idx;

        indeg[from as usize] = indeg[from as usize]
            .checked_add(1)
            .ok_or(ToposortError::IndegreeOverflow { node: from })?;
    }

    let mut queue: Vec<u32> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut queue,
        n,
        "toposort CPU oracle",
        "toposort zero-indegree queue",
        toposort_allocation,
    )?;
    for v in 0..node_count {
        if indeg[v as usize] == 0 {
            queue.push(v);
        }
    }
    let mut out: Vec<u32> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut out,
        n,
        "toposort CPU oracle",
        "toposort output order",
        toposort_allocation,
    )?;

    while let Some(&v) = queue.last() {
        queue.pop();
        out.push(v);
        let mut edge = outgoing_head[v as usize];
        while edge != NONE {
            let u = outgoing_to[edge];
            let slot = &mut indeg[u as usize];
            *slot = slot.checked_sub(1).ok_or_else(|| {
                ToposortError::InconsistentState {
                    message: format!(
                        "toposort indegree underflow for node {u}. Fix: rebuild dependency edges before scheduling."
                    ),
                }
            })?;
            if *slot == 0 {
                queue.push(u);
            }
            edge = outgoing_next[edge];
        }
    }

    if out.len() != n {
        // AUDIT_2026-04-24 F-TS-03: returning the first node with
        // indeg > 0 is misleading  -  that node may be *downstream* of
        // a cycle (its predecessor is stuck, not itself). Instead,
        // walk outgoing "depends on" edges from any unemitted node
        // until we revisit a node already on the walk  -  that revisit
        // point is guaranteed to lie on the cycle.
        let seed = indeg
            .iter()
            .enumerate()
            .find(|(_, deg)| **deg > 0)
            .map(|(i, _)| i as u32)
            .ok_or_else(|| {
                ToposortError::InconsistentState {
                    message: format!(
                        "toposort could not find a positive-indegree seed while output_len={} node_count={n}. Fix: rebuild dependency indegrees before scheduling.",
                        out.len()
                    ),
                }
            });
        let seed = seed?;
        let mut on_stack: Vec<bool> = Vec::new();
        crate::plumbing::host::scratch::reserve_items_with(
            &mut on_stack,
            n,
            "toposort CPU oracle",
            "toposort cycle diagnosis stack",
            toposort_allocation,
        )?;
        on_stack.resize(n, false);
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
                None => {
                    return Err(ToposortError::InconsistentState {
                        message: format!(
                            "toposort cycle diagnosis found stuck node {cursor} without an unemitted dependency. Fix: rebuild the dependency adjacency; this state is inconsistent with Kahn's invariant."
                        ),
                    });
                }
            }
        };
        return Err(ToposortError::Cycle { node: cycle_node });
    }
    Ok(out)
}

fn validate_toposort_edge_ids(node_count: u32, edges: &[(u32, u32)]) -> Result<(), ToposortError> {
    for (edge_idx, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(ToposortError::UnknownNode {
                edge: edge_idx,
                node: from,
            });
        }
        if to >= node_count {
            return Err(ToposortError::UnknownNode {
                edge: edge_idx,
                node: to,
            });
        }
    }
    Ok(())
}

/// Reference alias for topological ordering.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_topo_order(
    node_count: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<u32>, ToposortError> {
    toposort(node_count, edges)
}

/// Compute the set of nodes reachable from `sources` over `edges`.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_reachable_set(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
) -> Result<std::collections::HashSet<u32>, crate::graph::reachable::UnknownNode> {
    crate::graph::reachable::reachable(node_count, edges, sources)
}

/// True iff every node in `targets` is reachable from `sources`.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_all_reachable(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
    targets: &[u32],
) -> Result<bool, crate::graph::reachable::UnknownNode> {
    let reach = reference_reachable_set(node_count, edges, sources)?;
    Ok(targets.iter().all(|t| reach.contains(t)))
}
