//! Sequential mathematical witnesses for matroid intersection, augmentation, and exchange BFS.

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

/// One dense matroid-exchange BFS frontier expansion writing into caller storage.
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
    let expected_adj = element_count.checked_mul(element_count).expect(
        "Fix: keep element_count * element_count within usize bounds for dense matroid adjacency",
    );
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
