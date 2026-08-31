//! The edge-pair oracle: Kahn over `(from, to)` pairs.

#[cfg(test)]
use super::error::ToposortError;

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
#[cfg(test)]
pub(crate) fn toposort(node_count: u32, edges: &[(u32, u32)]) -> Result<Vec<u32>, ToposortError> {
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
    vyre_reference::composition_witness::toposort_witness(node_count, edges).map_err(|err| {
        if let Some(rest) = err.strip_prefix("Cycle detected involving node ") {
            if let Ok(node) = rest.parse::<u32>() {
                return ToposortError::Cycle { node };
            }
        }
        ToposortError::InconsistentState { message: err }
    })
}

/// Reference alias for topological ordering.
#[cfg(test)]
pub(crate) fn reference_topo_order(
    node_count: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<u32>, ToposortError> {
    toposort(node_count, edges)
}

/// Compute the set of nodes reachable from `sources` over `edges`.
#[cfg(test)]
pub(crate) fn reference_reachable_set(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
) -> Result<std::collections::HashSet<u32>, crate::graph::reachable::UnknownNode> {
    crate::graph::reachable::reachable(node_count, edges, sources)
}

/// True iff every node in `targets` is reachable from `sources`.
#[cfg(test)]
pub(crate) fn reference_all_reachable(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
    targets: &[u32],
) -> Result<bool, crate::graph::reachable::UnknownNode> {
    let reach = reference_reachable_set(node_count, edges, sources)?;
    Ok(targets.iter().all(|t| reach.contains(t)))
}
