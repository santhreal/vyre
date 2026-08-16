use crate::graph::csr_closure_inputs::CsrClosureInputs;
#[cfg(test)]
use crate::graph::csr_closure_inputs::CsrGraphView;
use crate::graph::persistent_bfs::{
    try_cpu_ref as try_reference_persistent_bfs, try_cpu_ref_converged, PersistentBfsConvergence,
};

/// Run up to `inputs.max_iters` BFS steps starting from `frontier_in`,
/// returning the saturated frontier and a sticky changed-flag (1 if
/// any iteration added new bits, 0 if the seed was already
/// saturated). Bumps the dataflow-fixpoint substrate counter.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn bfs_expand(inputs: CsrClosureInputs<'_>, frontier_in: &[u32]) -> (Vec<u32>, u32) {
    try_bfs_expand(inputs, frontier_in).unwrap_or_else(|err| {
        panic!("persistent BFS self-substrate reference rejected input. {err}")
    })
}

/// Fallible persistent-BFS substrate reference wrapper.
///
/// # Errors
///
/// Rejects malformed CSR/frontier shapes, propagating the primitive diagnostic.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_bfs_expand(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, u32), String> {
    use crate::telemetry::{bump, graph_dispatch_calls};
    bump(&graph_dispatch_calls);
    try_reference_persistent_bfs(inputs, frontier_in)
}

/// Persistent-BFS reference that also reports the convergence outcome: the
/// saturated frontier plus [`PersistentBfsConvergence`] (sticky changed flag,
/// whether the fixpoint was reached within `inputs.max_iters`, and the stop
/// step). Use this to check a device converged word against the CPU oracle.
/// Bumps the dataflow-fixpoint substrate counter.
///
/// # Errors
///
/// Rejects malformed CSR/frontier shapes, propagating the primitive diagnostic.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_bfs_expand_converged(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, PersistentBfsConvergence), String> {
    use crate::telemetry::{bump, graph_dispatch_calls};
    bump(&graph_dispatch_calls);
    try_cpu_ref_converged(inputs, frontier_in)
}

/// Convenience: compute the forward-reachable set of `seed` under `allow_mask`
/// with a budget of `graph.node_count` steps, which saturates any simple
/// reachability chain. Returns just the frontier; callers wanting the
/// changed-flag should use [`bfs_expand`] directly.
#[must_use]
#[cfg(test)]
pub fn forward_reach(graph: CsrGraphView<'_>, seed: &[u32], allow_mask: u32) -> Vec<u32> {
    let (out, _changed) = bfs_expand(
        CsrClosureInputs {
            graph,
            allow_mask,
            max_iters: graph.node_count,
        },
        seed,
    );
    out
}
