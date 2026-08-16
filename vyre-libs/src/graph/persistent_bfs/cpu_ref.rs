#[cfg(any(test, feature = "cpu-parity"))]
use super::validate::validate_persistent_bfs_inputs;
#[cfg(any(test, feature = "cpu-parity"))]
use crate::graph::csr_closure_inputs::CsrClosureInputs;

/// Convergence outcome of one persistent-BFS CPU reference run.
///
/// The `changed` flag alone (`1` if any step added new nodes) cannot tell a
/// caller whether the fixpoint was actually reached or the loop merely ran
/// out of `max_iters` while still growing. This struct separates the two so a
/// consumer can enforce a loud non-convergence policy: a run that exhausts
/// `max_iters` while still adding nodes returns `converged = false` and an
/// under-approximated frontier, which the caller can reject instead of
/// silently trusting a partial closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(test, feature = "cpu-parity"))]
pub struct PersistentBfsConvergence {
    /// Sticky flag: `1` if any step added new nodes, else `0`.
    pub changed: u32,
    /// `true` if a step added nothing (the fixpoint was reached) before
    /// `max_iters` was exhausted; `false` if the loop ran all `max_iters`
    /// steps while still adding nodes, in which case the frontier is an
    /// under-approximation of the true closure.
    pub converged: bool,
    /// Number of traversal steps actually run: the step at which the loop
    /// stopped. Equals `max_iters` exactly when `converged` is `false`.
    pub stop_iter: u32,
}

/// CPU reference: run BFS up to `inputs.max_iters` steps, accumulating into a
/// running bitset.  Returns the final frontier and a sticky `changed`
/// flag (`1` if any step added new nodes, else `0`).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(inputs: CsrClosureInputs<'_>, frontier_in: &[u32]) -> (Vec<u32>, u32) {
    try_cpu_ref(inputs, frontier_in).expect(
        "Fix: reject malformed CSR/frontier via try_cpu_ref; parity wrappers must not pass hostile layouts",
    )
}

/// Fallible CPU reference for persistent BFS.
///
/// This is the primitive-owned entry point for parity wrappers that must reject
/// hostile CSR/frontier inputs without panicking.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, u32), String> {
    let mut out = Vec::new();
    let changed = try_cpu_ref_into(inputs, frontier_in, &mut out)?;
    Ok((out, changed))
}

/// Caller-owned workspace for repeated persistent-BFS CPU oracle runs.
///
/// Conformance and backend parity sweeps call this oracle across large generated
/// graph corpora. Reusing the per-iteration frontier scratch avoids a heap
/// allocation per proof case while preserving the allocating compatibility API.
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Debug, Default, Clone)]
pub(crate) struct PersistentBfsCpuScratch {
    /// Temporary frontier produced by one CSR expansion step.
    pub step: Vec<u32>,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl PersistentBfsCpuScratch {
    /// Create an empty reusable persistent-BFS workspace.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// CPU reference into caller-owned output storage.
///
/// Runs BFS up to `inputs.max_iters` steps, accumulating into `frontier_out`.
/// Returns a sticky changed flag (`1` if any step added new nodes, else `0`).
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn cpu_ref_into(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
) -> u32 {
    try_cpu_ref_into(inputs, frontier_in, frontier_out).expect(
        "Fix: reject malformed CSR/frontier via try_cpu_ref_into; parity wrappers must not pass hostile layouts",
    )
}

/// Fallible CPU reference into caller-owned output storage.
///
/// On error, `frontier_out` is left unchanged. This lets integration tests and
/// dispatch wrappers treat malformed graph/frontier data as a typed finding
/// instead of a panic or partially clobbered oracle output.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
) -> Result<u32, String> {
    let mut scratch = PersistentBfsCpuScratch::default();
    try_cpu_ref_into_with_scratch(inputs, frontier_in, frontier_out, &mut scratch)
}

/// Fallible CPU reference into caller-owned output and scratch storage.
///
/// On validation error, `frontier_out` and `scratch` are left unchanged. This
/// lets integration tests and dispatch wrappers treat malformed graph/frontier
/// data as a typed finding instead of a panic or partially clobbered oracle
/// state.
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn try_cpu_ref_into_with_scratch(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
    scratch: &mut PersistentBfsCpuScratch,
) -> Result<u32, String> {
    Ok(
        try_cpu_ref_converged_into_with_scratch(inputs, frontier_in, frontier_out, scratch, None)?
            .changed,
    )
}

/// Fallible CPU reference reporting convergence into caller-owned output and
/// scratch storage.
///
/// This is the single owner of the persistent-BFS accumulation loop. The
/// sticky-`changed`-only wrappers above delegate here and drop the extra
/// convergence detail; callers that must distinguish a reached fixpoint from a
/// `max_iters` exhaustion use this entry point directly.
///
/// `density_active`, when `Some`, is filled with exactly `inputs.max_iters`
/// entries where entry `i` is the popcount of the frontier after traversal step
/// `i`. Once the closure converges every later entry repeats the converged
/// popcount, mirroring the device density buffer (whose loop keeps running the
/// remaining budget over an unchanged frontier). This is the CPU source of truth
/// for the device per-iteration density readback; it shares the one accumulation
/// loop so the popcount trajectory cannot diverge from the convergence
/// trajectory.
///
/// On validation error, `frontier_out` and `scratch` are left unchanged.
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn try_cpu_ref_converged_into_with_scratch(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
    scratch: &mut PersistentBfsCpuScratch,
    mut density_active: Option<&mut Vec<u32>>,
) -> Result<PersistentBfsConvergence, String> {
    let graph = inputs.graph;
    let max_iters = inputs.max_iters;
    let layout = validate_persistent_bfs_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        frontier_in,
    )?;
    let words = layout.words;
    crate::scratch::reserve_items(
        frontier_out,
        words,
        "persistent BFS CPU oracle",
        "frontier output",
    )?;
    crate::scratch::reserve_items(
        &mut scratch.step,
        words,
        "persistent BFS CPU oracle",
        "per-iteration frontier scratch",
    )?;
    frontier_out.clear();
    frontier_out.extend_from_slice(frontier_in);
    frontier_out.resize(words, 0);
    scratch.step.clear();
    scratch.step.resize(words, 0);
    if let Some(density) = density_active.as_deref_mut() {
        density.clear();
        density.reserve(max_iters as usize);
    }
    let mut changed = 0u32;
    let mut converged = false;
    let mut stop_iter = 0u32;

    for iter in 0..max_iters {
        crate::graph::csr_forward_traverse::cpu_ref_into(
            graph.node_count,
            graph.edge_offsets,
            graph.edge_targets,
            graph.edge_kind_mask,
            frontier_out,
            inputs.allow_mask,
            &mut scratch.step,
        );
        stop_iter = iter + 1;
        let mut step_changed = false;
        for w in 0..words {
            let old = frontier_out[w];
            frontier_out[w] |= scratch.step[w];
            if frontier_out[w] != old {
                step_changed = true;
            }
        }
        if let Some(density) = density_active.as_deref_mut() {
            density.push(frontier_out.iter().map(|w| w.count_ones()).sum());
        }
        if step_changed {
            changed = 1;
        } else {
            converged = true;
            break;
        }
    }
    // Pad the density trajectory to the full budget: after convergence the device
    // loop keeps running the remaining iterations over an unchanged frontier, so
    // every entry past `stop_iter` repeats the converged popcount.
    if let Some(density) = density_active {
        let fill = density.last().copied().unwrap_or(0);
        while (density.len() as u32) < max_iters {
            density.push(fill);
        }
    }
    Ok(PersistentBfsConvergence {
        changed,
        converged,
        stop_iter,
    })
}

/// Fallible CPU reference reporting convergence, allocating a fresh frontier.
///
/// Like [`try_cpu_ref`] but returns a [`PersistentBfsConvergence`] so a caller
/// can reject an under-approximated closure (`converged == false`) loudly
/// instead of silently trusting a frontier the loop never drove to a fixpoint.
/// This is the CPU source of truth for the device converged readback.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_converged(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, PersistentBfsConvergence), String> {
    let mut out = Vec::new();
    let mut scratch = PersistentBfsCpuScratch::default();
    let outcome =
        try_cpu_ref_converged_into_with_scratch(inputs, frontier_in, &mut out, &mut scratch, None)?;
    Ok((out, outcome))
}

/// Fallible CPU reference reporting the per-iteration frontier-density
/// trajectory, allocating a fresh frontier and density array.
///
/// Returns the converged frontier, the [`PersistentBfsConvergence`] outcome, and
/// a `max_iters`-length `active` array where `active[i]` is the popcount of the
/// frontier after traversal step `i` (flat once the closure converges). This is
/// the CPU source of truth for the device density readback emitted by
/// [`super::program::persistent_bfs_with_density`]: a host caller reconstructs
/// every per-iteration frontier-density aggregate from `active` plus the seed
/// popcount without a per-step device round-trip, and this oracle proves the
/// device trajectory matches the reference bit-for-bit.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_density(
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, PersistentBfsConvergence, Vec<u32>), String> {
    let mut out = Vec::new();
    let mut active = Vec::new();
    let mut scratch = PersistentBfsCpuScratch::default();
    let outcome = try_cpu_ref_converged_into_with_scratch(
        inputs,
        frontier_in,
        &mut out,
        &mut scratch,
        Some(&mut active),
    )?;
    Ok((out, outcome, active))
}
