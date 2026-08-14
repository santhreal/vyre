//! Strongly-connected-component decomposition driven through the dataflow
//! fixpoint: per-pivot forward/backward bitset packing and the host driver.

#[cfg(any(test, feature = "cpu-parity"))]
use super::reference_closures::reachability_closure_into;
#[cfg(any(test, feature = "cpu-parity"))]
use super::DataflowFixpointScratch;
#[cfg(any(test, feature = "cpu-parity"))]
use crate::device::scratch::reserve_vec_capacity_or_panic;

#[cfg(any(test, feature = "cpu-parity"))]
impl DataflowFixpointScratch {
    /// Forward-reach bitset produced by the last pivot query.
    #[must_use]
    pub fn forward_bitset(&self) -> &[u32] {
        &self.forward
    }

    /// Backward-reach bitset produced by the last pivot query.
    #[must_use]
    pub fn backward_bitset(&self) -> &[u32] {
        &self.backward
    }
}

/// Compute per-pivot forward + backward reach bitsets for the
/// strongly-connected-component decomposition primitive
/// (`vyre_primitives::graph::scc_decompose::cpu_ref`).
///
/// Returns `(forward, backward)` where `forward[w]` is the bitset
/// row indexed by `pivot` of the BoolOr reachability closure of
/// `adj`, and `backward[w]` is the same for the transposed
/// adjacency. The bitsets are packed 32-bits-per-u32, length
/// `bitset_words(n)`. Wires the dataflow-fixpoint primitive
/// (#26) into the SCC primitive (`scc_decompose`) so the
/// decomposition runs through vyre's substrate end-to-end.
///
/// # Panics
///
/// Panics if `pivot >= n` or `adj.len() != n*n`.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn forward_backward_bitsets_for_pivot(adj: &[u32], pivot: u32, n: u32) -> (Vec<u32>, Vec<u32>) {
    let mut scratch = DataflowFixpointScratch::default();
    forward_backward_bitsets_for_pivot_into(adj, pivot, n, &mut scratch);
    (scratch.forward, scratch.backward)
}

/// Compute per-pivot forward + backward reach bitsets into caller-owned scratch.
///
/// Results are written to `scratch.forward` and `scratch.backward`.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn forward_backward_bitsets_for_pivot_into(
    adj: &[u32],
    pivot: u32,
    n: u32,
    scratch: &mut DataflowFixpointScratch,
) {
    use crate::telemetry::observability::{bump, dataflow_fixpoint_calls};
    bump(&dataflow_fixpoint_calls);
    assert!(
        n > 0,
        "Fix: forward_backward_bitsets_for_pivot requires n > 0."
    );
    assert!(pivot < n, "Fix: pivot index must be < n.");
    let n_us = n as usize;
    assert_eq!(
        adj.len(),
        n_us * n_us,
        "Fix: adjacency must contain n*n entries."
    );

    let words = ((n + 31) / 32) as usize;

    reachability_closure_into(
        adj,
        n,
        n,
        &mut scratch.fwd_closure,
        &mut scratch.bwd_closure,
    );
    scratch.transpose.clear();
    scratch.transpose.resize(n_us * n_us, 0);
    for i in 0..n_us {
        for j in 0..n_us {
            scratch.transpose[j * n_us + i] = adj[i * n_us + j];
        }
    }
    reachability_closure_into(
        &scratch.transpose,
        n,
        n,
        &mut scratch.bwd_closure,
        &mut scratch.next_components,
    );

    scratch.forward.resize(words, 0);
    scratch.backward.resize(words, 0);
    write_pivot_bitsets(
        &scratch.fwd_closure,
        &scratch.bwd_closure,
        pivot,
        n_us,
        &mut scratch.forward,
        &mut scratch.backward,
    );
}

pub(super) fn write_pivot_bitsets(
    fwd_closure: &[u32],
    bwd_closure: &[u32],
    pivot: u32,
    n_us: usize,
    forward: &mut [u32],
    backward: &mut [u32],
) {
    forward.fill(0);
    backward.fill(0);
    let pivot_us = pivot as usize;
    // Pivot reaches itself.
    let pivot_word = pivot_us / 32;
    let pivot_bit = 1u32 << (pivot_us % 32);
    forward[pivot_word] |= pivot_bit;
    backward[pivot_word] |= pivot_bit;
    for v in 0..n_us {
        if fwd_closure[pivot_us * n_us + v] != 0 {
            forward[v / 32] |= 1u32 << (v % 32);
        }
        if bwd_closure[pivot_us * n_us + v] != 0 {
            backward[v / 32] |= 1u32 << (v % 32);
        }
    }
}

/// Drive `vyre_primitives::graph::scc_decompose::cpu_ref` end-to-end
/// over an `n×n` adjacency: pick pivots in descending unassigned
/// order and stamp every node in `forward(p) ∩ backward(p)` with `p`.
/// Returns the per-node component-id vector. Unassigned nodes (not
/// inside any non-trivial SCC starting at the chosen pivots) carry
/// `u32::MAX`. Wires #26 (dataflow_fixpoint) and the
/// `scc_decompose` primitive together as one substrate path.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn scc_components_via_substrate(adj: &[u32], n: u32) -> Vec<u32> {
    let mut components = Vec::new();
    let mut scratch = DataflowFixpointScratch::default();
    reference_scc_components_via_substrate_into(adj, n, &mut components, &mut scratch);
    components
}

/// Drive SCC decomposition into caller-owned output and scratch buffers.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_scc_components_via_substrate_into(
    adj: &[u32],
    n: u32,
    components: &mut Vec<u32>,
    scratch: &mut DataflowFixpointScratch,
) {
    use crate::telemetry::observability::{bump, dataflow_fixpoint_calls};
    bump(&dataflow_fixpoint_calls);
    components.clear();
    if n == 0 {
        return;
    }
    let n_us = n as usize;
    components.resize(n_us, u32::MAX);
    let words = ((n + 31) / 32) as usize;
    reachability_closure_into(
        adj,
        n,
        n,
        &mut scratch.fwd_closure,
        &mut scratch.bwd_closure,
    );
    scratch.transpose.clear();
    scratch.transpose.resize(n_us * n_us, 0);
    for i in 0..n_us {
        for j in 0..n_us {
            scratch.transpose[j * n_us + i] = adj[i * n_us + j];
        }
    }
    reachability_closure_into(
        &scratch.transpose,
        n,
        n,
        &mut scratch.bwd_closure,
        &mut scratch.next_components,
    );
    scratch.forward.resize(words, 0);
    scratch.backward.resize(words, 0);
    scratch.next_components.clear();
    reserve_vec_capacity_or_panic(
        &mut scratch.next_components,
        n_us,
        "SCC component staging scratch",
    );
    for pivot in 0..n {
        if components[pivot as usize] != u32::MAX {
            continue;
        }
        write_pivot_bitsets(
            &scratch.fwd_closure,
            &scratch.bwd_closure,
            pivot,
            n_us,
            &mut scratch.forward,
            &mut scratch.backward,
        );
        vyre_primitives::graph::scc_decompose::cpu_ref_into(
            n,
            &scratch.forward,
            &scratch.backward,
            components,
            pivot,
            &mut scratch.next_components,
        );
        std::mem::swap(components, &mut scratch.next_components);
    }
}

#[cfg(test)]
mod tests {
    use super::super::DataflowFixpointScratch;
    use super::{
        forward_backward_bitsets_for_pivot, forward_backward_bitsets_for_pivot_into,
        reference_scc_components_via_substrate_into, scc_components_via_substrate,
    };

    #[test]
    fn fb_bitsets_chain_pivot_zero() {
        // 0 → 1 → 2. From pivot 0: forward = {0,1,2}, backward = {0}.
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let (fwd, bwd) = forward_backward_bitsets_for_pivot(&adj, 0, 3);
        assert_eq!(fwd, vec![0b111]);
        assert_eq!(bwd, vec![0b001]);
    }

    #[test]
    fn fb_bitsets_chain_pivot_two() {
        // 0 → 1 → 2. From pivot 2: forward = {2}, backward = {0,1,2}.
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let (fwd, bwd) = forward_backward_bitsets_for_pivot(&adj, 2, 3);
        assert_eq!(fwd, vec![0b100]);
        assert_eq!(bwd, vec![0b111]);
    }

    #[test]
    fn fb_bitsets_into_reuses_capacity_and_matches_owned() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mut scratch = DataflowFixpointScratch::default();
        forward_backward_bitsets_for_pivot_into(&adj, 2, 3, &mut scratch);
        let fwd_capacity = scratch.forward.capacity();
        let bwd_capacity = scratch.backward.capacity();
        assert_eq!(scratch.forward_bitset(), &[0b100]);
        assert_eq!(scratch.backward_bitset(), &[0b111]);

        forward_backward_bitsets_for_pivot_into(&adj, 0, 3, &mut scratch);
        assert_eq!(scratch.forward.capacity(), fwd_capacity);
        assert_eq!(scratch.backward.capacity(), bwd_capacity);
        assert_eq!(scratch.forward_bitset(), &[0b111]);
        assert_eq!(scratch.backward_bitset(), &[0b001]);
    }

    #[test]
    fn scc_components_chain_each_node_singleton() {
        // 0 → 1 → 2 (DAG). Every SCC is a singleton.
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let comps = scc_components_via_substrate(&adj, 3);
        // Each node stamped by itself (first pivot wins).
        assert_eq!(comps, vec![0, 1, 2]);
    }

    #[test]
    fn scc_components_two_cycle_collapses_to_first_pivot() {
        // 0 → 1, 1 → 0. {0,1} is one SCC. First pivot 0 stamps both.
        let adj = vec![0, 1, 1, 0];
        let comps = scc_components_via_substrate(&adj, 2);
        assert_eq!(comps, vec![0, 0]);
    }

    #[test]
    fn scc_components_into_reuses_output_and_matches_owned() {
        let adj = vec![0, 1, 1, 0];
        let mut comps = Vec::new();
        let mut scratch = DataflowFixpointScratch::default();
        reference_scc_components_via_substrate_into(&adj, 2, &mut comps, &mut scratch);
        let comps_capacity = comps.capacity();
        let scratch_capacity = scratch.next_components.capacity();
        assert_eq!(comps, vec![0, 0]);

        reference_scc_components_via_substrate_into(&adj, 2, &mut comps, &mut scratch);
        assert_eq!(comps.capacity(), comps_capacity);
        assert_eq!(scratch.next_components.capacity(), scratch_capacity);
        assert_eq!(comps, scc_components_via_substrate(&adj, 2));
    }

    /// Closure-bar: the substrate-driven SCC must agree with running
    /// `scc_decompose::cpu_ref` directly with manually-prepared
    /// forward/backward bitsets. Asserts the wiring doesn't drift.
    #[test]
    fn scc_components_match_direct_primitive_call() {
        // 0 → 1 → 2 → 0 (one big cycle), 3 → 4 separate.
        let adj = vec![
            0, 1, 0, 0, 0, // 0 -> 1
            0, 0, 1, 0, 0, // 1 -> 2
            1, 0, 0, 0, 0, // 2 -> 0
            0, 0, 0, 0, 1, // 3 -> 4
            0, 0, 0, 0, 0, // 4
        ];
        let via_substrate = scc_components_via_substrate(&adj, 5);

        // Manual replay: pivot 0 stamps {0,1,2}; pivot 3 stamps {3};
        // pivot 4 stamps {4}.
        let mut manual = vec![u32::MAX; 5];
        for pivot in [0u32, 3, 4] {
            let (fwd, bwd) = forward_backward_bitsets_for_pivot(&adj, pivot, 5);
            manual = vyre_primitives::graph::scc_decompose::cpu_ref(5, &fwd, &bwd, &manual, pivot);
        }
        assert_eq!(via_substrate, manual);
        // The cycle members all carry pivot 0.
        assert_eq!(via_substrate[0..3], [0, 0, 0]);
        // Singletons keep their own pivot id.
        assert_eq!(via_substrate[3], 3);
        assert_eq!(via_substrate[4], 4);
    }

    /// Adversarial: a fully disconnected graph (no edges) must yield
    /// `[0, 1, 2, ..., n-1]` because every pivot stamps only itself.
    #[test]
    fn scc_components_no_edges_each_pivot_stamps_only_itself() {
        let n = 4;
        let adj = vec![0u32; (n * n) as usize];
        let comps = scc_components_via_substrate(&adj, n);
        assert_eq!(comps, vec![0, 1, 2, 3]);
    }

    /// Adversarial: a self-loop on a node must NOT pull other nodes
    /// into its SCC. A common bug is to over-eagerly mark every node
    /// reached via the closure's reflexive-transitive interpretation.
    #[test]
    fn scc_components_self_loop_does_not_merge_distinct_components() {
        // 0 -> 0 (self-loop), 1 isolated, 2 isolated.
        let adj = vec![1, 0, 0, 0, 0, 0, 0, 0, 0];
        let comps = scc_components_via_substrate(&adj, 3);
        assert_eq!(comps, vec![0, 1, 2]);
    }
}
