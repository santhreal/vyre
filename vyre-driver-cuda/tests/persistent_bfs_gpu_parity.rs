//! Parity test: GPU persistent-BFS dispatch matches the reference oracle.
//!
//! Drives the new `vyre_libs::graph::dispatch::persistent_bfs::bfs_expand_via`
//! GPU dispatch path against the existing reference oracle on real CUDA
//! hardware. Asserts identical (frontier_out, changed) on a battery
//! of graph shapes and allow_mask values.

#![cfg(all(test, feature = "device-tests"))]

use vyre_libs::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
mod harness;

use harness::with_cuda_optimizer_dispatcher;
use vyre_libs::graph::dispatch::persistent_bfs::bfs_expand_via;
use vyre_reference::composition_witness::{
    csr_persistent_closure_detailed_witness, CsrPersistentClosureWitness,
};

fn reference_bfs_expand_detailed(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
) -> CsrPersistentClosureWitness {
    csr_persistent_closure_detailed_witness(
        inputs.graph.node_count,
        inputs.graph.edge_offsets,
        inputs.graph.edge_targets,
        inputs.graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
    )
}

fn reference_bfs_expand(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> (Vec<u32>, u32) {
    let detailed = reference_bfs_expand_detailed(inputs, seed);
    (detailed.frontier, detailed.changed)
}

fn linear_chain(n: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    // 0 -> 1 -> 2 -> ... -> n-1
    let mut offsets = Vec::with_capacity((n + 1) as usize);
    let mut targets = Vec::with_capacity((n.saturating_sub(1)) as usize);
    let mut masks = Vec::with_capacity((n.saturating_sub(1)) as usize);
    let mut e = 0u32;
    for i in 0..n {
        offsets.push(e);
        if i + 1 < n {
            targets.push(i + 1);
            masks.push(0b0001);
            e += 1;
        }
    }
    offsets.push(e);
    (n, offsets, targets, masks)
}

#[test]
fn cuda_bfs_expand_via_matches_reference_chain() {
    with_cuda_optimizer_dispatcher(
        "cuda_bfs_expand_via_matches_reference_chain",
        |dispatcher, policy| {
            let (n, off, tgt, msk) = linear_chain(8);
            let seed = vec![0b0000_0001u32]; // node 0 only
            let (gpu_out, gpu_changed, gpu_converged) = bfs_expand_via(
                dispatcher,
                policy,
                CsrClosureInputs::allow_all(
                    CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    n,
                ),
                &seed,
            )
            .expect("GPU bfs_expand_via dispatch");
            let reference = reference_bfs_expand_detailed(
                CsrClosureInputs::allow_all(
                    CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    n,
                ),
                &seed,
            );
            assert_eq!(
                gpu_out, reference.frontier,
                "frontier_out diverged on chain n={n}; gpu={gpu_out:?} reference={:?}",
                reference.frontier
            );
            assert_eq!(
                gpu_changed, reference.changed,
                "changed-flag diverged on chain n={n}"
            );
            assert_eq!(
                gpu_converged,
                u32::from(reference.converged),
                "converged word diverged on chain n={n}"
            );
        },
    );
}

#[test]
fn cuda_bfs_expand_via_respects_allow_mask() {
    // A graph with mixed edge kinds. allow_mask filters which edges
    // to follow.
    with_cuda_optimizer_dispatcher(
        "cuda_bfs_expand_via_respects_allow_mask",
        |dispatcher, policy| {
            // 0 -[k=1]-> 1, 0 -[k=2]-> 2, 1 -[k=1]-> 3
            let n = 4;
            let off = vec![0u32, 2, 3, 3, 3];
            let tgt = vec![1u32, 2, 3];
            let msk = vec![1u32, 2, 1];
            let seed = vec![0b0001u32];

            // allow_mask = 1 -> only k=1 edges followed: 0->1, 1->3. Reach {0,1,3}.
            let (gpu_out, _, _) = bfs_expand_via(
                dispatcher,
                policy,
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0001,
                    max_iters: n,
                },
                &seed,
            )
            .expect("dispatch");
            let (reference_out, _) = reference_bfs_expand(
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0001,
                    max_iters: n,
                },
                &seed,
            );
            assert_eq!(gpu_out, reference_out, "allow_mask=1 divergence");

            // allow_mask = 2 -> only k=2 edges: 0->2. Reach {0,2}.
            let (gpu_out, _, _) = bfs_expand_via(
                dispatcher,
                policy,
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0010,
                    max_iters: n,
                },
                &seed,
            )
            .expect("dispatch");
            let (reference_out, _) = reference_bfs_expand(
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0010,
                    max_iters: n,
                },
                &seed,
            );
            assert_eq!(gpu_out, reference_out, "allow_mask=2 divergence");

            // allow_mask = 3 -> both kinds: 0->1, 0->2, 1->3. Reach {0,1,2,3}.
            let (gpu_out, _, _) = bfs_expand_via(
                dispatcher,
                policy,
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0011,
                    max_iters: n,
                },
                &seed,
            )
            .expect("dispatch");
            let (reference_out, _) = reference_bfs_expand(
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    allow_mask: 0b0011,
                    max_iters: n,
                },
                &seed,
            );
            assert_eq!(gpu_out, reference_out, "allow_mask=3 divergence");
        },
    );
}

#[test]
fn cuda_bfs_expand_via_saturated_seed_reports_no_change() {
    with_cuda_optimizer_dispatcher(
        "cuda_bfs_expand_via_saturated_seed_reports_no_change",
        |dispatcher, policy| {
            let (n, off, tgt, msk) = linear_chain(4);
            // Seed = full chain already.
            let seed = vec![0b1111u32];
            let (_gpu_out, gpu_changed, gpu_converged) = bfs_expand_via(
                dispatcher,
                policy,
                CsrClosureInputs::allow_all(
                    CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    n,
                ),
                &seed,
            )
            .expect("dispatch");
            let reference = reference_bfs_expand_detailed(
                CsrClosureInputs::allow_all(
                    CsrGraphView {
                        node_count: n,
                        edge_offsets: &off,
                        edge_targets: &tgt,
                        edge_kind_mask: &msk,
                    },
                    n,
                ),
                &seed,
            );
            assert_eq!(gpu_changed, reference.changed);
            assert_eq!(gpu_changed, 0);
            assert_eq!(gpu_converged, u32::from(reference.converged));
            assert_eq!(
                gpu_converged, 1,
                "a saturated seed is an immediate fixpoint and must report converged=1"
            );
        },
    );
}
