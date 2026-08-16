//! Parity test: GPU persistent-BFS dispatch matches the reference oracle.
//!
//! Drives the new `vyre_libs::graph::dispatch::persistent_bfs::bfs_expand_via`
//! GPU dispatch path against the existing reference oracle on real CUDA
//! hardware. Asserts identical (frontier_out, changed) on a battery
//! of graph shapes and allow_mask values.

#![cfg(test)]

use vyre_libs::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
mod common;

use common::{with_live_backend, CudaProgramDispatcher};
use vyre_driver_cuda::CudaProgramDispatcher as CudaResidentProgramDispatcher;
use vyre_libs::graph::dispatch::persistent_bfs::{
    bfs_expand as reference_bfs_expand, bfs_expand_resident_graph_batch_with_scratch_into,
    bfs_expand_resident_graph_with_scratch_into, bfs_expand_via, try_bfs_expand_converged,
    upload_resident_bfs_graph, PersistentBfsPlanCacheSnapshot, PersistentBfsResidentScratch,
};

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
    with_live_backend("cuda_bfs_expand_via_matches_reference_chain", |backend| {
        let dispatcher = CudaProgramDispatcher { backend };
        let (n, off, tgt, msk) = linear_chain(8);
        let seed = vec![0b0000_0001u32]; // node 0 only
        let (gpu_out, gpu_changed, gpu_converged) = bfs_expand_via(
            &dispatcher,
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
        let (reference_out, reference) = try_bfs_expand_converged(
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
        .expect("reference persistent BFS convergence");
        assert_eq!(
            gpu_out, reference_out,
            "frontier_out diverged on chain n={n}; gpu={gpu_out:?} reference={reference_out:?}"
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
    });
}

#[test]
fn cuda_bfs_expand_via_respects_allow_mask() {
    // A graph with mixed edge kinds. allow_mask filters which edges
    // to follow.
    with_live_backend("cuda_bfs_expand_via_respects_allow_mask", |backend| {
        let dispatcher = CudaProgramDispatcher { backend };
        // 0 -[k=1]-> 1, 0 -[k=2]-> 2, 1 -[k=1]-> 3
        let n = 4;
        let off = vec![0u32, 2, 3, 3, 3];
        let tgt = vec![1u32, 2, 3];
        let msk = vec![1u32, 2, 1];
        let seed = vec![0b0001u32];

        // allow_mask = 1 -> only k=1 edges followed: 0->1, 1->3. Reach {0,1,3}.
        let (gpu_out, _, _) = bfs_expand_via(
            &dispatcher,
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
            &dispatcher,
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
            &dispatcher,
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
    });
}

#[test]
fn cuda_bfs_expand_via_saturated_seed_reports_no_change() {
    with_live_backend(
        "cuda_bfs_expand_via_saturated_seed_reports_no_change",
        |backend| {
            let dispatcher = CudaProgramDispatcher { backend };
            let (n, off, tgt, msk) = linear_chain(4);
            // Seed = full chain already.
            let seed = vec![0b1111u32];
            let (_gpu_out, gpu_changed, gpu_converged) = bfs_expand_via(
                &dispatcher,
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
            let (_reference_out, reference) = try_bfs_expand_converged(
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
            .expect("reference persistent BFS convergence");
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

#[test]
fn cuda_resident_bfs_graph_matches_reference_across_repeated_queries() {
    with_live_backend(
        "cuda_resident_bfs_graph_matches_reference_across_repeated_queries",
        |backend| {
            let dispatcher = CudaResidentProgramDispatcher::new(backend);
            let (n, off, tgt, msk) = linear_chain(8);
            let graph = upload_resident_bfs_graph(&dispatcher, n, &off, &tgt, &msk)
                .expect("resident graph upload");
            let mut scratch = PersistentBfsResidentScratch::default();
            let mut frontier = Vec::with_capacity(1);
            let frontier_ptr = frontier.as_ptr();

            for seed in [0b0000_0001u32, 0b0000_0011u32] {
                let seed_words = [seed];
                let (changed, converged) = bfs_expand_resident_graph_with_scratch_into(
                    &dispatcher,
                    &graph,
                    &seed_words,
                    0xFFFF_FFFF,
                    n,
                    &mut scratch,
                    &mut frontier,
                )
                .expect("resident graph BFS query");
                let (reference_out, reference) = try_bfs_expand_converged(
                    CsrClosureInputs::allow_all(
                        CsrGraphView {
                            node_count: n,
                            edge_offsets: &off,
                            edge_targets: &tgt,
                            edge_kind_mask: &msk,
                        },
                        n,
                    ),
                    &seed_words,
                )
                .expect("reference persistent BFS convergence");
                assert_eq!(frontier, reference_out);
                assert_eq!(changed, reference.changed);
                assert_eq!(converged, u32::from(reference.converged));
                assert_eq!(
                    frontier.as_ptr(),
                    frontier_ptr,
                    "caller-owned frontier Vec must be reused across resident graph queries"
                );
            }
            assert_eq!(
                scratch.plan_cache_snapshot(),
                PersistentBfsPlanCacheSnapshot {
                    entries: 1,
                    hits: 1,
                    misses: 1,
                },
                "CUDA resident BFS must reuse the cached single-query plan across repeated graph queries"
            );

            scratch.free(&dispatcher).expect("resident scratch free");
            graph.free(&dispatcher).expect("resident graph free");
        },
    );
}

#[test]
fn cuda_resident_bfs_graph_batch_matches_reference() {
    with_live_backend(
        "cuda_resident_bfs_graph_batch_matches_reference",
        |backend| {
            let dispatcher = CudaResidentProgramDispatcher::new(backend);
            let (n, off, tgt, msk) = linear_chain(8);
            let graph = upload_resident_bfs_graph(&dispatcher, n, &off, &tgt, &msk)
                .expect("resident graph upload");
            let mut scratch = PersistentBfsResidentScratch::default();
            let mut frontier_outputs = Vec::with_capacity(3);
            let frontier_ptr = frontier_outputs.as_ptr();
            let mut changed_outputs = Vec::with_capacity(3);
            let changed_ptr = changed_outputs.as_ptr();
            let mut converged_outputs = Vec::with_capacity(3);
            let converged_ptr = converged_outputs.as_ptr();
            let seeds = [0b0000_0001u32, 0b0000_0011u32, 0b0000_1111u32];

            bfs_expand_resident_graph_batch_with_scratch_into(
                &dispatcher,
                &graph,
                &seeds,
                seeds.len(),
                0xFFFF_FFFF,
                n,
                &mut scratch,
                &mut frontier_outputs,
                &mut changed_outputs,
                &mut converged_outputs,
            )
            .expect("resident graph batch BFS query");

            let mut expected_frontiers = Vec::with_capacity(seeds.len());
            let mut expected_changed = Vec::with_capacity(seeds.len());
            for seed in seeds {
                let (frontier, changed) = reference_bfs_expand(
                    CsrClosureInputs::allow_all(
                        CsrGraphView {
                            node_count: n,
                            edge_offsets: &off,
                            edge_targets: &tgt,
                            edge_kind_mask: &msk,
                        },
                        n,
                    ),
                    &[seed],
                );
                expected_frontiers.extend_from_slice(&frontier);
                expected_changed.push(changed);
            }

            assert_eq!(frontier_outputs, expected_frontiers);
            assert_eq!(changed_outputs, expected_changed);
            // max_iters == n bounds the n-node chain diameter, so every query
            // reaches a fixpoint within budget: converged is 1 for all queries.
            assert_eq!(converged_outputs, vec![1u32; seeds.len()]);
            assert_eq!(frontier_outputs.as_ptr(), frontier_ptr);
            assert_eq!(changed_outputs.as_ptr(), changed_ptr);
            assert_eq!(converged_outputs.as_ptr(), converged_ptr);

            scratch.free(&dispatcher).expect("resident scratch free");
            graph.free(&dispatcher).expect("resident graph free");
        },
    );
}
