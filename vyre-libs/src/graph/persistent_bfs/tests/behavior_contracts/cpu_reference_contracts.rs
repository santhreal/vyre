use super::*;
use crate::graph::csr_closure_inputs::graphs::{CHAIN_4, DIAMOND_4};
use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

#[test]
fn persistent_bfs_reaches_closure() {
    let (frontier, changed) = cpu_ref(CsrClosureInputs::allow_all(DIAMOND_4, 4), &[0b0001]);
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(changed, 1);
}

#[test]
fn cpu_ref_into_reuses_frontier_storage() {
    let mut frontier = Vec::with_capacity(8);
    let changed = cpu_ref_into(
        CsrClosureInputs::allow_all(CHAIN_4, 8),
        &[0b0001],
        &mut frontier,
    );
    let capacity = frontier.capacity();
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(changed, 1);

    let changed = cpu_ref_into(CsrClosureInputs::allow_all(CHAIN_4, 8), &[0], &mut frontier);
    assert_eq!(frontier.capacity(), capacity);
    assert_eq!(frontier, vec![0]);
    assert_eq!(changed, 0);
}

#[test]
fn try_cpu_ref_into_with_scratch_reuses_step_storage_and_clears_stale_state() {
    let mut frontier = Vec::with_capacity(8);
    let mut step = Vec::with_capacity(8);
    step.extend_from_slice(&[0xDEAD_BEEF, 0xCAFE_BABE, 0xBADC_0FFE]);
    let mut scratch = PersistentBfsCpuScratch { step };
    let frontier_capacity = frontier.capacity();
    let step_capacity = scratch.step.capacity();

    let changed = try_cpu_ref_into_with_scratch(
        CsrClosureInputs::allow_all(CHAIN_4, 8),
        &[0b0001],
        &mut frontier,
        &mut scratch,
    )
    .expect("Fix: valid persistent BFS chain must run with reusable scratch.");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(changed, 1);
    assert_eq!(frontier.capacity(), frontier_capacity);
    assert_eq!(scratch.step.capacity(), step_capacity);
    assert_eq!(scratch.step.len(), 1);

    let changed = try_cpu_ref_into_with_scratch(
        CsrClosureInputs::allow_all(CHAIN_4, 8),
        &[0],
        &mut frontier,
        &mut scratch,
    )
    .expect("Fix: second persistent BFS run must clear stale step bits.");
    assert_eq!(frontier, vec![0]);
    assert_eq!(changed, 0);
    assert_eq!(frontier.capacity(), frontier_capacity);
    assert_eq!(scratch.step.capacity(), step_capacity);
    assert_eq!(
        scratch.step,
        vec![0],
        "Fix: reusable step scratch must be resized to live words and cleared by traversal."
    );
}

#[test]
fn try_cpu_ref_into_rejects_bad_input_without_clobbering_frontier() {
    let mut frontier = vec![0xDEAD_BEEF];
    let capacity = frontier.capacity();

    let err = try_cpu_ref_into(
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 4,
                edge_offsets: &[0, 1, 2],
                edge_targets: &[1, 2, 3],
                edge_kind_mask: &[1, 1, 1],
            },
            8,
        ),
        &[0b0001],
        &mut frontier,
    )
    .expect_err("Fix: fallible persistent BFS oracle must reject malformed CSR inputs");

    assert!(err.contains("CSR offsets"));
    assert_eq!(frontier, vec![0xDEAD_BEEF]);
    assert_eq!(frontier.capacity(), capacity);
}

#[test]
fn try_cpu_ref_into_with_scratch_rejects_bad_input_without_clobbering_storage() {
    let mut frontier = vec![0xDEAD_BEEF];
    let mut scratch = PersistentBfsCpuScratch {
        step: vec![0xCAFE_BABE, 0xBADC_0FFE],
    };

    let err = try_cpu_ref_into_with_scratch(
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 4,
                edge_offsets: &[0, 1, 2],
                edge_targets: &[1, 2, 3],
                edge_kind_mask: &[1, 1, 1],
            },
            8,
        ),
        &[0b0001],
        &mut frontier,
        &mut scratch,
    )
    .expect_err("Fix: fallible persistent BFS oracle must reject malformed CSR inputs.");

    assert!(err.contains("CSR offsets"));
    assert_eq!(
        frontier,
        vec![0xDEAD_BEEF],
        "Fix: validation failures must not clobber reusable frontier output."
    );
    assert_eq!(
        scratch.step,
        vec![0xCAFE_BABE, 0xBADC_0FFE],
        "Fix: validation failures must not clear reusable step scratch."
    );
}

#[test]
fn fallible_cpu_ref_matches_compatibility_oracle_on_generated_chains() {
    for node_count in [0_u32, 1, 2, 3, 31, 32, 33, 64, 65, 257] {
        let mut offsets = Vec::with_capacity(node_count as usize + 1);
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        offsets.push(0);
        for node in 0..node_count {
            if node + 1 < node_count {
                targets.push(node + 1);
                masks.push(1);
            }
            offsets.push(targets.len() as u32);
        }
        let words = bitset_words(node_count) as usize;
        let mut seed = vec![0; words];
        if node_count != 0 {
            seed[0] = 1;
        }

        let expected = cpu_ref(
            CsrClosureInputs::allow_all(
                CsrGraphView {
                    node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                node_count.saturating_add(1),
            ),
            &seed,
        );
        let actual = try_cpu_ref(
            CsrClosureInputs::allow_all(
                CsrGraphView {
                    node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                node_count.saturating_add(1),
            ),
            &seed,
        )
        .expect("Fix: generated valid persistent BFS chain should run fallibly");
        assert_eq!(actual, expected, "node_count={node_count}");
    }
}

#[test]
fn generated_try_cpu_ref_into_with_scratch_matches_allocating_reference() {
    let mut frontier = Vec::new();
    let mut scratch = PersistentBfsCpuScratch::new();

    for case in 0..1024usize {
        let node_count = (case % 67) as u32;
        let mut offsets = Vec::with_capacity(node_count as usize + 1);
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        offsets.push(0);
        for src in 0..node_count {
            for dst in 0..node_count {
                let mixed = case
                    .wrapping_mul(43)
                    .wrapping_add((src as usize).wrapping_mul(17))
                    .wrapping_add((dst as usize).wrapping_mul(29));
                if src != dst && (mixed % 23 == 0 || (case % 19 == 0 && dst == src + 1)) {
                    targets.push(dst);
                    masks.push(if mixed % 2 == 0 { 1 } else { 2 });
                }
            }
            offsets.push(targets.len() as u32);
        }

        let words = bitset_words(node_count) as usize;
        let mut seed = vec![0; words];
        for node in 0..node_count {
            let mixed = case
                .wrapping_mul(11)
                .wrapping_add((node as usize).wrapping_mul(7));
            if mixed % 13 == 0 || (node == 0 && node_count != 0) {
                seed[(node / 32) as usize] |= 1u32 << (node % 32);
            }
        }
        let allow_mask = if case % 3 == 0 { 1 } else { 0xFFFF_FFFF };
        let max_iters = (case % 11) as u32;
        let expected = try_cpu_ref(
            CsrClosureInputs {
                graph: CsrGraphView {
                    node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                allow_mask,
                max_iters,
            },
            &seed,
        )
        .expect("Fix: generated persistent BFS graph must be valid for allocating oracle.");
        let changed = try_cpu_ref_into_with_scratch(
            CsrClosureInputs {
                graph: CsrGraphView {
                    node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                allow_mask,
                max_iters,
            },
            &seed,
            &mut frontier,
            &mut scratch,
        )
        .expect("Fix: generated persistent BFS graph must run with reusable scratch.");
        assert_eq!(
            (frontier.clone(), changed),
            expected,
            "Fix: scratch-backed persistent BFS diverged from allocating oracle at case {case}."
        );
    }
}

// The 4-node chain 0->1->2->3, seeded at node 0, has true closure 0b1111
// reached after 3 growth steps; a 4th step adds nothing and proves the
// fixpoint. The offsets/targets below encode exactly that chain.
const CHAIN4_OFFSETS: &[u32] = &[0, 1, 2, 3, 3];
const CHAIN4_TARGETS: &[u32] = &[1, 2, 3];
const CHAIN4_MASKS: &[u32] = &[1, 1, 1];

#[test]
fn converged_reports_false_and_partial_frontier_when_max_iters_below_diameter() {
    // Two steps grow {0}->{0,1}->{0,1,2}; the closure is still growing, so the
    // loop exhausts max_iters without proving a fixpoint.
    let (frontier, outcome) =
        try_cpu_ref_converged(CsrClosureInputs::allow_all(CHAIN_4, 2), &[0b0001])
            .expect("Fix: valid chain must run under the convergence-reporting oracle.");
    assert_eq!(frontier, vec![0b0111]);
    assert_eq!(
        outcome,
        PersistentBfsConvergence {
            changed: 1,
            converged: false,
            stop_iter: 2,
        },
        "Fix: exhausting max_iters while still growing must report converged=false with stop_iter==max_iters, not a silent partial closure."
    );
}

#[test]
fn converged_reports_true_at_true_stop_iter_when_max_iters_above_diameter() {
    // Three growth steps reach 0b1111; the 4th step adds nothing and proves the
    // fixpoint, so the loop stops at iteration 4 well within the budget of 8.
    let (frontier, outcome) =
        try_cpu_ref_converged(CsrClosureInputs::allow_all(CHAIN_4, 8), &[0b0001])
            .expect("Fix: valid chain must run under the convergence-reporting oracle.");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(
        outcome,
        PersistentBfsConvergence {
            changed: 1,
            converged: true,
            stop_iter: 4,
        },
        "Fix: a reached fixpoint must report converged=true at the confirming step, not max_iters."
    );
}

#[test]
fn converged_is_false_when_full_set_is_reached_only_on_the_last_allowed_step() {
    // Exactly 3 iterations reach 0b1111, but the loop never runs the 4th
    // confirming step, so it cannot prove the fixpoint: converged stays false
    // even though the frontier is already complete.
    let (frontier, outcome) =
        try_cpu_ref_converged(CsrClosureInputs::allow_all(CHAIN_4, 3), &[0b0001])
            .expect("Fix: valid chain must run under the convergence-reporting oracle.");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(
        outcome,
        PersistentBfsConvergence {
            changed: 1,
            converged: false,
            stop_iter: 3,
        },
        "Fix: reaching the full set on the last allowed step is not a convergence proof; a confirming no-growth step is required."
    );
}

#[test]
fn converged_reports_true_with_no_change_when_seed_is_already_a_fixpoint() {
    // A seed that already contains the whole closure never grows: the first
    // step adds nothing, so the run converges immediately with changed=0.
    let (frontier, outcome) =
        try_cpu_ref_converged(CsrClosureInputs::allow_all(CHAIN_4, 8), &[0b1111])
            .expect("Fix: valid chain must run under the convergence-reporting oracle.");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(
        outcome,
        PersistentBfsConvergence {
            changed: 0,
            converged: true,
            stop_iter: 1,
        },
        "Fix: an already-complete seed must converge on the first step with changed=0."
    );
}

#[test]
fn converged_changed_flag_matches_sticky_oracle_on_generated_chains() {
    // The convergence-reporting oracle must agree with the sticky-changed
    // oracle on both the frontier and the changed flag across a spread of
    // sizes and budgets, and its non-convergence signal must be consistent:
    // an exhausted run (converged=false) always stops exactly at max_iters.
    for node_count in [0_u32, 1, 2, 3, 4, 7, 31, 32, 33, 64, 65] {
        let mut offsets = Vec::with_capacity(node_count as usize + 1);
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        offsets.push(0);
        for node in 0..node_count {
            if node + 1 < node_count {
                targets.push(node + 1);
                masks.push(1);
            }
            offsets.push(targets.len() as u32);
        }
        let words = bitset_words(node_count) as usize;
        let mut seed = vec![0; words];
        if node_count != 0 {
            seed[0] = 1;
        }
        for max_iters in [0_u32, 1, 2, node_count, node_count.saturating_add(2)] {
            let (sticky_frontier, sticky_changed) = try_cpu_ref(
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count,
                        edge_offsets: &offsets,
                        edge_targets: &targets,
                        edge_kind_mask: &masks,
                    },
                    allow_mask: 0xFFFF_FFFF,
                    max_iters,
                },
                &seed,
            )
            .expect("Fix: generated valid chain must run under the sticky oracle.");
            let (converged_frontier, outcome) = try_cpu_ref_converged(
                CsrClosureInputs {
                    graph: CsrGraphView {
                        node_count,
                        edge_offsets: &offsets,
                        edge_targets: &targets,
                        edge_kind_mask: &masks,
                    },
                    allow_mask: 0xFFFF_FFFF,
                    max_iters,
                },
                &seed,
            )
            .expect("Fix: generated valid chain must run under the convergence oracle.");
            assert_eq!(
                converged_frontier, sticky_frontier,
                "node_count={node_count} max_iters={max_iters}: frontiers must match"
            );
            assert_eq!(
                outcome.changed, sticky_changed,
                "node_count={node_count} max_iters={max_iters}: changed flags must match"
            );
            if !outcome.converged {
                assert_eq!(
                    outcome.stop_iter, max_iters,
                    "node_count={node_count} max_iters={max_iters}: a non-converged run must stop exactly at max_iters"
                );
            } else {
                assert!(
                    outcome.stop_iter <= max_iters,
                    "node_count={node_count} max_iters={max_iters}: a converged run cannot exceed its budget"
                );
            }
        }
    }
}
