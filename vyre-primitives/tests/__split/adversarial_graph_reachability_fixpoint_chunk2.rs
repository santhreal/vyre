#[test]
fn csr_forward_or_changed_empty_frontier() {
    let (frontier, changed) =
        csr_cpu_ref(4, &[0, 2, 3, 4, 4], &[1, 2, 3, 3], &[1, 1, 1, 1], &[0], 1);
    assert_eq!(frontier, vec![0]);
    assert_eq!(changed, 0);
}

#[test]
fn csr_forward_or_changed_edge_mask_blocks() {
    let (frontier, changed) = csr_cpu_ref(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[0b10, 0b01, 0b01, 0b01],
        &[0b0001],
        0b01,
    );
    // In-place expansion: node 0 adds node 2 (allowed edge), then node 2
    // (now set in the same buffer) adds node 3, producing {0,2,3}.
    assert_eq!(
        frontier,
        vec![0b1101],
        "in-place expansion cascades within one pass"
    );
    assert_eq!(changed, 1);
}

#[test]
fn csr_forward_or_changed_zero_nodes() {
    let (frontier, changed) = csr_cpu_ref(0, &[0], &[], &[], &[], 1);
    assert!(frontier.is_empty());
    assert_eq!(changed, 0);
}

// ---------------------------------------------------------------------------
// Dominator frontier
// ---------------------------------------------------------------------------

#[test]
fn dominator_frontier_empty_seed_empty_frontier() {
    let out = dom_cpu_ref(4, &[0, 0, 0, 0, 0], &[], &[0, 0, 0, 0, 0], &[], &[0]);
    assert_eq!(out, vec![0]);
}

#[test]
fn dominator_frontier_single_node_no_predecessors() {
    let out = dom_cpu_ref(2, &[0, 0, 0], &[], &[0, 0, 0], &[], &[0b01]);
    assert_eq!(out, vec![0]);
}

#[test]
fn dominator_frontier_join_node_appears() {
    // CFG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let pred_offsets = vec![0u32, 0, 1, 2, 4];
    let pred_targets = vec![0u32, 0, 1, 2];
    // Dominator sets: 0 dominates everyone; 1 dominates {1}; 2 dominates {2}; 3 dominates {3}
    let dom_offsets = vec![0u32, 4, 5, 6, 7];
    let dom_targets = vec![0u32, 1, 2, 3, 1, 2, 3];
    let out = dom_cpu_ref(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
    );
    assert_eq!(out, vec![0b1000], "df(1) must include join node 3");
}

#[test]
#[should_panic(expected = "complete seed bitset")]
fn dominator_frontier_missing_seed_fails_loudly() {
    let _ = dom_cpu_ref(2, &[0, 0, 0], &[], &[0, 0, 0], &[], &[]);
}

// ---------------------------------------------------------------------------
// Fixpoint convergence invariants
// ---------------------------------------------------------------------------

#[test]
fn fixpoint_reference_eval_equal_is_zero() {
    assert_eq!(reference_eval(&[0b1010], &[0b1010]), 0);
    assert_eq!(reference_eval(&[0xFFFF_FFFF; 16], &[0xFFFF_FFFF; 16]), 0);
    assert_eq!(reference_eval(&[], &[]), 0);
}

#[test]
fn fixpoint_reference_eval_different_is_one() {
    assert_eq!(reference_eval(&[0b1010], &[0b1011]), 1);
    assert_eq!(reference_eval(&[0; 16], &[1; 16]), 1);
}

#[test]
fn fixpoint_reference_eval_mismatched_lengths_is_one() {
    assert_eq!(reference_eval(&[0, 0], &[0]), 1);
}

#[test]
fn fixpoint_warm_start_zero_seed_equals_cold() {
    let (updated, flag) = reference_eval_warm_start(&[0b0001], &[0b0001], &[0]);
    assert_eq!(updated, vec![0b0001]);
    assert_eq!(flag, 0);
}

#[test]
fn fixpoint_warm_start_seed_overwrites_current() {
    let (updated, flag) = reference_eval_warm_start(&[0b0001], &[0b0011], &[0b1111]);
    assert_eq!(updated, vec![0b1111]);
    assert_eq!(flag, 1, "c0 (0b0001) != next (0b0011) → flag must be 1");
}

#[test]
fn fixpoint_warm_start_empty_bitsets() {
    let (updated, flag) = reference_eval_warm_start(&[], &[], &[]);
    assert!(updated.is_empty());
    assert_eq!(flag, 0);
}

#[test]
fn fixpoint_warm_start_large_bitsets() {
    let current = vec![0xAAAAAAAAu32; 1024];
    let next = vec![0xBBBBBBBBu32; 1024];
    let seed = vec![0x11111111u32; 1024];
    let (updated, flag) = reference_eval_warm_start(&current, &next, &seed);
    assert_eq!(updated.len(), 1024);
    assert_eq!(updated[0], 0xBBBBBBBB);
    assert_eq!(flag, 1);
}

#[test]
fn fixpoint_monotonic_growth_invariant() {
    // Simulate two fixpoint steps: current0 -> next0 -> current1
    let current0 = vec![0b0001u32];
    let next0 = vec![0b0011u32];
    let next1 = vec![0b0011u32]; // no further growth

    let flag0 = reference_eval(&current0, &next0);
    assert_eq!(flag0, 1, "first step must signal change");

    let flag1 = reference_eval(&next0, &next1);
    assert_eq!(flag1, 0, "second step must signal convergence");
}

#[test]
fn fixpoint_idempotence_after_convergence() {
    let converged = vec![0b1111u32];
    let flag = reference_eval(&converged, &converged);
    assert_eq!(flag, 0, "identical inputs must always yield 0");
}

#[test]
fn fixpoint_warm_start_anticipates_transfer() {
    // current = 0b0001, transfer says next = 0b0011, seed = 0b0010 anticipates delta.
    // c0 != next → flag must still be 1 because transfer added new bits.
    let (updated, flag) = reference_eval_warm_start(&[0b0001], &[0b0011], &[0b0010]);
    assert_eq!(updated, vec![0b0011]);
    assert_eq!(flag, 1);
}
