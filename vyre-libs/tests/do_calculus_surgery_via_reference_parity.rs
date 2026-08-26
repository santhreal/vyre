//! End-to-end parity for the three do-calculus surgery `_via` dispatcher forms.
//!
//! The `*_via` functions in `logic::do_calculus_change_impact` are the GPU/IR production entry
//! points for do-calculus graph surgery. They pack inputs into LE bytes, build the primitive
//! Program, execute it through a `SemanticExecutor`, and decode canonical graph outputs.
//!
//! This test uses `ReferenceSemanticExecutor` backed by `vyre_reference::reference_eval` and proves
//! each surgery `_via` reproduces its CPU oracle:
//!   - `intervention_delete_incoming_via` (column-zeroing, single output)
//!   - `rule2_reverse_incoming_via` (edge reversal, single output)
//!   - `rule3_subgraph_via` (subgraph extraction, reduced k×k + kept map)
//! Rule 3 is the one that never had a `_via` at all before; this locks its full round-trip
//! (compaction + gather + kept-index map + stride-k block) through the real dispatch boundary.
#![forbid(unsafe_code)]

mod semantic_execution_support;

use vyre_libs::reasoning::do_calculus_change_impact::{
    intervention_delete_incoming_via, rule2_reverse_incoming_via, rule3_subgraph_via,
};
use vyre_reference::composition_witness::{
    do_intervention_delete_incoming_witness as do_intervention_delete_incoming_cpu,
    do_rule2_reverse_incoming_witness as do_rule2_reverse_incoming_cpu,
    do_rule3_subgraph_witness as do_rule3_subgraph_cpu,
};

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_test_support::fixed_point::xorshift32 as xorshift;

#[test]
fn intervention_delete_incoming_via_matches_cpu_oracle() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0xC0FF_EE01u32;
    let mut nontrivial = 0u32;
    for case in 0..300u32 {
        let n = 2 + xorshift(&mut state) % 6; // 2..=7
        let cells = (n * n) as usize;
        let adj: Vec<u32> = (0..cells).map(|_| xorshift(&mut state) & 1).collect();
        let mask: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 1).collect();

        let via = intervention_delete_incoming_via(
            &dispatcher,
            &semantic_execution_support::policy(),
            &adj,
            &mask,
            n,
        )
        .expect("intervention_delete_incoming_via must dispatch");
        let cpu = do_intervention_delete_incoming_cpu(&adj, &mask, n);
        if via != adj && mask.iter().any(|&m| m != 0) {
            nontrivial += 1;
        }
        assert_eq!(
            via, cpu,
            "case {case} (n={n}): intervention _via {via:?} != cpu oracle {cpu:?} \
             (adj={adj:?}, mask={mask:?})"
        );
    }
    assert!(
        nontrivial > 80,
        "only {nontrivial}/300 intervention cases deleted an edge, weak input distribution"
    );
}

#[test]
fn rule2_reverse_incoming_via_matches_cpu_oracle() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0x1BAD_B002u32;
    let mut nontrivial = 0u32;
    for case in 0..300u32 {
        let n = 2 + xorshift(&mut state) % 6;
        let cells = (n * n) as usize;
        let adj: Vec<u32> = (0..cells).map(|_| xorshift(&mut state) & 1).collect();
        let mask: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 1).collect();

        let via = rule2_reverse_incoming_via(
            &dispatcher,
            &semantic_execution_support::policy(),
            &adj,
            &mask,
            n,
        )
        .expect("rule2_reverse_incoming_via must dispatch");
        let cpu = do_rule2_reverse_incoming_cpu(&adj, &mask, n);
        if via != adj && mask.iter().any(|&m| m != 0) {
            nontrivial += 1;
        }
        assert_eq!(
            via, cpu,
            "case {case} (n={n}): rule2 _via {via:?} != cpu oracle {cpu:?} \
             (adj={adj:?}, mask={mask:?})"
        );
    }
    assert!(
        nontrivial > 80,
        "only {nontrivial}/300 rule2 cases changed the graph, weak input distribution"
    );
}

#[test]
fn rule3_subgraph_via_matches_cpu_oracle() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0x5EED_3003u32;
    let mut nontrivial = 0u32;
    for case in 0..300u32 {
        let n = 2 + xorshift(&mut state) % 6;
        let cells = (n * n) as usize;
        let adj: Vec<u32> = (0..cells).map(|_| xorshift(&mut state) & 1).collect();
        // Mixed full/partial/empty keep masks.
        let keep_mask: Vec<u32> = (0..n).map(|_| (xorshift(&mut state) % 3) & 1).collect();

        let (via_reduced, via_kept) = rule3_subgraph_via(
            &dispatcher,
            &semantic_execution_support::policy(),
            &adj,
            &keep_mask,
            n,
        )
        .expect("rule3_subgraph_via must dispatch");
        let (cpu_reduced, cpu_kept) = do_rule3_subgraph_cpu(&adj, &keep_mask, n);
        let k = cpu_kept.len();
        if k > 0 && k < n as usize && cpu_reduced.iter().any(|&e| e != 0) {
            nontrivial += 1;
        }
        assert_eq!(
            via_kept, cpu_kept,
            "case {case} (n={n}): rule3 _via kept map {via_kept:?} != cpu {cpu_kept:?} \
             (keep_mask={keep_mask:?})"
        );
        assert_eq!(
            via_reduced, cpu_reduced,
            "case {case} (n={n}): rule3 _via reduced k×k {via_reduced:?} != cpu {cpu_reduced:?} \
             (adj={adj:?}, keep_mask={keep_mask:?})"
        );
    }
    assert!(
        nontrivial > 80,
        "only {nontrivial}/300 rule3 cases were partial edge-preserving extractions, weak distribution"
    );
}

#[test]
fn rule3_subgraph_via_round_trips_a_known_stride_k_extraction() {
    // The same hand-checked case as the primitive-level parity test, but proven through the full
    // dispatcher boundary (pack → three-output dispatch → exact decode → truncate to k×k / k).
    let dispatcher = ReferenceSemanticExecutor;
    let n = 4u32;
    let adj = vec![
        0u32, 0, 1, 1, // 0 -> 2, 0 -> 3
        0, 0, 1, 0, // 1 -> 2 (dropped node)
        0, 0, 0, 1, // 2 -> 3
        1, 0, 0, 0, // 3 -> 0
    ];
    let keep_mask = vec![1u32, 0, 1, 1]; // keep 0, 2, 3
    let (via_reduced, via_kept) = rule3_subgraph_via(
        &dispatcher,
        &semantic_execution_support::policy(),
        &adj,
        &keep_mask,
        n,
    )
    .expect("rule3_subgraph_via must dispatch");
    assert_eq!(
        via_kept,
        vec![0, 2, 3],
        "kept map is the retained originals in order"
    );
    assert_eq!(
        via_reduced,
        vec![0, 1, 1, 0, 0, 1, 1, 0, 0],
        "the stride-3 dense subgraph survives the dispatch round-trip"
    );
}

#[test]
fn intervention_delete_incoming_via_matches_cpu_oracle_empty_graph() {
    let dispatcher = ReferenceSemanticExecutor;
    let via = intervention_delete_incoming_via(
        &dispatcher,
        &semantic_execution_support::policy(),
        &[],
        &[],
        0,
    )
    .expect("empty graph intervention must dispatch cleanly");
    let cpu = do_intervention_delete_incoming_cpu(&[], &[], 0);
    assert_eq!(via, cpu);
    assert!(via.is_empty());
}

#[test]
fn rule2_reverse_incoming_via_matches_cpu_oracle_empty_graph() {
    let dispatcher = ReferenceSemanticExecutor;
    let via = rule2_reverse_incoming_via(
        &dispatcher,
        &semantic_execution_support::policy(),
        &[],
        &[],
        0,
    )
    .expect("empty graph rule 2 must dispatch cleanly");
    let cpu = do_rule2_reverse_incoming_cpu(&[], &[], 0);
    assert_eq!(via, cpu);
    assert!(via.is_empty());
}

#[test]
fn rule3_subgraph_via_matches_cpu_oracle_empty_graph() {
    let dispatcher = ReferenceSemanticExecutor;
    let (via_reduced, via_kept) = rule3_subgraph_via(
        &dispatcher,
        &semantic_execution_support::policy(),
        &[],
        &[],
        0,
    )
    .expect("empty graph rule 3 must dispatch cleanly");
    let (cpu_reduced, cpu_kept) = do_rule3_subgraph_cpu(&[], &[], 0);
    assert_eq!(via_reduced, cpu_reduced);
    assert_eq!(via_kept, cpu_kept);
    assert!(via_reduced.is_empty());
    assert!(via_kept.is_empty());
}
