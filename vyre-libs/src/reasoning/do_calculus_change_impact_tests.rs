//! Tests for do-calculus change impact against the reference witnesses.

use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use vyre_reference::composition_witness::{
    do_rule3_subgraph_witness, do_rule3_subgraph_witness_into,
    predict_impact_observation_form_witness, predict_impact_observation_form_witness_into,
    predict_impact_witness, predict_impact_witness_into,
};

fn predict_impact(adj: &[u32], intervention_mask: &[u32], n: u32) -> Vec<u32> {
    predict_impact_witness(adj, intervention_mask, n)
}

fn predict_impact_with_scratch(
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) {
    predict_impact_witness_into(
        adj,
        intervention_mask,
        n,
        &mut scratch.surgically_modified_adj,
        &mut scratch.closure,
        &mut scratch.impact_mask,
    );
    scratch.scratch.clear();
    scratch.scratch.resize((n * n) as usize, 0);
}

fn impact_subgraph(adj: &[u32], mask: &[u32], n: u32) -> (Vec<u32>, Vec<u32>) {
    let impact = predict_impact_witness(adj, mask, n);
    do_rule3_subgraph_witness(adj, &impact, n)
}

fn reference_impact_subgraph_with_scratch(
    adj: &[u32],
    mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) {
    predict_impact_with_scratch(adj, mask, n, scratch);
    do_rule3_subgraph_witness_into(
        adj,
        &scratch.impact_mask,
        n,
        &mut scratch.reduced_adjacency,
        &mut scratch.kept_indices,
    );
}

fn predict_impact_observation_form(adj: &[u32], observation_mask: &[u32], n: u32) -> Vec<u32> {
    predict_impact_observation_form_witness(adj, observation_mask, n)
}

fn predict_impact_observation_form_with_scratch(
    adj: &[u32],
    observation_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) {
    predict_impact_observation_form_witness_into(
        adj,
        observation_mask,
        n,
        &mut scratch.surgically_modified_adj,
        &mut scratch.closure,
        &mut scratch.impact_mask,
    );
    scratch.scratch.clear();
    scratch.scratch.resize((n * n) as usize, 0);
}

#[test]
fn zero_node_validation_precedes_scratch_mutation() {
    struct NoDispatch;
    impl SemanticExecutor for NoDispatch {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                panic!("invalid zero-node inputs must fail before dispatch");
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    let assert_untouched = |scratch: &DoCalculusImpactScratch, case: &str| {
        assert_eq!(scratch.impact_mask, [11], "{case}: impact mask changed");
        assert_eq!(
            scratch.surgically_modified_adj,
            [12],
            "{case}: surgery scratch changed"
        );
        assert_eq!(scratch.closure, [13], "{case}: closure scratch changed");
    };

    let mut scratch = seeded_impact_scratch();
    let result = predict_impact_via_into(
        &NoDispatch,
        &crate::test_parity_oracles::policy(),
        &[1],
        &[],
        0,
        &mut scratch,
    );
    assert!(matches!(
        result,
        Err(SemanticExecutionError::InvalidRequest(_))
    ));
    assert_untouched(&scratch, "impact adjacency");

    let mut scratch = seeded_impact_scratch();
    let result = predict_impact_via_into(
        &NoDispatch,
        &crate::test_parity_oracles::policy(),
        &[],
        &[1],
        0,
        &mut scratch,
    );
    assert!(matches!(
        result,
        Err(SemanticExecutionError::InvalidRequest(_))
    ));
    assert_untouched(&scratch, "impact mask");

    let mut scratch = seeded_impact_scratch();
    let result = predict_impact_observation_form_via_into(
        &NoDispatch,
        &crate::test_parity_oracles::policy(),
        &[1],
        &[],
        0,
        &mut scratch,
    );
    assert!(matches!(
        result,
        Err(SemanticExecutionError::InvalidRequest(_))
    ));
    assert_untouched(&scratch, "observation adjacency");

    let mut scratch = seeded_impact_scratch();
    let result = predict_impact_observation_form_via_into(
        &NoDispatch,
        &crate::test_parity_oracles::policy(),
        &[],
        &[1],
        0,
        &mut scratch,
    );
    assert!(matches!(
        result,
        Err(SemanticExecutionError::InvalidRequest(_))
    ));
    assert_untouched(&scratch, "observation mask");
}

fn seeded_impact_scratch() -> DoCalculusImpactScratch {
    DoCalculusImpactScratch {
        impact_mask: vec![11],
        surgically_modified_adj: vec![12],
        closure: vec![13],
        ..DoCalculusImpactScratch::default()
    }
}

#[test]
fn chain_impact() {
    // 0 -> 1 -> 2
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    // Change node 0
    let mask = vec![1, 0, 0];
    let impact = predict_impact(&adj, &mask, 3);
    // All impacted
    assert_eq!(impact, vec![1, 1, 1]);
}

#[test]
fn impact_scratch_reuses_matrix_buffers() {
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    let mask = vec![1, 0, 0];
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_with_scratch(&adj, &mask, 3, &mut scratch);
    let modified_capacity = scratch.surgically_modified_adj.capacity();
    let closure_capacity = scratch.closure.capacity();
    let temp_capacity = scratch.scratch.capacity();
    let mask_capacity = scratch.impact_mask.capacity();
    assert_eq!(scratch.impact_mask(), &[1, 1, 1]);

    predict_impact_with_scratch(&adj, &[0, 1, 0], 3, &mut scratch);
    assert_eq!(
        scratch.surgically_modified_adj.capacity(),
        modified_capacity
    );
    assert_eq!(scratch.closure.capacity(), closure_capacity);
    assert_eq!(scratch.scratch.capacity(), temp_capacity);
    assert_eq!(scratch.impact_mask.capacity(), mask_capacity);
    assert_eq!(scratch.impact_mask(), &[0, 1, 1]);
}

#[test]
fn middle_chain_impact() {
    // 0 -> 1 -> 2
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    // Change node 1
    let mask = vec![0, 1, 0];
    let impact = predict_impact(&adj, &mask, 3);
    // 1 and 2 impacted, 0 not impacted
    assert_eq!(impact, vec![0, 1, 1]);
}

#[test]
fn branched_impact() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let adj = vec![0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0];
    // Change node 2
    let mask = vec![0, 0, 1, 0];
    let impact = predict_impact(&adj, &mask, 4);
    // 2 and 3 impacted
    assert_eq!(impact, vec![0, 0, 1, 1]);
}

#[test]
fn disjoint_impact() {
    // 0 -> 1, 2 -> 3
    let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    // Change node 0
    let mask = vec![1, 0, 0, 0];
    let impact = predict_impact(&adj, &mask, 4);
    // 0 and 1 impacted
    assert_eq!(impact, vec![1, 1, 0, 0]);
}

#[test]
fn cycle_impact() {
    // 0 -> 1, 1 -> 0, 1 -> 2
    let adj = vec![0, 1, 0, 1, 0, 1, 0, 0, 0];
    // Change node 0.
    // do(0) removes 1 -> 0.
    // 0 -> 1 -> 2 remains.
    let mask = vec![1, 0, 0];
    let impact = predict_impact(&adj, &mask, 3);
    // All impacted
    assert_eq!(impact, vec![1, 1, 1]);
}

#[test]
fn empty_graph() {
    let impact = predict_impact(&[], &[], 0);
    assert!(impact.is_empty());
}

// ---- impact_subgraph (Rule 3 consumer) ----

#[test]
fn impact_subgraph_chain_extracts_downstream() {
    // 0 -> 1 -> 2. Intervene 0: impact = {0,1,2}, subgraph = full.
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    let mask = vec![1, 0, 0];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 3);
    assert_eq!(kept, vec![0, 1, 2]);
    assert_eq!(reduced, adj);
}

#[test]
fn impact_subgraph_branch_compresses_unimpacted_rows() {
    // 0 -> 1, 2 -> 3 (disjoint). Intervene 0: impact = {0,1};
    // reduced is 2×2, kept = [0, 1].
    let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    let mask = vec![1, 0, 0, 0];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 4);
    assert_eq!(kept, vec![0, 1]);
    // Edge 0->1 preserved, 2x2 layout.
    assert_eq!(reduced, vec![0, 1, 0, 0]);
}

#[test]
fn impact_subgraph_scratch_reuses_reduction_buffers() {
    let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    let mut scratch = DoCalculusImpactScratch::default();
    reference_impact_subgraph_with_scratch(&adj, &[1, 0, 0, 0], 4, &mut scratch);
    let reduced_capacity = scratch.reduced_adjacency.capacity();
    let kept_capacity = scratch.kept_indices.capacity();
    assert_eq!(scratch.kept_indices(), &[0, 1]);
    assert_eq!(scratch.reduced_adjacency(), &[0, 1, 0, 0]);

    reference_impact_subgraph_with_scratch(&adj, &[0, 0, 1, 0], 4, &mut scratch);
    assert_eq!(scratch.reduced_adjacency.capacity(), reduced_capacity);
    assert_eq!(scratch.kept_indices.capacity(), kept_capacity);
    assert_eq!(scratch.kept_indices(), &[2, 3]);
    assert_eq!(scratch.reduced_adjacency(), &[0, 1, 0, 0]);
}

#[test]
fn impact_subgraph_empty_intervention_empty_subgraph() {
    let adj = vec![0, 1, 0, 0];
    let mask = vec![0, 0];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 2);
    assert!(reduced.is_empty());
    assert!(kept.is_empty());
}

#[test]
fn impact_subgraph_empty_graph() {
    let (r, k) = impact_subgraph(&[], &[], 0);
    assert!(r.is_empty());
    assert!(k.is_empty());
}

/// Closure-bar test: the reduced adjacency must have **exactly**
/// `kept.len()²` cells AND every cell must equal the original
/// adjacency restricted to the corresponding kept-index pair. If
/// the consumer ever drifts (off-by-one indexing into the kept
/// vector, mis-sized output buffer, etc.) this test fires.
#[test]
fn impact_subgraph_size_invariant_holds_under_partial_impact() {
    // 0 -> 1 -> 2, plus disjoint 3 -> 4. Intervene 1.
    // Impact = {1, 2}; subgraph keeps those two with edge 1->2.
    let adj = vec![
        0, 1, 0, 0, 0, // 0 -> 1
        0, 0, 1, 0, 0, // 1 -> 2
        0, 0, 0, 0, 0, // 2
        0, 0, 0, 0, 1, // 3 -> 4
        0, 0, 0, 0, 0, // 4
    ];
    let mask = vec![0, 1, 0, 0, 0];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 5);
    // Exact size invariant.
    assert_eq!(reduced.len(), kept.len() * kept.len());
    assert_eq!(kept, vec![1, 2]);
    // Edge 1->2 preserved at (0,1) in the reduced 2×2.
    assert_eq!(reduced, vec![0, 1, 0, 0]);
}

/// Adversarial: intervention on a leaf must not pull in upstream
/// nodes. `do(leaf)` only impacts leaf itself; if the consumer
/// accidentally also kept ancestors, the kept vec would grow.
#[test]
fn impact_subgraph_adversarial_leaf_intervention_keeps_only_leaf() {
    // 0 -> 1 -> 2. Intervene 2 (leaf).
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    let mask = vec![0, 0, 1];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 3);
    assert_eq!(kept, vec![2]);
    // 1×1, value = adj[2,2] = 0.
    assert_eq!(reduced, vec![0]);
}

/// Adversarial: every edge between kept nodes must survive in
/// the reduced adjacency, and no edge to a dropped node may
/// appear. A common bug is to copy the edge weight from the
/// wrong (i, j) cell of the original  -  a permutation error.
#[test]
fn impact_subgraph_adversarial_dense_must_drop_unkept_edges() {
    // K3 over {0,1,2} plus isolated 3.
    let adj = vec![
        0, 1, 1, 0, // 0 -> 1, 0 -> 2
        1, 0, 1, 0, // 1 -> 0, 1 -> 2
        1, 1, 0, 0, // 2 -> 0, 2 -> 1
        0, 0, 0, 0, // 3 isolated
    ];
    // Intervene 0: rule-1 impact closure walks 0 -> 1 -> 2.
    let mask = vec![1, 0, 0, 0];
    let (reduced, kept) = impact_subgraph(&adj, &mask, 4);
    assert_eq!(kept, vec![0, 1, 2]);
    // Reduced is the original 3×3 corner. Every original edge
    // among {0,1,2} preserved; no row/col for 3.
    assert_eq!(
        reduced,
        vec![
            0, 1, 1, // 0 -> 1, 0 -> 2
            1, 0, 1, // 1 -> 0, 1 -> 2
            1, 1, 0, // 2 -> 0, 2 -> 1
        ]
    );
}

// ---- predict_impact_observation_form (Rule 2 consumer) ----

/// On a DAG, observation-form impact equals intervention-form
/// impact at the observed node itself (no feedback edges to
/// reverse).
#[test]
fn observation_form_dag_observed_self_only() {
    // 0 -> 1 -> 2 (no incoming edges into observed node 0).
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    let mask = vec![1, 0, 0];
    let observed = predict_impact_observation_form(&adj, &mask, 3);
    let intervened = predict_impact(&adj, &mask, 3);
    // On this DAG, observing 0 = intervening on 0.
    assert_eq!(observed, intervened);
}

#[test]
fn observation_form_scratch_reuses_buffers() {
    let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_observation_form_with_scratch(&adj, &[1, 0, 0], 3, &mut scratch);
    let reversed_capacity = scratch.surgically_modified_adj.capacity();
    let closure_capacity = scratch.closure.capacity();
    assert_eq!(scratch.impact_mask(), &[1, 1, 1]);

    predict_impact_observation_form_with_scratch(&adj, &[0, 1, 0], 3, &mut scratch);
    assert_eq!(
        scratch.surgically_modified_adj.capacity(),
        reversed_capacity
    );
    assert_eq!(scratch.closure.capacity(), closure_capacity);
    assert_eq!(scratch.impact_mask(), &[1, 1, 1]);
}

/// Closure-bar: observation-form must include the observed node
/// itself as impact.
#[test]
fn observation_form_marks_observed_node() {
    let adj = vec![0, 1, 0, 0];
    let mask = vec![0, 1];
    let impact = predict_impact_observation_form(&adj, &mask, 2);
    assert_eq!(impact[1], 1, "observed node must be in impact set");
}

/// Adversarial: feedback loop into observed node. Rule-2 reverses
/// the loop edge, so observation-form sees the loop's source as
/// reachable along the reversed edge.
#[test]
fn observation_form_walks_reversed_feedback_edge() {
    // 0 -> 1, 1 -> 0 (mutual feedback), 1 -> 2.
    // Observe 0. Rule-2 reverses 1 -> 0 to 0 -> 1 (already exists,
    // OR-merged); it does NOT reverse 0 -> 1 (target is 0 only).
    // Reachable from 0 in modified graph: 0, 1, 2.
    let adj = vec![0, 1, 0, 1, 0, 1, 0, 0, 0];
    let mask = vec![1, 0, 0];
    let impact = predict_impact_observation_form(&adj, &mask, 3);
    assert_eq!(impact, vec![1, 1, 1]);
}

/// Adversarial: empty observation yields empty impact.
#[test]
fn observation_form_empty_mask_yields_empty() {
    let adj = vec![0, 1, 0, 0];
    let mask = vec![0, 0];
    let impact = predict_impact_observation_form(&adj, &mask, 2);
    assert_eq!(impact, vec![0, 0]);
}

/// Adversarial: empty graph returns empty result.
#[test]
fn observation_form_empty_graph() {
    assert!(predict_impact_observation_form(&[], &[], 0).is_empty());
}

fn assert_mock_dispatch_contract(inputs: &[Vec<u8>], expected_len: usize) {
    assert_eq!(inputs.len(), expected_len);
}

struct InterventionDispatcher;

impl SemanticExecutor for InterventionDispatcher {
    fn execute(
        &self,
        request: &vyre_megakernel::SemanticExecutionRequest<'_>,
    ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
        let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
            assert_mock_dispatch_contract(&inputs, 3);
            let adj = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let mask = crate::dispatch_buffers::read_u32s(&inputs[1]);
            let n = mask.len();
            let mut out = adj;
            for j in 0..n {
                if mask[j] != 0 {
                    for i in 0..n {
                        out[i * n + j] = 0;
                    }
                }
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        };
        let ordered = compute_ordered()?;
        crate::test_parity_oracles::semantic_output(request, ordered)
    }
}

#[test]
fn intervention_delete_incoming_via_dispatches_rule1() {
    let adj = vec![1, 2, 3, 4];
    let out = intervention_delete_incoming_via(
        &InterventionDispatcher,
        &crate::test_parity_oracles::policy(),
        &adj,
        &[1, 0],
        2,
    )
    .unwrap();
    assert_eq!(out, vec![0, 2, 0, 4]);
}

#[test]
fn intervention_delete_incoming_via_rejects_bad_shape() {
    let err = intervention_delete_incoming_via(
        &InterventionDispatcher,
        &crate::test_parity_oracles::policy(),
        &[1, 2, 3],
        &[1, 0],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
}

struct Rule2Dispatcher;

impl SemanticExecutor for Rule2Dispatcher {
    fn execute(
        &self,
        request: &vyre_megakernel::SemanticExecutionRequest<'_>,
    ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
        let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
            assert_mock_dispatch_contract(&inputs, 3);
            let adj = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let mask = crate::dispatch_buffers::read_u32s(&inputs[1]);
            let n = mask.len();
            let mut out = vec![0u32; n * n];
            for row in 0..n {
                for col in 0..n {
                    let idx = row * n + col;
                    if row == col {
                        out[idx] = adj[idx];
                        continue;
                    }
                    if mask[col] == 0 {
                        out[idx] |= adj[idx];
                    }
                    if mask[row] != 0 {
                        out[idx] |= adj[col * n + row];
                    }
                }
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        };
        let ordered = compute_ordered()?;
        crate::test_parity_oracles::semantic_output(request, ordered)
    }
}

#[test]
fn rule2_reverse_incoming_via_dispatches_rule2() {
    let adj = vec![
        0, 1, 0, //
        0, 0, 1, //
        0, 0, 0,
    ];
    let out = rule2_reverse_incoming_via(
        &Rule2Dispatcher,
        &crate::test_parity_oracles::policy(),
        &adj,
        &[0, 1, 0],
        3,
    )
    .unwrap();
    assert_eq!(
        out,
        vec![
            0, 0, 0, //
            1, 0, 1, //
            0, 0, 0,
        ]
    );
}

#[test]
fn rule2_reverse_incoming_via_preserves_bidirectional_fully_treated_edges() {
    let adj = vec![0, 1, 1, 0];
    let out = rule2_reverse_incoming_via(
        &Rule2Dispatcher,
        &crate::test_parity_oracles::policy(),
        &adj,
        &[1, 1],
        2,
    )
    .unwrap();
    assert_eq!(out, adj);
}

#[test]
fn rule2_reverse_incoming_via_rejects_bad_shape() {
    let err = rule2_reverse_incoming_via(
        &Rule2Dispatcher,
        &crate::test_parity_oracles::policy(),
        &[1, 2, 3],
        &[1, 0],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
}

#[test]
fn intervention_delete_incoming_via_handles_zero_nodes() {
    let out = intervention_delete_incoming_via(
        &InterventionDispatcher,
        &crate::test_parity_oracles::policy(),
        &[],
        &[],
        0,
    )
    .unwrap();
    assert!(out.is_empty());
}

#[test]
fn intervention_delete_incoming_via_rejects_non_empty_when_n_zero() {
    let err = intervention_delete_incoming_via(
        &InterventionDispatcher,
        &crate::test_parity_oracles::policy(),
        &[1],
        &[],
        0,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
}

#[test]
fn intervention_delete_incoming_via_rejects_extra_outputs() {
    struct ExtraOutDispatcher;
    impl SemanticExecutor for ExtraOutDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 2, 0, 4]),
                    u32_slice_to_le_bytes(&[0, 0]),
                ])
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let err = intervention_delete_incoming_via(
        &ExtraOutDispatcher,
        &crate::test_parity_oracles::policy(),
        &[1, 2, 3, 4],
        &[1, 0],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn rule2_reverse_incoming_via_handles_zero_nodes() {
    let out = rule2_reverse_incoming_via(
        &Rule2Dispatcher,
        &crate::test_parity_oracles::policy(),
        &[],
        &[],
        0,
    )
    .unwrap();
    assert!(out.is_empty());
}

#[test]
fn rule2_reverse_incoming_via_rejects_extra_outputs() {
    struct ExtraOutDispatcher;
    impl SemanticExecutor for ExtraOutDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 1, 0]),
                    u32_slice_to_le_bytes(&[0]),
                ])
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let err = rule2_reverse_incoming_via(
        &ExtraOutDispatcher,
        &crate::test_parity_oracles::policy(),
        &[0, 1, 1, 0],
        &[1, 1],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn rule3_subgraph_via_handles_zero_nodes() {
    struct DummyDispatcher;
    impl SemanticExecutor for DummyDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                panic!("dispatch should not be invoked for n=0");
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let (reduced, kept) = rule3_subgraph_via(
        &DummyDispatcher,
        &crate::test_parity_oracles::policy(),
        &[],
        &[],
        0,
    )
    .unwrap();
    assert!(reduced.is_empty());
    assert!(kept.is_empty());
}

#[test]
fn rule3_subgraph_via_derives_shape_from_inputs_not_gpu_scalar() {
    struct CorruptedRedundantScalarDispatcher;
    impl SemanticExecutor for CorruptedRedundantScalarDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                    u32_slice_to_le_bytes(&[999]),
                ])
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let (reduced, kept) = rule3_subgraph_via(
        &CorruptedRedundantScalarDispatcher,
        &crate::test_parity_oracles::policy(),
        &[0, 1, 0, 0],
        &[1, 1],
        2,
    )
    .unwrap();
    assert_eq!(reduced, [0, 1, 0, 0]);
    assert_eq!(kept, [0, 1]);
}

#[test]
fn rule3_subgraph_via_rejects_missing_outputs() {
    /// Returns every written graph value but one, which is what an executor
    /// that drops a declared output looks like across the semantic seam.
    struct MissingOutDispatcher;
    impl SemanticExecutor for MissingOutDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let mut output = crate::test_parity_oracles::semantic_output(
                request,
                vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                ],
            )?;
            let dropped = *output.outputs.keys().next_back().ok_or_else(|| {
                SemanticExecutionError::Backend(
                    "Fix: the graph must write at least one value to drop one.".to_string(),
                )
            })?;
            output.outputs.remove(&dropped);
            Ok(output)
        }
    }
    let err = rule3_subgraph_via(
        &MissingOutDispatcher,
        &crate::test_parity_oracles::policy(),
        &[0, 1, 0, 0],
        &[1, 1],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn rule3_subgraph_via_rejects_extra_outputs() {
    struct ExtraOutDispatcher;
    impl SemanticExecutor for ExtraOutDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                    u32_slice_to_le_bytes(&[0]),
                    u32_slice_to_le_bytes(&[0]),
                ])
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let err = rule3_subgraph_via(
        &ExtraOutDispatcher,
        &crate::test_parity_oracles::policy(),
        &[0, 1, 0, 0],
        &[1, 1],
        2,
    )
    .unwrap_err();
    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn project_impacted_lineage_entries_handles_empty_lineage() {
    struct PanicDispatcher;
    impl SemanticExecutor for PanicDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                panic!("empty lineage cells must not dispatch");
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }
    let mut out = vec![99; 4];
    let mut scratch = ImpactedLineageProjectionScratch::default();
    project_impacted_lineage_entries_via_into(
        &PanicDispatcher,
        &crate::test_parity_oracles::policy(),
        &[1, 0],
        &[0; 4],
        2,
        &[],
        &mut scratch,
        &mut out,
    )
    .unwrap();
    assert!(out.is_empty());
}

#[test]
fn project_impacted_lineage_entries_parity_via_reference() {
    use vyre_driver_reference::ReferenceSemanticExecutor;
    let dispatcher = ReferenceSemanticExecutor;
    let impact_mask = vec![1, 0, 0];
    let mut closure = vec![0u32; 9];
    closure[2 * 3 + 0] = 1; // 2 -> 0
    let lineage_cells = vec![0, 1, 2, 99];
    let mut out = Vec::new();
    let mut scratch = ImpactedLineageProjectionScratch::default();
    project_impacted_lineage_entries_via_into(
        &dispatcher,
        &crate::test_parity_oracles::policy(),
        &impact_mask,
        &closure,
        3,
        &lineage_cells,
        &mut scratch,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![1, 0, 1, 0]);
}

#[test]
fn project_impacted_lineage_entries_zero_n_nonempty_lineage_dispatches_zeros() {
    use vyre_driver_reference::ReferenceSemanticExecutor;
    let dispatcher = ReferenceSemanticExecutor;
    let lineage_cells = vec![0, 1, 99];
    let mut out = Vec::new();
    let mut scratch = ImpactedLineageProjectionScratch::default();
    project_impacted_lineage_entries_via_into(
        &dispatcher,
        &crate::test_parity_oracles::policy(),
        &[],
        &[],
        0,
        &lineage_cells,
        &mut scratch,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![0, 0, 0]);
}
