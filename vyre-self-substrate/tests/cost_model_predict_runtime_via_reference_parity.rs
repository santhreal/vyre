//! End-to-end parity for the COMPOSITE
//! `analysis::cost_model::predict_runtime_fixed_via`: probabilistic dispatch-cost prediction, through
//! the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the release cost model chains TWO dispatches, a 16.16 sum-product circuit evaluation over the
//! feature DAG, then a conformal order-statistic over historical residuals, and NO
//! `vyre-primitives/tests/*` file runs THIS consumer's composition through a faithful boundary. This is
//! the FIRST-EVER execution of the full circuit→conformal chain through a boundary that models the real
//! backend.
//!
//! Contract (audited CLEAN): two dispatches on one dispatcher 
//!   (1) `sum_product_evaluate_leveled`: depths RO + kinds/offsets/counts RO + children/weights RO +
//!       leaf_values RO + out RW = 8 IC (depths is the host-assigned per-node topological depth driving
//!       the depth-wave harness); decode outputs[0] → per-node values; the point estimate is the LAST
//!       node.
//!   (2) `conformal_threshold`: scores_sorted RO + q_hat RW = 2 IC; decode outputs[0] → the calibrated
//!       upper bound = `sorted[k-1]`, `k = clamp(⌈(1-α)(n+1)⌉, 1, n)`.
//! Both stages are EXACT integer arithmetic, so the composite oracle is BIT-EXACT (no tolerance):
//!   circuit  LEAF: `out[t] = leaf[t]`;
//!            SUM:  `out[t] = Σ_k fixed_mul_16_16(out[children[co+k]], weights[co+k])` (wrapping);
//!            PROD: `out[t] = ∏_k fixed_mul_16_16(acc, out[children[co+k]])`, acc seeded 1.0;
//!   conformal via the importable `conformal_threshold_cpu` oracle.
//!
//! ANY-DEPTH (the consumer now drives the LEVELED evaluator): `predict_runtime_fixed_via` assigns each
//! node its topological depth (`sum_product_depths`) and dispatches `sum_product_evaluate_leveled`,
//! which runs the per-node body through the depth-wave harness with a `SeqCst`/`GridSync` barrier
//! between levels, so a node reading ANOTHER INTERNAL node's `out` sees the committed value, and the
//! multi-level DAG evaluates in topological order through the faithful boundary. This closes
//! `BUG-sum-product-multilevel-dag-no-topo-barrier` for the cost-model consumer (previously the
//! single-pass `sum_product_evaluate` raced across topo levels, a PRODUCT reading a SUM node's output
//! got 0, silently wrong on GPU, masked only by the topo-array `_cpu` oracle). The suite exercises both
//! depth-1 circuits (the sweep) and genuine multi-level DAGs (`..evaluates_multilevel_dag_correctly`),
//! all end to end through the circuit→conformal composite.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::graph::sum_product_circuit::{KIND_LEAF, KIND_PRODUCT, KIND_SUM};
use vyre_primitives::math::conformal::conformal_threshold_cpu;
use vyre_self_substrate::analysis::cost_model::predict_runtime_fixed_via;

mod common;
use common::fixed_mul as fixed_mul_16_16;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A topo-ordered fixed-point sum-product circuit (children strictly before parents).
struct Circuit {
    kinds: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    children: Vec<u32>,
    weights: Vec<u32>,
    leaf_values: Vec<u32>,
}

impl Circuit {
    /// Inline fixed-point replica of the `sum_product_evaluate` IR, evaluated in node (topo) order.
    fn eval_fixed(&self) -> Vec<u32> {
        let n = self.kinds.len();
        let mut out = vec![0u32; n];
        for t in 0..n {
            let co = self.offsets[t] as usize;
            let cc = self.counts[t] as usize;
            out[t] = match self.kinds[t] {
                KIND_LEAF => self.leaf_values[t],
                KIND_SUM => {
                    let mut acc = 0u32;
                    for k in 0..cc {
                        let cn = self.children[co + k] as usize;
                        acc = acc.wrapping_add(fixed_mul_16_16(out[cn], self.weights[co + k]));
                    }
                    acc
                }
                KIND_PRODUCT => {
                    let mut acc = FIXED_ONE;
                    for k in 0..cc {
                        let cn = self.children[co + k] as usize;
                        acc = fixed_mul_16_16(acc, out[cn]);
                    }
                    acc
                }
                other => panic!("unknown circuit kind {other}"),
            };
        }
        out
    }

    fn point_estimate(&self) -> u32 {
        *self.eval_fixed().last().expect("non-empty circuit")
    }
}

/// Build a random VALID DEPTH-1 circuit: `n_leaves` leaves, then `n_internal` sum/product nodes each
/// referencing 1..=fanout LEAVES (children strictly in `[0, n_leaves)`). Depth-1 is the contract
/// `sum_product_evaluate` evaluates correctly through a faithful single-pass boundary (see the module
/// doc / `BUG-sum-product-multilevel-dag-no-topo-barrier`). Guarantees >= 1 edge.
fn random_depth1_circuit(state: &mut u32, n_leaves: usize, n_internal: usize) -> Circuit {
    let mut kinds = Vec::new();
    let mut offsets = Vec::new();
    let mut counts = Vec::new();
    let mut children = Vec::new();
    let mut weights = Vec::new();
    let mut leaf_values = Vec::new();

    for _ in 0..n_leaves {
        kinds.push(KIND_LEAF);
        offsets.push(children.len() as u32);
        counts.push(0);
        // Leaf values in (0, 1.0] in 16.16 keep the products from underflowing to 0 immediately.
        leaf_values.push(1 + xorshift(state) % FIXED_ONE);
    }
    for _ in 0..n_internal {
        let is_sum = xorshift(state) & 1 == 0;
        kinds.push(if is_sum { KIND_SUM } else { KIND_PRODUCT });
        offsets.push(children.len() as u32);
        let fanout = 1 + (xorshift(state) % 3) as usize; // 1..=3 children
        counts.push(fanout as u32);
        for _ in 0..fanout {
            // Children are LEAVES ONLY (indices [0, n_leaves)) → depth-1, committed before this node.
            let child = xorshift(state) % n_leaves as u32;
            children.push(child);
            // Weights in (0, 1.0] so a weighted sum of unit values stays well-scaled.
            weights.push(1 + xorshift(state) % FIXED_ONE);
        }
        leaf_values.push(0); // unused for internal nodes
    }

    Circuit {
        kinds,
        offsets,
        counts,
        children,
        weights,
        leaf_values,
    }
}

fn predict(
    d: &ReferenceEvalDispatcher,
    c: &Circuit,
    residuals_sorted: &[u32],
    alpha: f64,
) -> (u32, u32) {
    predict_runtime_fixed_via(
        d,
        &c.kinds,
        &c.offsets,
        &c.counts,
        &c.children,
        &c.weights,
        &c.leaf_values,
        residuals_sorted,
        alpha,
    )
    .expect("predict_runtime_fixed_via must dispatch circuit + conformal")
}

#[test]
fn predict_runtime_via_matches_exact_composite_oracle() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0xC057_0001u32;
    let mut nonzero_point = 0u32;
    let mut nondegenerate_bound = 0u32;
    for case in 0..400u32 {
        let n_leaves = 1 + (case % 4) as usize; // 1..4
        let n_internal = 1 + (case % 5) as usize; // 1..5 (last node = the root point estimate)
        let circuit = random_depth1_circuit(&mut state, n_leaves, n_internal);

        // Sorted residuals for the conformal stage (ascending, as the release path requires).
        let n_res = 2 + (case % 12) as usize;
        let mut residuals: Vec<u32> = (0..n_res).map(|_| xorshift(&mut state) % 100_000).collect();
        residuals.sort_unstable();
        let alpha = 0.05 + 0.9 * (f64::from(xorshift(&mut state) >> 8) / f64::from(1u32 << 24));

        let (got_point, got_bound) = predict(&d, &circuit, &residuals, alpha);
        let want_point = circuit.point_estimate();
        let want_bound = conformal_threshold_cpu(&residuals, alpha);
        assert_eq!(
            got_point, want_point,
            "case {case}: point estimate (last circuit node) must match the fixed-point oracle"
        );
        assert_eq!(
            got_bound, want_bound,
            "case {case}: conformal upper bound must match conformal_threshold_cpu; alpha={alpha}"
        );

        if got_point != 0 {
            nonzero_point += 1;
        }
        // The conformal bound should land on an actual residual sample.
        if residuals.contains(&got_bound) {
            nondegenerate_bound += 1;
        }
    }
    assert!(
        nonzero_point > 200,
        "sweep must produce nonzero circuit point estimates, got {nonzero_point}"
    );
    assert_eq!(
        nondegenerate_bound, 400,
        "every conformal bound must equal one of the sorted residual samples, got {nondegenerate_bound}/400"
    );
}

#[test]
fn predict_runtime_via_hand_checked_two_level_circuit() {
    let d = ReferenceEvalDispatcher;

    // Two leaves (0.5, 0.25 in 16.16) → one SUM root with unit weights:
    // root = fixed_mul(0.5, 1.0) + fixed_mul(0.25, 1.0) = 0.5 + 0.25 = 0.75.
    let half = FIXED_ONE / 2;
    let quarter = FIXED_ONE / 4;
    let sum_circuit = Circuit {
        kinds: vec![KIND_LEAF, KIND_LEAF, KIND_SUM],
        offsets: vec![0, 0, 0],
        counts: vec![0, 0, 2],
        children: vec![0, 1],
        weights: vec![FIXED_ONE, FIXED_ONE],
        leaf_values: vec![half, quarter, 0],
    };
    let residuals = [10u32, 20, 30, 40];
    let alpha = 0.5;
    let (point, bound) = predict(&d, &sum_circuit, &residuals, alpha);
    assert_eq!(
        point,
        half + quarter,
        "SUM root = 0.5 + 0.25 = 0.75 in 16.16"
    );
    assert_eq!(
        point,
        sum_circuit.point_estimate(),
        "matches the inline oracle"
    );
    // k = clamp(ceil(0.5*(4+1)),1,4) = ceil(2.5)=3 → sorted[2] = 30.
    assert_eq!(
        bound, 30,
        "conformal bound = 3rd order statistic of [10,20,30,40]"
    );
    assert_eq!(bound, conformal_threshold_cpu(&residuals, alpha));

    // Two leaves (0.5, 0.5) → one PRODUCT root: fixed_mul(1.0, 0.5) then *0.5 = 0.25.
    let prod_circuit = Circuit {
        kinds: vec![KIND_LEAF, KIND_LEAF, KIND_PRODUCT],
        offsets: vec![0, 0, 0],
        counts: vec![0, 0, 2],
        children: vec![0, 1],
        weights: vec![0, 0],
        leaf_values: vec![half, half, 0],
    };
    let (point2, _) = predict(&d, &prod_circuit, &residuals, alpha);
    assert_eq!(point2, quarter, "PRODUCT root = 0.5 * 0.5 = 0.25 in 16.16");
    assert_eq!(
        point2,
        prod_circuit.point_estimate(),
        "matches the inline oracle"
    );
}

#[test]
fn predict_runtime_via_hand_checked_multi_internal_depth1() {
    let d = ReferenceEvalDispatcher;
    // Two leaves + TWO internal nodes, both reading only leaves (depth-1). The LAST node is the point
    // estimate; an earlier internal node's value is not read by anyone, but both must evaluate.
    //   n0,n1 leaves = 0.5, 0.25
    //   n2 SUM(n0, n1) with unit weights = 0.75
    //   n3 PRODUCT(n0, n1) = 0.5 * 0.25 = 0.125   (root / point estimate)
    let half = FIXED_ONE / 2;
    let quarter = FIXED_ONE / 4;
    let circuit = Circuit {
        kinds: vec![KIND_LEAF, KIND_LEAF, KIND_SUM, KIND_PRODUCT],
        offsets: vec![0, 0, 0, 2],
        counts: vec![0, 0, 2, 2],
        children: vec![0, 1, 0, 1],
        // Parallel to `children`: n2 (SUM, offset 0) uses weights[0..2]; n3 (PRODUCT, offset 2) ignores.
        weights: vec![FIXED_ONE, FIXED_ONE, 0, 0],
        leaf_values: vec![half, quarter, 0, 0],
    };
    let residuals = [5u32, 15, 25];
    let alpha = 0.2;
    let (point, bound) = predict(&d, &circuit, &residuals, alpha);
    // n3 PRODUCT = fixed_mul(fixed_mul(1.0, 0.5), 0.25) = 0.5 * 0.25 = 0.125 = 1/8 in 16.16.
    assert_eq!(point, FIXED_ONE / 8, "root = 0.5 * 0.25 = 0.125 in 16.16");
    assert_eq!(
        point,
        circuit.point_estimate(),
        "depth-1 multi-internal circuit matches the inline oracle"
    );
    assert_eq!(bound, conformal_threshold_cpu(&residuals, alpha));
    // Full internal vector: n2 (SUM over leaves) = 0.75, n3 (PRODUCT over leaves) = 0.125.
    let full = circuit.eval_fixed();
    assert_eq!(full[2], half + quarter, "n2 SUM over leaves = 0.75");
    assert_eq!(full[3], FIXED_ONE / 8, "n3 PRODUCT over leaves = 0.125");
}

/// PROOF that the multi-level gap is CLOSED (`BUG-sum-product-multilevel-dag-no-topo-barrier`): a node
/// that reads ANOTHER INTERNAL node's output now sees the committed value through the faithful
/// boundary, because `predict_runtime_fixed_via` drives the LEVELED depth-wave evaluator (a `SeqCst`/
/// `GridSync` barrier commits each level before the next reads it). Previously this exact circuit
/// yielded `got_point == 0` (the single-pass race); it now equals the topo-correct oracle. If this ever
/// regresses to 0, the consumer has reverted to a single-pass dispatch.
#[test]
fn predict_runtime_via_evaluates_multilevel_dag_correctly() {
    let d = ReferenceEvalDispatcher;
    let half = FIXED_ONE / 2;
    // n2 SUM(n0=1.0, n1=0.5) = 1.5 ;  n3 PRODUCT(n2, n1) topo-correct = 1.5 * 0.5 = 0.75.
    let circuit = Circuit {
        kinds: vec![KIND_LEAF, KIND_LEAF, KIND_SUM, KIND_PRODUCT],
        offsets: vec![0, 0, 0, 2],
        counts: vec![0, 0, 2, 2],
        children: vec![0, 1, 2, 1], // n3's child `2` is an INTERNAL node (depth 2)
        weights: vec![FIXED_ONE, FIXED_ONE, 0, 0],
        leaf_values: vec![FIXED_ONE, half, 0, 0],
    };
    let residuals = [5u32, 15, 25];
    let (got_point, got_bound) = predict(&d, &circuit, &residuals, 0.2);
    let topo_correct = circuit.point_estimate(); // 0.75 in 16.16
    assert_eq!(
        topo_correct,
        FIXED_ONE * 3 / 4,
        "the topo-correct value is 0.75"
    );
    assert_eq!(
        got_point, topo_correct,
        "multi-level DAG now evaluates in topological order through the leveled evaluator \
         (BUG-sum-product-multilevel-dag-no-topo-barrier CLOSED): the PRODUCT root reads the committed \
         SUM node = 0.75, not the racing 0 the single-pass path used to yield"
    );
    assert_eq!(got_bound, conformal_threshold_cpu(&residuals, 0.2));
}

/// The leveled evaluator must be correct across a RANGE of depths, not just the depth-2 special case:
/// a linear alternating SUM/PRODUCT chain (leaf → internal → internal → …) of depth 2..=5 evaluated by
/// the consumer must match the topo-correct fixed-point oracle at every depth.
#[test]
fn predict_runtime_via_evaluates_deep_chains_at_every_depth() {
    let d = ReferenceEvalDispatcher;
    let residuals = [5u32, 15, 25];
    for depth in 2u32..=5 {
        // Node 0 = leaf(0.5); node k (1..=depth) reads node k-1 (and, for a valid SUM/PRODUCT edge,
        // itself needs >=1 child). Alternate SUM (unit weight = identity on one child) and PRODUCT.
        let n = (depth + 1) as usize;
        let mut kinds = vec![KIND_LEAF];
        let mut offsets = vec![0u32];
        let mut counts = vec![0u32];
        let mut children: Vec<u32> = Vec::new();
        let mut weights: Vec<u32> = Vec::new();
        let mut leaf_values = vec![FIXED_ONE / 2]; // 0.5
        for k in 1..=depth {
            offsets.push(children.len() as u32);
            counts.push(1);
            children.push(k - 1); // reads the previous internal/leaf node
            if k % 2 == 1 {
                kinds.push(KIND_SUM);
                weights.push(FIXED_ONE); // SUM with unit weight = pass-through of the single child
            } else {
                kinds.push(KIND_PRODUCT);
                weights.push(0); // unused by PRODUCT
            }
            leaf_values.push(0);
        }
        let circuit = Circuit {
            kinds,
            offsets,
            counts,
            children,
            weights,
            leaf_values,
        };
        let (got_point, _) = predict(&d, &circuit, &residuals, 0.2);
        let want = circuit.point_estimate();
        assert_eq!(
            got_point, want,
            "depth-{depth} chain (n={n}) must evaluate to the topo-correct oracle through the leveled \
             evaluator; got {got_point} want {want}"
        );
        // A pass-through SUM of a 0.5 leaf followed by PRODUCT-of-one stays 0.5 at every level (unit
        // weight / single-factor product), so the root is a genuine nonzero value the single-pass path
        // could not have produced past depth 1.
        assert_eq!(
            want,
            FIXED_ONE / 2,
            "the chain preserves 0.5 through every level"
        );
    }
}
