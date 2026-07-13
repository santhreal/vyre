//! GPU-IR parity for `graph/sum_product_circuit::sum_product_evaluate`: bottom-up evaluation of a
//! topologically-ordered weighted sum-product DAG in 16.16 fixed point, driven through
//! `vyre_reference::reference_eval` with SIGNED leaf values and edge weights.
//!
//! Why this test exists: a SUM node computes `Σ_k fixed_mul(out[child_k], weight_k)` and a PRODUCT
//! node computes `Π_k out[child_k]` (via `fixed_mul_16_16_expr`). The kernel makes NO non-negativity
//! assumption: DISCRIMINATIVE sum-product networks (Dennis & Ventura) carry SIGNED mixture weights,
//! and leaf values in a general weighted DAG are signed. Before the signed-multiply fix (BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`) `fixed_mul` reconstructed its product from the
//! UNSIGNED high word, so a negative weight or child value (a u32 with the top bit set, read as ~2^32)
//! produced a garbage node value, corrupting every marginal above it. Existing coverage
//! (`cost_model_predict_runtime`) uses only NON-negative runtime costs, so the signed regime was
//! untested; this drives the primitive directly with signed weights/leaves.
//!
//! TWO EVALUATORS, BOTH COVERED HERE:
//! - `sum_product_evaluate` (single-pass) has NO barrier between topo levels, so an internal node
//!   reading ANOTHER internal node's `out` races it, correct ONLY for DEPTH-1 circuits (every
//!   SUM/PRODUCT reads leaves). The depth-1 sweeps below lock its SIGNED behavior exactly (the signed
//!   complement to cost_model's non-negative depth-1 coverage).
//! - `sum_product_evaluate_leveled` (the fix for `BUG-sum-product-multilevel-dag-no-topo-barrier`)
//!   drives the same body through the depth-wave harness with a per-level barrier, so it is correct at
//!   ANY depth. The multi-level tests below prove it matches the topo-correct oracle for depth-2 AND
//!   deep chains (depths 2..=6, each level reading the level below), exactly where the single-pass
//!   form is observably wrong.
//!
//! The topo-correct `evaluate_fixed` oracle here matches `sum_product_evaluate_cpu` and is valid at
//! any depth; it equals the single-pass IR only at depth-1, and equals the leveled IR at every depth.
//!
//! BIT-EXACT: SUM = `Σ fixed_mul(leaf, weight)` with wrapping add, PRODUCT = fold from `1.0` via
//! `fixed_mul`. `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`. Any
//! divergence is a real IR/dispatch defect, not a rounding artifact.
#![cfg(feature = "graph")]

use vyre_primitives::graph::sum_product_circuit::{
    sum_product_depths, sum_product_evaluate, sum_product_evaluate_leveled, KIND_LEAF,
    KIND_PRODUCT, KIND_SUM,
};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;
const FIXED_ONE_U: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated.
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Flat CSR-style circuit buffers.
struct Circuit {
    kinds: Vec<u32>,
    child_offsets: Vec<u32>,
    child_counts: Vec<u32>,
    children: Vec<u32>,
    weights: Vec<u32>,
    leaf_values: Vec<u32>,
}

/// Build a DEPTH-1 circuit: `L` leaves, then a SUM over all leaves (signed weights) and a PRODUCT over
/// two leaves. BOTH internal nodes read LEAVES ONLY, so they commit-before-read faithfully through the
/// single-pass IR (avoiding the multi-level topo-barrier race).
fn build_circuit(leaves: &[u32], sum_weights: &[u32]) -> Circuit {
    let l = leaves.len();
    let n_nodes = l + 2;
    let sum_node = l;
    let prod_node = l + 1;

    let mut kinds = vec![KIND_LEAF; n_nodes];
    kinds[sum_node] = KIND_SUM;
    kinds[prod_node] = KIND_PRODUCT;

    let mut leaf_values = vec![0u32; n_nodes];
    leaf_values[..l].copy_from_slice(leaves);

    // children/weights flat: [0..l-1] for the sum, then [0, 1] (two leaves) for the product.
    let mut children = Vec::new();
    let mut weights = Vec::new();
    let mut child_offsets = vec![0u32; n_nodes];
    let mut child_counts = vec![0u32; n_nodes];

    // SUM node: children = all leaves, weights = sum_weights (all children are leaves → depth-1).
    child_offsets[sum_node] = children.len() as u32;
    child_counts[sum_node] = l as u32;
    for i in 0..l {
        children.push(i as u32);
        weights.push(sum_weights[i]);
    }
    // PRODUCT node: children = leaves [0, 1] (depth-1); weights unused by product (fill 0).
    child_offsets[prod_node] = children.len() as u32;
    child_counts[prod_node] = 2;
    children.push(0);
    weights.push(0);
    children.push(1);
    weights.push(0);

    Circuit {
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
    }
}

/// Generic bit-exact oracle: evaluate every node bottom-up (topo order guarantees children first).
fn evaluate_fixed(c: &Circuit) -> Vec<u32> {
    let n = c.kinds.len();
    let mut out = vec![0u32; n];
    for t in 0..n {
        let co = c.child_offsets[t] as usize;
        let cc = c.child_counts[t] as usize;
        out[t] = match c.kinds[t] {
            KIND_LEAF => c.leaf_values[t],
            KIND_SUM => {
                let mut acc = 0u32;
                for k in 0..cc {
                    let child = c.children[co + k] as usize;
                    acc = acc.wrapping_add(fixed_mul(out[child], c.weights[co + k]));
                }
                acc
            }
            KIND_PRODUCT => {
                let mut acc = FIXED_ONE_U;
                for k in 0..cc {
                    let child = c.children[co + k] as usize;
                    acc = fixed_mul(acc, out[child]);
                }
                acc
            }
            other => panic!("unknown node kind {other}"),
        };
    }
    out
}

fn run_via_reference(c: &Circuit) -> Vec<u32> {
    let n_nodes = c.kinds.len() as u32;
    let n_edges = c.children.len() as u32;
    let program = sum_product_evaluate(
        "kinds",
        "child_offsets",
        "child_counts",
        "children",
        "weights",
        "leaf_values",
        "out",
        n_nodes,
        n_edges,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(&c.kinds)),
            Value::from(pack_u32(&c.child_offsets)),
            Value::from(pack_u32(&c.child_counts)),
            Value::from(pack_u32(&c.children)),
            Value::from(pack_u32(&c.weights)),
            Value::from(pack_u32(&c.leaf_values)),
            Value::from(pack_u32(&vec![0u32; n_nodes as usize])),
        ],
    )
    .expect("sum_product_evaluate reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn sum_product_signed_matches_exact_fixed_point_evaluation() {
    let mut state = 0x5A17_C0DEu32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let l = 2 + (case % 3) as usize; // 2..=4 leaves
        let leaves: Vec<u32> = (0..l).map(|_| signed_fixed(&mut state)).collect();
        let sum_weights: Vec<u32> = (0..l).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += leaves
            .iter()
            .chain(&sum_weights)
            .filter(|&&v| (v as i32) < 0)
            .count() as u32;

        let circuit = build_circuit(&leaves, &sum_weights);
        let got = run_via_reference(&circuit);
        let want = evaluate_fixed(&circuit);
        assert_eq!(
            got, want,
            "case {case} (l={l}): SIGNED sum-product evaluation _via {got:?} != exact signed oracle \
             {want:?} (leaves={leaves:?} weights={sum_weights:?})"
        );

        // Root = product node = last index.
        let root = *want.last().unwrap();
        if root != 0 {
            nontrivial += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative leaf/weight entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed circuits must produce negative node values, got {neg_outputs}"
    );
    assert!(
        nontrivial > 300,
        "expected >300 nonzero roots, got {nontrivial}"
    );
}

#[test]
fn sum_product_hand_checked_signed() {
    // 2 leaves [2.0, -1.0]; SUM weights [1.0, 3.0]; PRODUCT over leaves [0, 1] (depth-1):
    //   out[0] = 2.0 (leaf)
    //   out[1] = -1.0 (leaf)
    //   out[2] = SUM = fixed_mul(2.0,1.0) + fixed_mul(-1.0,3.0) = 2.0 - 3.0 = -1.0
    //   out[3] = PRODUCT = 1.0 · out[0] · out[1] = (2.0)(-1.0) = -2.0
    let leaves = vec![to_fixed(2.0), to_fixed(-1.0)];
    let sum_weights = vec![to_fixed(1.0), to_fixed(3.0)];
    let circuit = build_circuit(&leaves, &sum_weights);
    let got = run_via_reference(&circuit);
    let want = evaluate_fixed(&circuit);
    assert_eq!(
        want,
        vec![
            to_fixed(2.0),
            to_fixed(-1.0),
            to_fixed(-1.0),
            to_fixed(-2.0)
        ],
        "sanity: signed sum-product node values = [2.0, -1.0, -1.0, -2.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched circuit must preserve sign; root = -2.0"
    );
}

/// Build a DEPTH-2 circuit: `L` leaves (depth 0), a SUM over all leaves (depth 1), then a PRODUCT over
/// {sum_node, leaf 0} (depth 2, reads the internal SUM node). Returns the circuit, per-node depths,
/// and `max_depth` (one past the deepest node).
fn build_depth2_circuit(leaves: &[u32], sum_weights: &[u32]) -> (Circuit, Vec<u32>, u32) {
    let l = leaves.len();
    let n_nodes = l + 2;
    let sum_node = l;
    let prod_node = l + 1;

    let mut kinds = vec![KIND_LEAF; n_nodes];
    kinds[sum_node] = KIND_SUM;
    kinds[prod_node] = KIND_PRODUCT;

    let mut leaf_values = vec![0u32; n_nodes];
    leaf_values[..l].copy_from_slice(leaves);

    let mut children = Vec::new();
    let mut weights = Vec::new();
    let mut child_offsets = vec![0u32; n_nodes];
    let mut child_counts = vec![0u32; n_nodes];

    // SUM over all leaves (depth 1).
    child_offsets[sum_node] = children.len() as u32;
    child_counts[sum_node] = l as u32;
    for i in 0..l {
        children.push(i as u32);
        weights.push(sum_weights[i]);
    }
    // PRODUCT over {sum_node, leaf 0} (depth 2 (reads the internal SUM node)).
    child_offsets[prod_node] = children.len() as u32;
    child_counts[prod_node] = 2;
    children.push(sum_node as u32);
    weights.push(0);
    children.push(0);
    weights.push(0);

    let mut depths = vec![0u32; n_nodes];
    depths[sum_node] = 1;
    depths[prod_node] = 2;

    let circuit = Circuit {
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
    };
    (circuit, depths, 3)
}

fn run_leveled(c: &Circuit, depths: &[u32], max_depth: u32) -> Vec<u32> {
    let n_nodes = c.kinds.len() as u32;
    let n_edges = c.children.len() as u32;
    let program = sum_product_evaluate_leveled(
        "depths",
        "kinds",
        "child_offsets",
        "child_counts",
        "children",
        "weights",
        "leaf_values",
        "out",
        n_nodes,
        n_edges,
        max_depth,
    );
    // Buffer/binding order: depths(0), kinds(1), child_offsets(2), child_counts(3), children(4),
    // weights(5), leaf_values(6), out(7). outputs[0] = the sole ReadWrite buffer `out`.
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(depths)),
            Value::from(pack_u32(&c.kinds)),
            Value::from(pack_u32(&c.child_offsets)),
            Value::from(pack_u32(&c.child_counts)),
            Value::from(pack_u32(&c.children)),
            Value::from(pack_u32(&c.weights)),
            Value::from(pack_u32(&c.leaf_values)),
            Value::from(pack_u32(&vec![0u32; n_nodes as usize])),
        ],
    )
    .expect("sum_product_evaluate_leveled reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn sum_product_leveled_evaluates_multilevel_dag_that_single_pass_gets_wrong() {
    // A DEPTH-2 circuit (PRODUCT reads the internal SUM node) is exactly the case the single-pass IR
    // races (BACKLOG BUG-sum-product-multilevel-dag-no-topo-barrier): the product reads the sum's
    // `out` cell before it commits → root = 0. The LEVELED evaluator drives the same body through the
    // depth-wave harness with a barrier between levels, so the product reads the committed sum → the
    // topo-correct value. This proves the fix AND that it preserves sign.
    let mut state = 0x0DEF_ACE5u32;
    let mut checked_wrong_single_pass = 0u32;
    let mut neg_roots = 0u32;
    for case in 0..300u32 {
        let l = 2 + (case % 3) as usize; // 2..=4 leaves
        let leaves: Vec<u32> = (0..l).map(|_| signed_fixed(&mut state)).collect();
        let sum_weights: Vec<u32> = (0..l).map(|_| signed_fixed(&mut state)).collect();

        let (circuit, depths, max_depth) = build_depth2_circuit(&leaves, &sum_weights);
        let want = evaluate_fixed(&circuit); // topo-correct oracle at any depth
        let root_idx = want.len() - 1;

        // LEVELED must reproduce the topo-correct evaluation exactly.
        let leveled = run_leveled(&circuit, &depths, max_depth);
        assert_eq!(
            leveled, want,
            "case {case} (l={l}): leveled sum-product must match the topo oracle at depth 2; \
             leaves={leaves:?} weights={sum_weights:?}"
        );

        // Demonstrate the bug the fix closes: single-pass races the depth-2 read → wrong root.
        let single = run_via_reference(&circuit);
        if single[root_idx] != want[root_idx] {
            checked_wrong_single_pass += 1;
        }
        if (want[root_idx] as i32) < 0 {
            neg_roots += 1;
        }
    }
    assert!(
        checked_wrong_single_pass > 250,
        "the single-pass IR must be observably WRONG on these depth-2 circuits (the bug the leveled \
         form fixes), got {checked_wrong_single_pass}/300 wrong"
    );
    assert!(
        neg_roots > 80,
        "the leveled evaluator must handle SIGNED depth-2 roots, got {neg_roots} negative roots"
    );
}

#[test]
fn sum_product_leveled_hand_checked_depth2() {
    // 2 leaves [2.0, -1.0]; SUM weights [1.0, 3.0] (depth 1); PRODUCT over {sum, leaf0} (depth 2):
    //   out[2] = SUM = fixed_mul(2.0,1.0) + fixed_mul(-1.0,3.0) = -1.0
    //   out[3] = PRODUCT = 1.0 · out[2] · out[0] = (-1.0)(2.0) = -2.0   (needs the sum committed first)
    let leaves = vec![to_fixed(2.0), to_fixed(-1.0)];
    let sum_weights = vec![to_fixed(1.0), to_fixed(3.0)];
    let (circuit, depths, max_depth) = build_depth2_circuit(&leaves, &sum_weights);
    let want = evaluate_fixed(&circuit);
    assert_eq!(
        want,
        vec![
            to_fixed(2.0),
            to_fixed(-1.0),
            to_fixed(-1.0),
            to_fixed(-2.0)
        ],
        "sanity: topo-correct depth-2 values = [2.0, -1.0, -1.0, -2.0]"
    );
    let leveled = run_leveled(&circuit, &depths, max_depth);
    assert_eq!(
        leveled, want,
        "leveled evaluation must reach the depth-2 root -2.0 (single-pass races to 0)"
    );
}

/// Build a DEEP CHAIN of `depth` internal levels over two leaves: internal node `2+k` (depth `k+1`)
/// alternates SUM/PRODUCT and reads the PREVIOUS internal node (depth `k`) plus a leaf, so EVERY
/// level above the first reads another internal node. This is the general multi-level case (not just
/// the depth-2 special case): a correct evaluator must propagate a committed value up `depth` barriers.
fn build_deep_chain(leaves: &[u32], weight_pool: &[u32], depth: usize) -> (Circuit, Vec<u32>, u32) {
    let l = leaves.len();
    let n_nodes = l + depth;

    let mut kinds = vec![KIND_LEAF; n_nodes];
    let mut leaf_values = vec![0u32; n_nodes];
    leaf_values[..l].copy_from_slice(leaves);

    let mut children = Vec::new();
    let mut weights = Vec::new();
    let mut child_offsets = vec![0u32; n_nodes];
    let mut child_counts = vec![0u32; n_nodes];
    let mut depths = vec![0u32; n_nodes];

    let mut wp = 0usize;
    for k in 0..depth {
        let node = l + k;
        depths[node] = (k + 1) as u32;
        kinds[node] = if k % 2 == 0 { KIND_SUM } else { KIND_PRODUCT };
        // First internal level reads two leaves; every level above reads the previous internal node.
        let (c0, c1) = if k == 0 {
            (0u32, 1u32)
        } else {
            ((l + k - 1) as u32, 0u32)
        };
        child_offsets[node] = children.len() as u32;
        child_counts[node] = 2;
        children.push(c0);
        weights.push(weight_pool[wp % weight_pool.len()]);
        wp += 1;
        children.push(c1);
        weights.push(weight_pool[wp % weight_pool.len()]);
        wp += 1;
    }

    let circuit = Circuit {
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
    };
    (circuit, depths, (depth + 1) as u32)
}

#[test]
fn sum_product_leveled_evaluates_deep_chains_of_arbitrary_depth() {
    // Proves the level_wave fix is NOT specific to depth 2: a chain where every level reads the level
    // below must propagate the committed value up `depth` barriers. For each depth 2..=6 the leveled
    // evaluator must match the topo-correct oracle EXACTLY, while the single-pass form (racing at every
    // internal level) must be observably wrong.
    let mut state = 0xC0FF_EE42u32;
    let mut wrong_single_pass = 0u32;
    let mut total = 0u32;
    let mut neg_roots = 0u32;
    for depth in 2..=6usize {
        for _case in 0..80u32 {
            let leaves = vec![signed_fixed(&mut state), signed_fixed(&mut state)];
            let weight_pool: Vec<u32> = (0..4).map(|_| signed_fixed(&mut state)).collect();
            let (circuit, depths, max_depth) = build_deep_chain(&leaves, &weight_pool, depth);

            let want = evaluate_fixed(&circuit); // topo-correct at any depth
            let leveled = run_leveled(&circuit, &depths, max_depth);
            assert_eq!(
                leveled, want,
                "depth {depth}: leveled deep-chain evaluation must match the topo oracle exactly; \
                 leaves={leaves:?} weights={weight_pool:?} depths={depths:?}"
            );

            let root = *want.last().unwrap();
            let single = run_via_reference(&circuit);
            if *single.last().unwrap() != root {
                wrong_single_pass += 1;
            }
            if (root as i32) < 0 {
                neg_roots += 1;
            }
            total += 1;
        }
    }
    // The single-pass IR must be observably wrong on the vast majority of these deep chains (the race
    // the leveled form fixes). Not 100%: a chain can coincidentally converge (e.g. a zero factor).
    assert!(
        wrong_single_pass * 100 >= total * 80,
        "single-pass must be wrong on >=80% of deep chains, got {wrong_single_pass}/{total}"
    );
    assert!(
        neg_roots > 40,
        "deep signed chains must produce negative roots, got {neg_roots}/{total}"
    );
}

/// The leveled evaluator is driven by DEPTH, not node index, proven with a circuit whose node
/// indices are the REVERSE of topological order (root at index 0, leaf at the last index). Both
/// `sum_product_depths` (which must recover the depths from the DAG structure, not the node order)
/// and the depth-wave evaluation must produce the topologically-correct result; an index-order
/// evaluator physically cannot, since it would evaluate the root before its children.
#[test]
fn sum_product_leveled_evaluates_reverse_index_ordered_dag() {
    // node 0 = PRODUCT(node 1, node 2)  (root, depth 2)
    // node 1 = SUM(node 2) weight 1.0   (depth 1)
    // node 2 = LEAF 2.0                 (depth 0)
    // Topo order is [2, 1, 0] (the exact reverse of the node indices).
    let two = to_fixed(2.0);
    let one = to_fixed(1.0);
    let circuit = Circuit {
        kinds: vec![KIND_PRODUCT, KIND_SUM, KIND_LEAF],
        // node 0 children [1,2] at offset 0; node 1 child [2] at offset 2; node 2 leaf (none).
        child_offsets: vec![0, 2, 0],
        child_counts: vec![2, 1, 0],
        children: vec![1, 2, 2],
        // parallel to children: node 0 product weights unused (0,0); node 1 sum weight 1.0.
        weights: vec![0, 0, one],
        leaf_values: vec![0, 0, two],
    };

    // The depth helper must recover the topological depths from the DAG, not the node order.
    let (depths, max_depth) = sum_product_depths(
        &circuit.child_offsets,
        &circuit.child_counts,
        &circuit.children,
        3,
    )
    .expect("reverse-index DAG is valid");
    assert_eq!(
        depths,
        vec![2, 1, 0],
        "depths follow the DAG (the reverse of node index)"
    );
    assert_eq!(max_depth, 3);

    // Leveled: out[2]=2.0, out[1]=SUM(2.0*1.0)=2.0, out[0]=PRODUCT(2.0,2.0)=4.0. An index-order
    // evaluator would read the root's children before they commit and get a wrong root.
    let leveled = run_leveled(&circuit, &depths, max_depth);
    assert_eq!(
        leveled,
        vec![to_fixed(4.0), two, two],
        "leveled evaluates the reverse-index DAG in topological (depth) order: root = 2.0*2.0 = 4.0"
    );
}

/// Exercise the MULTI-BLOCK path of the depth-wave harness: with `lane_count > 256`,
/// `level_wave_program_with_buffers` emits unrolled per-depth waves separated by `GridSync`
/// barriers (not the single-workgroup `SeqCst` loop the small circuits above use). A 260-leaf
/// depth-2 circuit spans 262 nodes → 2 workgroups, so this is the first leveled parity test to
/// drive the GridSync wave path AND the OOB guard fix (BUG-level-wave-depth-guard-eager-oob-load)
/// on out-of-range lanes across a real multi-workgroup dispatch. The leveled result must still
/// match the topo-correct oracle bit-for-bit.
#[test]
fn sum_product_leveled_multiblock_gridsync_path_matches_oracle() {
    let mut state = 0x5EED_600Du32;
    let n_leaves = 260usize; // + SUM + PRODUCT = 262 nodes > 256 → multi-block GridSync path
    let leaves: Vec<u32> = (0..n_leaves).map(|_| signed_fixed(&mut state)).collect();
    let sum_weights: Vec<u32> = (0..n_leaves).map(|_| signed_fixed(&mut state)).collect();

    let (circuit, depths, max_depth) = build_depth2_circuit(&leaves, &sum_weights);
    assert!(
        circuit.kinds.len() > 256,
        "circuit must exceed one workgroup to reach the GridSync path, got {} nodes",
        circuit.kinds.len()
    );
    let want = evaluate_fixed(&circuit);
    let leveled = run_leveled(&circuit, &depths, max_depth);
    assert_eq!(
        leveled,
        want,
        "leveled evaluation over the multi-block GridSync path must equal the topo-correct oracle \
         (n_nodes={}, max_depth={max_depth})",
        circuit.kinds.len()
    );
}
