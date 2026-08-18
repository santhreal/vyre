use super::*;
use vyre_reference::composition_witness::sum_product_evaluate_witness_into;

#[derive(Default)]
struct SumProductCpuScratch {
    values: Vec<f64>,
}

impl SumProductCpuScratch {
    fn new() -> Self {
        Self::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn try_sum_product_evaluate_cpu_into_with_scratch(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topological_order: &[u32],
    out: &mut Vec<f64>,
    scratch: &mut SumProductCpuScratch,
) -> Result<(), String> {
    let n = kinds.len();
    if child_offsets.len() != n || child_counts.len() != n || leaf_values.len() != n {
        return Err("Fix: mismatched node slice lengths in sum-product evaluation.".to_owned());
    }
    for &node in topological_order {
        let node = node as usize;
        if node >= n {
            return Err(format!(
                "Fix: topological order node {node} outside node_count ({n})."
            ));
        }
        let start = child_offsets[node] as usize;
        let count = child_counts[node] as usize;
        if start.checked_add(count).is_none() || start + count > children.len() {
            return Err(format!(
                "Fix: child range {start}..{} exceeds child_count ({})",
                start + count,
                children.len()
            ));
        }
        if kinds[node] == KIND_SUM && (start + count > weights.len()) {
            return Err("Fix: sum node weight range exceeds weights length.".to_owned());
        }
        for &child in &children[start..start + count] {
            if child as usize >= n {
                return Err(format!(
                    "Fix: child index {child} outside node_count ({n})."
                ));
            }
        }
    }
    sum_product_evaluate_witness_into(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        topological_order,
        &mut scratch.values,
    );
    out.clear();
    out.extend_from_slice(&scratch.values);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_sum_product_evaluate_cpu(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topological_order: &[u32],
) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    let mut scratch = SumProductCpuScratch::new();
    try_sum_product_evaluate_cpu_into_with_scratch(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        topological_order,
        &mut out,
        &mut scratch,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn sum_product_evaluate_cpu(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topological_order: &[u32],
) -> Vec<f64> {
    try_sum_product_evaluate_cpu(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        topological_order,
    )
    .expect("sum_product_evaluate_cpu failed")
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
}

#[test]
fn cpu_single_leaf() {
    let kinds = vec![KIND_LEAF];
    let off = vec![0];
    let cnt = vec![0];
    let kids: Vec<u32> = vec![];
    let w: Vec<f64> = vec![];
    let leaf = vec![0.7];
    let order = vec![0];
    let out = sum_product_evaluate_cpu(&kinds, &off, &cnt, &kids, &w, &leaf, &order);
    assert!(approx_eq(out[0], 0.7));
}

#[test]
fn cpu_sum_of_two_leaves() {
    // Node 0,1 = leaves with values 0.6, 0.4
    // Node 2 = sum with weights 0.5, 0.5 → 0.3 + 0.2 = 0.5
    let kinds = vec![KIND_LEAF, KIND_LEAF, KIND_SUM];
    let off = vec![0, 0, 0];
    let cnt = vec![0, 0, 2];
    let kids = vec![0, 1];
    let w = vec![0.5, 0.5];
    let leaf = vec![0.6, 0.4, 0.0];
    let order = vec![0, 1, 2];
    let out = sum_product_evaluate_cpu(&kinds, &off, &cnt, &kids, &w, &leaf, &order);
    assert!(approx_eq(out[2], 0.5));
}

#[test]
fn cpu_product_of_two_leaves() {
    let kinds = vec![KIND_LEAF, KIND_LEAF, KIND_PRODUCT];
    let off = vec![0, 0, 0];
    let cnt = vec![0, 0, 2];
    let kids = vec![0, 1];
    let w = vec![0.0, 0.0];
    let leaf = vec![0.6, 0.4, 0.0];
    let order = vec![0, 1, 2];
    let out = sum_product_evaluate_cpu(&kinds, &off, &cnt, &kids, &w, &leaf, &order);
    assert!(approx_eq(out[2], 0.6 * 0.4));
}

#[test]
fn cpu_mixture_distribution() {
    // Build a 2-component mixture:
    //   leaf 0 = 0.8 (component 1 likelihood)
    //   leaf 1 = 0.3 (component 2 likelihood)
    //   sum  2 = 0.4 * 0.8 + 0.6 * 0.3 = 0.32 + 0.18 = 0.5
    let kinds = vec![KIND_LEAF, KIND_LEAF, KIND_SUM];
    let off = vec![0, 0, 0];
    let cnt = vec![0, 0, 2];
    let kids = vec![0, 1];
    let w = vec![0.4, 0.6];
    let leaf = vec![0.8, 0.3, 0.0];
    let order = vec![0, 1, 2];
    let out = sum_product_evaluate_cpu(&kinds, &off, &cnt, &kids, &w, &leaf, &order);
    assert!(approx_eq(out[2], 0.5));
}

#[test]
fn cpu_three_layer_circuit() {
    // 4 leaves → 2 product nodes → 1 sum (mixture of two products)
    // p1 = 0.5 * 0.6 = 0.30
    // p2 = 0.7 * 0.8 = 0.56
    // root = 0.3 * 0.30 + 0.7 * 0.56 = 0.09 + 0.392 = 0.482
    let kinds = vec![
        KIND_LEAF,
        KIND_LEAF,
        KIND_LEAF,
        KIND_LEAF,
        KIND_PRODUCT,
        KIND_PRODUCT,
        KIND_SUM,
    ];
    let off = vec![0, 0, 0, 0, 0, 2, 4];
    let cnt = vec![0, 0, 0, 0, 2, 2, 2];
    let kids = vec![0, 1, 2, 3, 4, 5];
    let w = vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.7];
    let leaf = vec![0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0];
    let order = vec![0, 1, 2, 3, 4, 5, 6];
    let out = sum_product_evaluate_cpu(&kinds, &off, &cnt, &kids, &w, &leaf, &order);
    assert!(approx_eq(out[6], 0.482));
}

#[test]
fn checked_cpu_oracle_rejects_missing_child() {
    let error = try_sum_product_evaluate_cpu(
        &[KIND_LEAF, KIND_SUM],
        &[0, 0],
        &[0, 1],
        &[],
        &[],
        &[1.0, 0.0],
        &[0, 1],
    )
    .expect_err("checked sum-product oracle must reject missing child entries");

    assert!(
        error.contains("exceeds child_count"),
        "error should describe the missing child entry: {error}"
    );
}

#[test]
fn scratch_cpu_oracle_rejects_bad_child_without_clobbering_storage() {
    let mut out = vec![9.0, 8.0];
    let mut scratch = SumProductCpuScratch {
        values: vec![7.0, 6.0, 5.0],
    };

    let err = try_sum_product_evaluate_cpu_into_with_scratch(
        &[KIND_LEAF, KIND_SUM],
        &[0, 0],
        &[0, 1],
        &[9],
        &[1.0],
        &[1.0, 0.0],
        &[0, 1],
        &mut out,
        &mut scratch,
    )
    .expect_err("scratch evaluator must reject child indices outside the node range");

    assert!(err.contains("outside node_count"));
    assert_eq!(out, vec![9.0, 8.0]);
    assert_eq!(scratch.values, vec![7.0, 6.0, 5.0]);
}

#[test]
fn scratch_cpu_oracle_reuses_values_and_truncates_stale_tail() {
    let kinds = vec![KIND_LEAF, KIND_LEAF, KIND_SUM];
    let child_offsets = vec![0, 0, 0];
    let child_counts = vec![0, 0, 2];
    let children = vec![0, 1];
    let weights = vec![0.25, 0.75];
    let leaf_values = vec![2.0, 4.0, 0.0];
    let topo_order = vec![0, 1, 2];
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&[99.0, 98.0, 97.0, 96.0]);
    let mut scratch = SumProductCpuScratch {
        values: Vec::with_capacity(8),
    };
    scratch.values.extend_from_slice(&[11.0, 12.0, 13.0, 14.0]);
    let out_capacity = out.capacity();
    let scratch_capacity = scratch.values.capacity();

    try_sum_product_evaluate_cpu_into_with_scratch(
        &kinds,
        &child_offsets,
        &child_counts,
        &children,
        &weights,
        &leaf_values,
        &topo_order,
        &mut out,
        &mut scratch,
    )
    .expect("Fix: scratch allocation must succeed for declared sizes; shrink test fixture or return Err - scratch evaluator should reuse preallocated storage");

    assert_eq!(out.len(), 3);
    assert!(approx_eq(out[0], 2.0));
    assert!(approx_eq(out[1], 4.0));
    assert!(approx_eq(out[2], 3.5));
    assert_eq!(scratch.values, out);
    assert_eq!(out.capacity(), out_capacity);
    assert_eq!(scratch.values.capacity(), scratch_capacity);

    try_sum_product_evaluate_cpu_into_with_scratch(
        &[KIND_LEAF],
        &[0],
        &[0],
        &[],
        &[],
        &[2.0],
        &[0],
        &mut out,
        &mut scratch,
    )
    .expect("Fix: scratch allocation must succeed for declared sizes; shrink test fixture or return Err - scratch evaluator should truncate stale tail values");

    assert_eq!(out, vec![2.0]);
    assert_eq!(scratch.values, vec![2.0]);
    assert_eq!(out.capacity(), out_capacity);
    assert_eq!(scratch.values.capacity(), scratch_capacity);
}

#[test]
fn generated_cpu_oracle_matches_independent_sum_product_evaluator() {
    let mut out = Vec::new();
    let mut scratch = SumProductCpuScratch::new();
    for case in 0..2048usize {
        let leaf_count = 1 + case % 6;
        let n_nodes = leaf_count + 4;
        let mut kinds = Vec::new();
        let mut child_offsets = Vec::new();
        let mut child_counts = Vec::new();
        let mut children = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..leaf_count {
            kinds.push(KIND_LEAF);
            child_offsets.push(0);
            child_counts.push(0);
        }

        for op_idx in 0..4usize {
            let available = leaf_count + op_idx;
            let count = 1 + ((case + op_idx * 3) % available);
            child_offsets.push(children.len() as u32);
            child_counts.push(count as u32);
            kinds.push(if op_idx % 2 == 0 {
                KIND_PRODUCT
            } else {
                KIND_SUM
            });
            for child in 0..count {
                children.push(((child * 5 + case + op_idx) % available) as u32);
                weights.push(((child * 7 + case + op_idx) % 19) as f64 / 23.0);
            }
        }

        let leaf_values: Vec<f64> = (0..n_nodes)
            .map(|idx| ((idx * 11 + case) % 29) as f64 / 31.0)
            .collect();
        let topo_order: Vec<u32> = (0..n_nodes as u32).collect();

        try_sum_product_evaluate_cpu_into_with_scratch(
            &kinds,
            &child_offsets,
            &child_counts,
            &children,
            &weights,
            &leaf_values,
            &topo_order,
            &mut out,
            &mut scratch,
        )
        .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - generated sum-product CPU oracle should reserve and evaluate");
        let expected = independent_sum_product_evaluate(
            &kinds,
            &child_offsets,
            &child_counts,
            &children,
            &weights,
            &leaf_values,
            &topo_order,
        );

        assert_eq!(out.len(), n_nodes, "case {case}: output length mismatch");
        for idx in 0..n_nodes {
            assert!(
                approx_eq(out[idx], expected[idx]),
                "case {case} idx {idx}: expected {}, got {}",
                expected[idx],
                out[idx]
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn independent_sum_product_evaluate(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topo_order: &[u32],
) -> Vec<f64> {
    let mut out = Vec::new();
    out.resize(kinds.len(), 0.0);
    for &node in topo_order {
        let i = node as usize;
        let offset = child_offsets[i] as usize;
        let count = child_counts[i] as usize;
        out[i] = match kinds[i] {
            KIND_LEAF => leaf_values[i],
            KIND_SUM => {
                let mut acc = 0.0;
                for child in 0..count {
                    let edge = offset + child;
                    acc += weights[edge] * out[children[edge] as usize];
                }
                acc
            }
            KIND_PRODUCT => {
                let mut acc = 1.0;
                for child in 0..count {
                    acc *= out[children[offset + child] as usize];
                }
                acc
            }
            _ => 0.0,
        };
    }
    out
}

#[test]
fn depths_all_zero_for_leaf_only_circuit() {
    // Three leaves (child_count 0) → all depth 0, one wave.
    let (depths, max_depth) =
        sum_product_depths(&[0, 0, 0], &[0, 0, 0], &[], 3).expect("leaf-only circuit is valid");
    assert_eq!(depths, vec![0, 0, 0]);
    assert_eq!(max_depth, 1, "one wave (0) fires all leaves");
}

#[test]
fn depths_assign_topological_levels() {
    // 4 leaves → 2 products (depth 1) → 1 sum-of-products root (depth 2).
    // kinds: 0..3 leaves, 4=PROD(0,1), 5=PROD(2,3), 6=SUM(4,5).
    let child_offsets = vec![0, 0, 0, 0, 0, 2, 4];
    let child_counts = vec![0, 0, 0, 0, 2, 2, 2];
    let children = vec![0, 1, 2, 3, 4, 5];
    let (depths, max_depth) =
        sum_product_depths(&child_offsets, &child_counts, &children, 7).expect("valid 3-level DAG");
    assert_eq!(depths, vec![0, 0, 0, 0, 1, 1, 2]);
    assert_eq!(max_depth, 3, "deepest node is level 2 → 3 waves");
    // Every edge strictly increases depth (the level-wave correctness precondition).
    for (parent, (&co, &cc)) in child_offsets.iter().zip(&child_counts).enumerate() {
        for &c in &children[co as usize..(co + cc) as usize] {
            assert!(
                depths[parent] > depths[c as usize],
                "parent {parent} (d={}) must be deeper than child {c} (d={})",
                depths[parent],
                depths[c as usize]
            );
        }
    }
}

#[test]
fn depths_handle_non_index_ordered_nodes() {
    // Root at index 0 reads internal node 1, which reads leaf 2, nodes are NOT
    // in topological index order, so a single forward index pass would be wrong.
    // 0=SUM(1), 1=PRODUCT(2), 2=LEAF.
    let child_offsets = vec![0, 1, 0];
    let child_counts = vec![1, 1, 0];
    let children = vec![1, 2];
    let (depths, max_depth) =
        sum_product_depths(&child_offsets, &child_counts, &children, 3).expect("valid DAG");
    assert_eq!(
        depths,
        vec![2, 1, 0],
        "depth follows the DAG, not node index"
    );
    assert_eq!(max_depth, 3);
}

#[test]
fn depths_reject_out_of_range_child() {
    let err = sum_product_depths(&[0, 0], &[0, 1], &[5], 2)
        .expect_err("child index 5 >= n_nodes=2 must be rejected");
    assert!(err.contains("outside n_nodes"), "{err}");
}

#[test]
fn depths_reject_cycle() {
    // 0 → 1 → 0 is a cycle; relaxation never converges.
    let err = sum_product_depths(&[0, 1], &[1, 1], &[1, 0], 2)
        .expect_err("a cyclic child graph is not a valid sum-product circuit");
    assert!(err.contains("cycle"), "{err}");
}

#[test]
fn depths_diamond_dag_takes_max_over_shared_child() {
    // A diamond: one leaf feeds two depth-1 nodes, whose shared parent is depth 2.
    //   0=LEAF, 1=SUM(0), 2=SUM(0), 3=PRODUCT(1,2).
    // child_offsets/counts: 1@0..1=[0], 2@1..2=[0], 3@2..4=[1,2].
    let child_offsets = vec![0, 0, 1, 2];
    let child_counts = vec![0, 1, 1, 2];
    let children = vec![0, 0, 1, 2];
    let (depths, max_depth) =
        sum_product_depths(&child_offsets, &child_counts, &children, 4).expect("valid diamond DAG");
    assert_eq!(
        depths,
        vec![0, 1, 1, 2],
        "the two depth-1 nodes share leaf 0; their common parent is 1+max(1,1)=2"
    );
    assert_eq!(max_depth, 3);
}

#[test]
fn ir_program_buffer_layout() {
    let p = sum_product_evaluate("k", "co", "cc", "ch", "w", "lv", "o", 8, 16);
    assert_eq!(p.workgroup_size, [256, 1, 1]);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["k", "co", "cc", "ch", "w", "lv", "o"]);
    // n_nodes-sized
    for i in [0, 1, 2, 5, 6] {
        assert_eq!(p.buffers[i].count(), 8);
    }
    // n_edges-sized
    assert_eq!(p.buffers[3].count(), 16);
    assert_eq!(p.buffers[4].count(), 16);
}

#[test]
fn zero_nodes_traps() {
    let p = sum_product_evaluate("k", "co", "cc", "ch", "w", "lv", "o", 0, 1);
    assert!(p.stats().trap());
}

#[test]
fn zero_edges_leaf_only_circuit_is_valid() {
    let p = sum_product_evaluate("k", "co", "cc", "ch", "w", "lv", "o", 1, 0);
    assert!(!p.stats().trap());
    assert_eq!(p.buffers[3].count(), 1);
    assert_eq!(p.buffers[4].count(), 1);
}

#[test]
fn checked_builder_rejects_zero_nodes() {
    let error = try_sum_product_evaluate("k", "co", "cc", "ch", "w", "lv", "o", 0, 0)
        .expect_err("checked sum-product builder must reject empty node domains");

    assert!(
        error.contains("requires n_nodes > 0"),
        "error should describe the invalid circuit shape: {error}"
    );
}
