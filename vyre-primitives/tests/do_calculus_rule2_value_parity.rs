//! Value parity for the three `graph::do_calculus` surgery IR PROGRAMS, each run through
//! `reference_eval` must match its own `_cpu` oracle.
//!
//! WHY: the do-calculus builders had IR-validity checks (via graph_builders_emit_valid_ir) and
//! `_cpu` oracles, but their actual IR was never run through `reference_eval` for VALUE, the exact
//! union_find/tensor_scc/matroid risk class (IR-validity ≠ IR-correctness). Covers:
//! - `do_rule2_reverse_incoming`: per-cell map (`t = InvocationId`, [256,1,1]): keep the original
//!   edge unless the destination column is a treated node, OR-in the reversed edge when the source
//!   row is treated (diagonal untouched).
//! - `do_intervention_delete_incoming`: per-cell map: zero the whole adjacency column of any
//!   intervened node (sever its incoming edges).
//! - `do_rule3_subgraph`: the one do-calculus op that is NOT a per-cell map: a lane-0-serial
//!   compaction (prefix scan of the kept indices) + gather producing a dense `k × k` subgraph with
//!   stride `k ≠ n`. This differential locks the compaction order, the kept-index map, and the
//!   stride-k gather against the CPU oracle.
#![cfg(all(feature = "all-lego", feature = "cpu-parity"))]

use vyre_primitives::graph::do_calculus::{
    do_intervention_delete_incoming, do_intervention_delete_incoming_cpu,
    do_rule2_reverse_incoming, do_rule2_reverse_incoming_cpu, do_rule3_subgraph,
    do_rule3_subgraph_cpu,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn run_ir(adjacency: &[u32], treatment_mask: &[u32], n: u32) -> Vec<u32> {
    let program = do_rule2_reverse_incoming("adjacency", "treatment_mask", "out_adjacency", n);
    let cells = (n * n) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(adjacency)),
            Value::from(pack(treatment_mask)),
            Value::from(pack(&vec![0u32; cells])),
        ],
    )
    .expect("do_rule2_reverse_incoming reference evaluation must succeed");
    let idx = vyre_reference::output_index(&program, "out_adjacency")
        .expect("Fix: do_rule2_reverse_incoming must declare output `out_adjacency`");
    unpack(&outputs[idx].to_bytes())[..cells].to_vec()
}

fn run_intervention_ir(adjacency: &[u32], intervention_mask: &[u32], n: u32) -> Vec<u32> {
    let program =
        do_intervention_delete_incoming("adjacency", "intervention_mask", "out_adjacency", n);
    let cells = (n * n) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(adjacency)),
            Value::from(pack(intervention_mask)),
            Value::from(pack(&vec![0u32; cells])),
        ],
    )
    .expect("do_intervention_delete_incoming reference evaluation must succeed");
    let idx = vyre_reference::output_index(&program, "out_adjacency")
        .expect("Fix: do_intervention_delete_incoming must declare output `out_adjacency`");
    unpack(&outputs[idx].to_bytes())[..cells].to_vec()
}

#[test]
fn rule2_ir_matches_cpu_over_generated_graphs() {
    let mut state = 0x0BAD_F00Du32;
    let mut next = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    };
    let mut reversed_edge_cases = 0u32;
    for case in 0..400u32 {
        let n = 2 + next(&mut state) % 6; // 2..=7
        let cells = (n * n) as usize;
        let adjacency: Vec<u32> = (0..cells).map(|_| next(&mut state) & 1).collect();
        let treatment_mask: Vec<u32> = (0..n).map(|_| next(&mut state) & 1).collect();

        let ir = run_ir(&adjacency, &treatment_mask, n);
        let cpu = do_rule2_reverse_incoming_cpu(&adjacency, &treatment_mask, n);
        // Count cases where the transform actually changed the graph (a treated node exists AND an
        // off-diagonal edge moved) so the differential isn't a vacuous identity check.
        if ir != adjacency && treatment_mask.iter().any(|&t| t != 0) {
            reversed_edge_cases += 1;
        }
        assert_eq!(
            ir, cpu,
            "case {case} (n={n}): rule2 IR {ir:?} != cpu oracle {cpu:?} \
             (adjacency={adjacency:?}, treatment_mask={treatment_mask:?})"
        );
    }
    assert!(
        reversed_edge_cases > 100,
        "only {reversed_edge_cases}/400 cases changed the graph, strengthen the input distribution \
         so the reversal/deletion rule is actually exercised"
    );
}

#[test]
fn rule2_ir_reverses_incoming_edge_of_treated_node() {
    // 2 nodes, edge 0->1, node 1 treated. Rule 2 reverses incoming edges of the treated node:
    // the 0->1 edge (incoming to treated node 1) becomes 1->0. Diagonal untouched; node 0 untreated
    // so its original outgoing edge to a treated column (col 1 treated) is DELETED.
    let n = 2u32;
    let adjacency = vec![0u32, 1, 0, 0]; // 0->1
    let treatment_mask = vec![0u32, 1]; // node 1 treated
    let ir = run_ir(&adjacency, &treatment_mask, n);
    let cpu = do_rule2_reverse_incoming_cpu(&adjacency, &treatment_mask, n);
    // Expected: cell (0,1) original edge deleted (col 1 treated) => 0; cell (1,0) gets the reversed
    // edge adjacency[(0,1)] = 1 (row 1 treated) => 1. Diagonal 0.
    assert_eq!(
        cpu,
        vec![0, 0, 1, 0],
        "sanity: oracle reverses the incoming edge"
    );
    assert_eq!(
        ir, cpu,
        "rule2 IR must reverse the treated node's incoming edge like the oracle"
    );
}

/// Run the Rule-3 subgraph-extraction IR and return `(reduced_kxk, kept_k, k)` truncated to the
/// live `k × k` / `k` prefixes, exactly as the CPU oracle lays them out.
fn run_rule3_ir(adjacency: &[u32], keep_mask: &[u32], n: u32) -> (Vec<u32>, Vec<u32>, u32) {
    let program = do_rule3_subgraph("adjacency", "keep_mask", "reduced", "kept", "kept_len", n);
    let cells = (n * n) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(adjacency)),
            Value::from(pack(keep_mask)),
            Value::from(pack(&vec![0u32; cells])),
            Value::from(pack(&vec![0u32; n as usize])),
            Value::from(pack(&[0u32])),
        ],
    )
    .expect("do_rule3_subgraph reference evaluation must succeed");
    let reduced_idx = vyre_reference::output_index(&program, "reduced")
        .expect("Fix: do_rule3_subgraph must declare output `reduced`");
    let kept_idx = vyre_reference::output_index(&program, "kept")
        .expect("Fix: do_rule3_subgraph must declare output `kept`");
    let len_idx = vyre_reference::output_index(&program, "kept_len")
        .expect("Fix: do_rule3_subgraph must declare output `kept_len`");
    let k = unpack(&outputs[len_idx].to_bytes())[0];
    let ku = k as usize;
    let reduced = unpack(&outputs[reduced_idx].to_bytes())[..ku * ku].to_vec();
    let kept = unpack(&outputs[kept_idx].to_bytes())[..ku].to_vec();
    (reduced, kept, k)
}

#[test]
fn rule3_ir_matches_cpu_over_generated_graphs() {
    let mut state = 0x7EED_1234u32;
    let mut next = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    };
    let mut nontrivial_extractions = 0u32;
    for case in 0..400u32 {
        let n = 2 + next(&mut state) % 6; // 2..=7
        let cells = (n * n) as usize;
        let adjacency: Vec<u32> = (0..cells).map(|_| next(&mut state) & 1).collect();
        // Bias keep_mask so a mix of full/partial/empty subsets is exercised (not almost-always-all).
        let keep_mask: Vec<u32> = (0..n)
            .map(|_| next(&mut state) % 3)
            .map(|m| m & 1)
            .collect();

        let (ir_reduced, ir_kept, ir_k) = run_rule3_ir(&adjacency, &keep_mask, n);
        let (cpu_reduced, cpu_kept) = do_rule3_subgraph_cpu(&adjacency, &keep_mask, n);
        let cpu_k = cpu_kept.len() as u32;

        // A partial extraction (some but not all nodes kept) where an edge actually survives is the
        // interesting case, count it so the differential isn't dominated by all-keep identity or
        // empty-keep no-ops.
        if cpu_k > 0 && (cpu_k as usize) < n as usize && cpu_reduced.iter().any(|&e| e != 0) {
            nontrivial_extractions += 1;
        }
        assert_eq!(
            ir_k, cpu_k,
            "case {case} (n={n}): rule3 IR kept_len {ir_k} != cpu k {cpu_k} \
             (keep_mask={keep_mask:?})"
        );
        assert_eq!(
            ir_kept, cpu_kept,
            "case {case} (n={n}): rule3 IR kept-index map {ir_kept:?} != cpu {cpu_kept:?} \
             (keep_mask={keep_mask:?})"
        );
        assert_eq!(
            ir_reduced, cpu_reduced,
            "case {case} (n={n}): rule3 IR reduced {ir_reduced:?} != cpu {cpu_reduced:?} \
             (adjacency={adjacency:?}, keep_mask={keep_mask:?})"
        );
    }
    assert!(
        nontrivial_extractions > 100,
        "only {nontrivial_extractions}/400 cases were partial edge-preserving extractions. \
         strengthen the keep_mask distribution so the k×k compaction+gather is actually exercised"
    );
}

#[test]
fn rule3_ir_extracts_dense_subgraph_with_stride_k() {
    // 4 nodes, keep {0,2,3} (drop node 1). The result is a dense 3×3 block laid out with STRIDE 3
    // (not 4): reduced[new_i*3+new_j] = adjacency[kept[new_i]*4 + kept[new_j]].
    let n = 4u32;
    // adjacency (row-major 4×4): edges 0->2, 0->3, 2->3, 3->0, plus a dropped-node edge 1->2.
    let adjacency = vec![
        0u32, 0, 1, 1, // row 0
        0, 0, 1, 0, // row 1 (dropped)
        0, 0, 0, 1, // row 2
        1, 0, 0, 0, // row 3
    ];
    let keep_mask = vec![1u32, 0, 1, 1]; // keep 0,2,3
    let (ir_reduced, ir_kept, ir_k) = run_rule3_ir(&adjacency, &keep_mask, n);
    let (cpu_reduced, cpu_kept) = do_rule3_subgraph_cpu(&adjacency, &keep_mask, n);
    // kept = [0,2,3]; k=3. reduced rows/cols indexed by [0,2,3]:
    //   (0,0)=adj[0,0]=0 (0,2)=adj[0,2]=1 (0,3)=adj[0,3]=1
    //   (2,0)=adj[2,0]=0 (2,2)=adj[2,2]=0 (2,3)=adj[2,3]=1
    //   (3,0)=adj[3,0]=1 (3,2)=adj[3,2]=0 (3,3)=adj[3,3]=0
    assert_eq!(ir_k, 3, "three nodes retained");
    assert_eq!(cpu_kept, vec![0, 2, 3], "sanity: oracle kept-index map");
    assert_eq!(
        cpu_reduced,
        vec![0, 1, 1, 0, 0, 1, 1, 0, 0],
        "sanity: oracle extracts the stride-3 dense subgraph (dropped node 1's row/col gone)"
    );
    assert_eq!(ir_kept, cpu_kept, "rule3 IR kept map must match the oracle");
    assert_eq!(
        ir_reduced, cpu_reduced,
        "rule3 IR must extract the same stride-k dense subgraph as the oracle"
    );
}

#[test]
fn rule3_ir_handles_empty_and_full_keep_masks() {
    let n = 3u32;
    let adjacency = vec![0u32, 1, 0, 0, 0, 1, 1, 0, 0];
    // Empty keep: k=0, no reduced cells, no kept indices.
    let (ir_reduced, ir_kept, ir_k) = run_rule3_ir(&adjacency, &[0, 0, 0], n);
    assert_eq!(ir_k, 0, "empty keep_mask retains no nodes");
    assert!(ir_kept.is_empty() && ir_reduced.is_empty());
    // Full keep: k=n, the subgraph is the whole graph (stride k == n here).
    let (ir_reduced_full, ir_kept_full, ir_k_full) = run_rule3_ir(&adjacency, &[1, 1, 1], n);
    let (cpu_reduced_full, cpu_kept_full) = do_rule3_subgraph_cpu(&adjacency, &[1, 1, 1], n);
    assert_eq!(ir_k_full, n, "full keep_mask retains every node");
    assert_eq!(ir_kept_full, cpu_kept_full);
    assert_eq!(
        ir_reduced_full, cpu_reduced_full,
        "full keep must reproduce the original adjacency"
    );
    assert_eq!(
        ir_reduced_full, adjacency,
        "full keep is the identity extraction"
    );
}

#[test]
fn intervention_ir_matches_cpu_over_generated_graphs() {
    let mut state = 0x1357_9BDFu32;
    let mut next = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    };
    let mut deleted_edge_cases = 0u32;
    for case in 0..400u32 {
        let n = 2 + next(&mut state) % 6; // 2..=7
        let cells = (n * n) as usize;
        let adjacency: Vec<u32> = (0..cells).map(|_| next(&mut state) & 1).collect();
        let intervention_mask: Vec<u32> = (0..n).map(|_| next(&mut state) & 1).collect();

        let ir = run_intervention_ir(&adjacency, &intervention_mask, n);
        let cpu = do_intervention_delete_incoming_cpu(&adjacency, &intervention_mask, n);
        // Count cases where the intervention actually deleted an incoming edge (a node is
        // intervened AND its column held a live edge) so this isn't a vacuous identity check.
        if ir != adjacency && intervention_mask.iter().any(|&m| m != 0) {
            deleted_edge_cases += 1;
        }
        assert_eq!(
            ir, cpu,
            "case {case} (n={n}): intervention IR {ir:?} != cpu oracle {cpu:?} \
             (adjacency={adjacency:?}, intervention_mask={intervention_mask:?})"
        );
    }
    assert!(
        deleted_edge_cases > 100,
        "only {deleted_edge_cases}/400 cases deleted an incoming edge, strengthen the input \
         distribution so the do(x) column-deletion rule is actually exercised"
    );
}

#[test]
fn intervention_ir_zeroes_the_full_column_of_the_intervened_node() {
    // do(X=x) severs every incoming edge to the intervened node: its entire adjacency COLUMN goes
    // to zero, while all other columns (including the intervened node's own outgoing row) survive.
    let n = 3u32;
    // Fully-connected off-diagonal graph: every i->j (i!=j) is an edge.
    let adjacency = vec![
        0u32, 1, 1, //
        1, 0, 1, //
        1, 1, 0,
    ];
    let intervention_mask = vec![0u32, 1, 0]; // intervene on node 1
    let ir = run_intervention_ir(&adjacency, &intervention_mask, n);
    let cpu = do_intervention_delete_incoming_cpu(&adjacency, &intervention_mask, n);
    // Column 1 (cells (0,1),(1,1),(2,1)) zeroed; node 1's OUTGOING row 1 (cells (1,0),(1,2)) intact.
    assert_eq!(
        cpu,
        vec![0, 0, 1, 1, 0, 1, 1, 0, 0],
        "sanity: oracle zeroes only the intervened node's incoming column"
    );
    assert_eq!(
        ir, cpu,
        "intervention IR must sever every incoming edge of the intervened node like the oracle"
    );
}
