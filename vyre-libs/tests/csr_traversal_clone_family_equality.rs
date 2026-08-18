//! Cross-entry-point equality guard for the CSR traversal clone family.
//!
//! Every public entry point below is a thin wrapper over the one owner module
//! `graph::csr_frontier_step`. Two obligations are pinned here, both permanent:
//! 1. Every entry point in the live clone family is verified against canonical
//!    semantic parity invariants and reference witnesses across structured test
//!    graphs, ensuring semantic equivalence without pinning incidental wire hashes.
//! 2. The `*_share_one_*` tests assert the wrappers really do share one
//!    implementation: after erasing the per-entry-point variable prefix, the
//!    queue bound check, the row lookup, the row-striping arithmetic, and the
//!    edge-kind/destination guard chain are literally the same node tree. A
//!    change made for one caller only cannot keep these green.

#![cfg(feature = "graph")]

use vyre_foundation::ir::Program;
use vyre_libs::graph::csr_backward_traverse::csr_backward_traverse;
use vyre_libs::graph::csr_bidirectional::csr_bidirectional;
use vyre_libs::graph::csr_forward_or_changed::{
    csr_forward_or_changed, csr_forward_or_changed_parallel,
};
use vyre_libs::graph::csr_forward_traverse::{
    csr_forward_traverse, csr_forward_traverse_excluding,
};
use vyre_libs::graph::csr_frontier_queue::csr_queue_forward_traverse;
use vyre_libs::graph::csr_queue_delta::{csr_queue_delta_enqueue, csr_queue_delta_strided_enqueue};
use vyre_libs::graph::csr_queue_split::csr_queue_split_low_forward_traverse;
use vyre_libs::graph::csr_queue_strided::csr_queue_strided_forward_traverse;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_test_support::ir_regions::{canonicalize, edge_guard, region};

const NODE_COUNT: u32 = 64;
const EDGE_COUNT: u32 = 7;
const QUEUE_CAPACITY: u32 = 8;
const NEXT_QUEUE_CAPACITY: u32 = 16;
const HIGH_QUEUE_CAPACITY: u32 = 4;
const HIGH_DEGREE_THRESHOLD: u32 = 32;
const ALLOW_MASK: u32 = 1;
/// Above `CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY`, so the strided
/// delta builder emits its grid-stride launch instead of one lane team per slot.
const CAPPED_QUEUE_CAPACITY: u32 = 131_072;

fn shape() -> ProgramGraphShape {
    ProgramGraphShape::new(NODE_COUNT, EDGE_COUNT)
}

fn queue_forward() -> Program {
    csr_queue_forward_traverse(
        "aq",
        "alen",
        "off",
        "tgt",
        "kind",
        "bits",
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

fn queue_strided() -> Program {
    csr_queue_strided_forward_traverse(
        "aq",
        "alen",
        "off",
        "tgt",
        "kind",
        "bits",
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

fn queue_delta(active_queue_capacity: u32) -> Program {
    csr_queue_delta_enqueue(
        "aq",
        "alen",
        "off",
        "tgt",
        "kind",
        "bits",
        "nq",
        "nlen",
        NODE_COUNT,
        EDGE_COUNT,
        active_queue_capacity,
        NEXT_QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

fn queue_delta_strided(active_queue_capacity: u32) -> Program {
    csr_queue_delta_strided_enqueue(
        "aq",
        "alen",
        "off",
        "tgt",
        "kind",
        "bits",
        "nq",
        "nlen",
        NODE_COUNT,
        EDGE_COUNT,
        active_queue_capacity,
        NEXT_QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

fn queue_split() -> Program {
    csr_queue_split_low_forward_traverse(
        "aq",
        "alen",
        "off",
        "tgt",
        "kind",
        "bits",
        "hq",
        "hlen",
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        HIGH_QUEUE_CAPACITY,
        HIGH_DEGREE_THRESHOLD,
        ALLOW_MASK,
    )
}

/// Every public entry point of the clone family, over one shared CSR fixture.
fn entry_points() -> Vec<(&'static str, Program)> {
    vec![
        ("csr_queue_forward_traverse", queue_forward()),
        ("csr_queue_strided_forward_traverse", queue_strided()),
        ("csr_queue_delta_enqueue", queue_delta(QUEUE_CAPACITY)),
        (
            "csr_queue_delta_strided_enqueue",
            queue_delta_strided(QUEUE_CAPACITY),
        ),
        (
            "csr_queue_delta_strided_enqueue.capped",
            queue_delta_strided(CAPPED_QUEUE_CAPACITY),
        ),
        ("csr_queue_split_low_forward_traverse", queue_split()),
        (
            "csr_forward_traverse",
            csr_forward_traverse(shape(), "fin", "fout", ALLOW_MASK),
        ),
        (
            "csr_forward_traverse_excluding",
            csr_forward_traverse_excluding(shape(), "fin", "excluded", "fout", ALLOW_MASK),
        ),
        (
            "csr_backward_traverse",
            csr_backward_traverse(shape(), "fin", "fout", ALLOW_MASK),
        ),
        (
            "csr_bidirectional",
            csr_bidirectional(shape(), "fin", "fout", ALLOW_MASK),
        ),
        (
            "csr_forward_or_changed",
            csr_forward_or_changed(shape(), "fout", "changed", ALLOW_MASK),
        ),
        (
            "csr_forward_or_changed_parallel",
            csr_forward_or_changed_parallel(shape(), "fout", "changed", ALLOW_MASK),
        ),
    ]
}

use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack_u32s, pack_u32_slice as pack_u32s};
use vyre_reference::composition_witness::{
    csr_backward_traverse_witness, csr_bidirectional_step_witness_into,
    csr_forward_or_changed_witness, csr_forward_traverse_witness,
    csr_queue_split_low_forward_witness, csr_queue_strided_forward_witness,
    frontier_to_queue_witness,
};
use vyre_reference::value::Value;

fn eval_queue_forward_program(
    program: &Program,
    queue_capacity: u32,
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    words: usize,
) -> Vec<u32> {
    let mut aq = active_queue.to_vec();
    aq.resize(queue_capacity as usize, 0);
    let inputs = vec![
        Value::from(pack_u32s(&aq)),
        Value::from(pack_u32s(&[queue_len])),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(&vec![0u32; words])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR queue forward traverse reference evaluation must succeed");
    unpack_u32s(&outputs[0].to_bytes())
}

fn eval_queue_delta_program(
    program: &Program,
    active_capacity: u32,
    next_capacity: u32,
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    initial_acc: &[u32],
) -> (Vec<u32>, Vec<u32>, u32) {
    let mut aq = active_queue.to_vec();
    aq.resize(active_capacity as usize, 0);
    let inputs = vec![
        Value::from(pack_u32s(&aq)),
        Value::from(pack_u32s(&[queue_len])),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(initial_acc)),
        Value::from(pack_u32s(&vec![0u32; next_capacity as usize])),
        Value::from(pack_u32s(&[0])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR queue delta reference evaluation must succeed");
    let acc_out = unpack_u32s(&outputs[0].to_bytes());
    let next_q = unpack_u32s(&outputs[1].to_bytes());
    let next_l = unpack_u32s(&outputs[2].to_bytes())[0];
    (acc_out, next_q, next_l)
}

fn eval_queue_split_program(
    program: &Program,
    queue_capacity: u32,
    high_capacity: u32,
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    words: usize,
) -> (Vec<u32>, Vec<u32>, u32) {
    let mut aq = active_queue.to_vec();
    aq.resize(queue_capacity as usize, 0);
    let inputs = vec![
        Value::from(pack_u32s(&aq)),
        Value::from(pack_u32s(&[queue_len])),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(&vec![0u32; words])),
        Value::from(pack_u32s(&vec![0u32; high_capacity as usize])),
        Value::from(pack_u32s(&[0])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR queue split reference evaluation must succeed");
    let frontier_out = unpack_u32s(&outputs[0].to_bytes());
    let high_q = unpack_u32s(&outputs[1].to_bytes());
    let high_l = unpack_u32s(&outputs[2].to_bytes())[0];
    (frontier_out, high_q, high_l)
}

fn eval_program_graph_frontier_step(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
) -> Vec<u32> {
    let words = (node_count as usize).div_ceil(32);
    let nodes = vec![0u32; node_count as usize];
    let node_tags = vec![0u32; node_count as usize];
    let inputs = vec![
        Value::from(pack_u32s(&nodes)),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(&node_tags)),
        Value::from(pack_u32s(frontier_in)),
        Value::from(pack_u32s(&vec![0u32; words])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR frontier step reference evaluation must succeed");
    unpack_u32s(&outputs[0].to_bytes())
}

fn eval_program_graph_frontier_step_excluding(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    excluded_sources: &[u32],
) -> Vec<u32> {
    let words = (node_count as usize).div_ceil(32);
    let nodes = vec![0u32; node_count as usize];
    let node_tags = vec![0u32; node_count as usize];
    let inputs = vec![
        Value::from(pack_u32s(&nodes)),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(&node_tags)),
        Value::from(pack_u32s(frontier_in)),
        Value::from(pack_u32s(excluded_sources)),
        Value::from(pack_u32s(&vec![0u32; words])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR forward excluding reference evaluation must succeed");
    unpack_u32s(&outputs[0].to_bytes())
}

fn eval_program_graph_forward_or_changed(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    initial_frontier: &[u32],
) -> (Vec<u32>, u32) {
    let nodes = vec![0u32; node_count as usize];
    let node_tags = vec![0u32; node_count as usize];
    let inputs = vec![
        Value::from(pack_u32s(&nodes)),
        Value::from(pack_u32s(edge_offsets)),
        Value::from(pack_u32s(edge_targets)),
        Value::from(pack_u32s(edge_kind_mask)),
        Value::from(pack_u32s(&node_tags)),
        Value::from(pack_u32s(initial_frontier)),
        Value::from(pack_u32s(&[0])),
    ];
    let outputs = vyre_reference::reference_eval(program, &inputs)
        .expect("CSR forward or changed reference evaluation must succeed");
    let frontier_out = unpack_u32s(&outputs[0].to_bytes());
    let changed = unpack_u32s(&outputs[1].to_bytes())[0];
    (frontier_out, changed)
}

#[test]
fn entry_points_satisfy_semantic_parity_and_clone_family_equivalence() {
    let eps = entry_points();
    assert_eq!(
        eps.len(),
        12,
        "Fix: entry_points() must cover all 12 canonical CSR traversal wrappers."
    );

    // Construct representative CSR graph fixture matching NODE_COUNT=64 and EDGE_COUNT=7.
    // Nodes 0, 1, 2, 3 have outgoing edges, with targets in both lower word (<32) and upper word (>=32).
    let mut offsets = vec![0u32; NODE_COUNT as usize + 1];
    offsets[0] = 0;
    offsets[1] = 2; // Node 0: edges 0..2 (targets 1, 33)
    offsets[2] = 4; // Node 1: edges 2..4 (targets 2, 63)
    offsets[3] = 6; // Node 2: edges 4..6 (targets 0, 4)
    offsets[4] = 7; // Node 3: edges 6..7 (target 5)
    for i in 5..=NODE_COUNT as usize {
        offsets[i] = 7;
    }
    let targets = vec![1u32, 33, 2, 63, 0, 4, 5];
    let kind_masks = vec![1u32, 1, 2, 1, 1, 1, 2]; // kind 1 allowed by ALLOW_MASK=1, kind 2 filtered
    let words = (NODE_COUNT as usize).div_ceil(32);

    // 1. Bitset-based forward traversal vs witness
    let frontier_in = vec![0b0011_u32, 0]; // nodes {0, 1}
    let fwd_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_forward_traverse")
        .unwrap()
        .1
        .clone();
    let fwd_actual = eval_program_graph_frontier_step(
        &fwd_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &frontier_in,
    );
    let fwd_expected = csr_forward_traverse_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &frontier_in,
        ALLOW_MASK,
    );
    assert_eq!(
        fwd_actual, fwd_expected,
        "Fix: csr_forward_traverse must match the canonical reference witness."
    );

    // 2. Excluding forward traversal vs witness
    let excluded_sources = vec![0b0010_u32, 0]; // exclude node 1
    let fwd_excl_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_forward_traverse_excluding")
        .unwrap()
        .1
        .clone();
    let fwd_excl_actual = eval_program_graph_frontier_step_excluding(
        &fwd_excl_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &frontier_in,
        &excluded_sources,
    );
    let effective_frontier = vec![
        frontier_in[0] & !excluded_sources[0],
        frontier_in[1] & !excluded_sources[1],
    ];
    let fwd_excl_expected = csr_forward_traverse_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &effective_frontier,
        ALLOW_MASK,
    );
    assert_eq!(
        fwd_excl_actual, fwd_excl_expected,
        "Fix: csr_forward_traverse_excluding must match forward step on effective non-excluded frontier."
    );

    // 3. Queue forward & queue strided forward equivalence
    let (active_queue, queue_len) =
        frontier_to_queue_witness(&frontier_in, NODE_COUNT, QUEUE_CAPACITY as usize);
    let q_fwd_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_forward_traverse")
        .unwrap()
        .1
        .clone();
    let q_strided_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_strided_forward_traverse")
        .unwrap()
        .1
        .clone();

    let q_fwd_actual = eval_queue_forward_program(
        &q_fwd_prog,
        QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        words,
    );
    let q_strided_actual = eval_queue_forward_program(
        &q_strided_prog,
        QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        words,
    );
    let q_expected = csr_queue_strided_forward_witness(
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        NODE_COUNT,
        ALLOW_MASK,
    );
    assert_eq!(
        q_fwd_actual, q_expected,
        "Fix: csr_queue_forward_traverse must match the canonical queue witness."
    );
    assert_eq!(
        q_strided_actual, q_fwd_actual,
        "Fix: csr_queue_strided_forward_traverse must be semantically equivalent to csr_queue_forward_traverse."
    );
    assert_eq!(
        q_fwd_actual, fwd_actual,
        "Fix: queue forward traversal must produce identical frontier to bitset forward traversal."
    );

    // 4. Queue delta (scalar, strided, and capped) equivalence
    let initial_acc = vec![0b0010_u32, 0]; // node 1 already known
    let q_delta_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_delta_enqueue")
        .unwrap()
        .1
        .clone();
    let q_delta_strided_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_delta_strided_enqueue")
        .unwrap()
        .1
        .clone();
    let q_delta_capped_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_delta_strided_enqueue.capped")
        .unwrap()
        .1
        .clone();

    let (delta_acc, delta_next_q, delta_next_l) = eval_queue_delta_program(
        &q_delta_prog,
        QUEUE_CAPACITY,
        NEXT_QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        &initial_acc,
    );
    let (delta_s_acc, delta_s_next_q, delta_s_next_l) = eval_queue_delta_program(
        &q_delta_strided_prog,
        QUEUE_CAPACITY,
        NEXT_QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        &initial_acc,
    );
    let (delta_c_acc, delta_c_next_q, delta_c_next_l) = eval_queue_delta_program(
        &q_delta_capped_prog,
        CAPPED_QUEUE_CAPACITY,
        NEXT_QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        &initial_acc,
    );

    assert_eq!(
        delta_s_acc, delta_acc,
        "Fix: strided queue delta must produce identical accumulator to scalar queue delta."
    );
    assert_eq!(
        delta_c_acc, delta_acc,
        "Fix: capped strided queue delta must produce identical accumulator to scalar queue delta."
    );
    assert_eq!(
        delta_s_next_l, delta_next_l,
        "Fix: strided queue delta must produce identical next_len to scalar queue delta."
    );
    assert_eq!(
        delta_c_next_l, delta_next_l,
        "Fix: capped strided queue delta must produce identical next_len to scalar queue delta."
    );
    // The accumulator must be initial_acc | forward_step_destinations
    let expected_delta_acc = vec![
        initial_acc[0] | fwd_actual[0],
        initial_acc[1] | fwd_actual[1],
    ];
    assert_eq!(delta_acc, expected_delta_acc);
    // The newly discovered nodes in delta_next_q must match the newly added bits
    let mut discovered: Vec<u32> = delta_next_q[..delta_next_l as usize].to_vec();
    discovered.sort_unstable();
    let mut expected_discovered = Vec::new();
    for bit in 0..NODE_COUNT {
        let was_set = initial_acc[bit as usize / 32] & (1 << (bit % 32)) != 0;
        let is_set = expected_delta_acc[bit as usize / 32] & (1 << (bit % 32)) != 0;
        if !was_set && is_set {
            expected_discovered.push(bit);
        }
    }
    expected_discovered.sort_unstable();
    assert_eq!(
        discovered, expected_discovered,
        "Fix: queue delta next_queue must contain exactly the newly discovered nodes."
    );

    // 5. Queue split low forward traverse
    let q_split_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_queue_split_low_forward_traverse")
        .unwrap()
        .1
        .clone();
    let (split_fout, split_hq, split_hl) = eval_queue_split_program(
        &q_split_prog,
        QUEUE_CAPACITY,
        HIGH_QUEUE_CAPACITY,
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        words,
    );
    let (exp_split_fout, exp_split_hq, exp_split_hl) = csr_queue_split_low_forward_witness(
        &active_queue,
        queue_len,
        &offsets,
        &targets,
        &kind_masks,
        &vec![0u32; words],
        NODE_COUNT,
        HIGH_QUEUE_CAPACITY as usize,
        HIGH_DEGREE_THRESHOLD,
        ALLOW_MASK,
    );
    assert_eq!(split_fout, exp_split_fout);
    assert_eq!(
        split_hq[..split_hl as usize],
        exp_split_hq[..exp_split_hl as usize]
    );
    assert_eq!(split_hl, exp_split_hl);
    // When degrees are below threshold, split behaves like queue forward
    assert_eq!(split_fout, q_fwd_actual);
    assert_eq!(split_hl, 0);

    // 6. Backward traverse vs witness
    let bwd_in = vec![0, (1u32 << (33 - 32))]; // node 33 is active
    let bwd_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_backward_traverse")
        .unwrap()
        .1
        .clone();
    let bwd_actual = eval_program_graph_frontier_step(
        &bwd_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bwd_in,
    );
    let bwd_expected = csr_backward_traverse_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bwd_in,
        ALLOW_MASK,
    );
    assert_eq!(
        bwd_actual, bwd_expected,
        "Fix: csr_backward_traverse must match canonical backward witness."
    );

    // 7. Bidirectional traverse vs witness & forward | backward union
    let bi_in = vec![0b0001_u32, (1u32 << (33 - 32))]; // nodes {0, 33}
    let bi_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_bidirectional")
        .unwrap()
        .1
        .clone();
    let bi_actual = eval_program_graph_frontier_step(
        &bi_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bi_in,
    );
    let mut bi_expected = Vec::new();
    csr_bidirectional_step_witness_into(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bi_in,
        ALLOW_MASK,
        &mut bi_expected,
    );
    assert_eq!(
        bi_actual, bi_expected,
        "Fix: csr_bidirectional must match bidirectional step witness."
    );
    let bi_fwd = csr_forward_traverse_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bi_in,
        ALLOW_MASK,
    );
    let bi_bwd = csr_backward_traverse_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &bi_in,
        ALLOW_MASK,
    );
    assert_eq!(
        bi_actual,
        vec![bi_fwd[0] | bi_bwd[0], bi_fwd[1] | bi_bwd[1]],
        "Fix: csr_bidirectional must equal fwd | bwd union."
    );

    // 8. Forward or changed (serial & parallel) vs witness & closure equivalence
    // Node 1 reaches 63 (which has no outgoing edges)
    let init_foc = vec![0b0010_u32, 0]; // node {1}
    let foc_serial_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_forward_or_changed")
        .unwrap()
        .1
        .clone();
    let foc_par_prog = eps
        .iter()
        .find(|(n, _)| *n == "csr_forward_or_changed_parallel")
        .unwrap()
        .1
        .clone();

    let (foc_s_out, foc_s_chg) = eval_program_graph_forward_or_changed(
        &foc_serial_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &init_foc,
    );
    let (foc_p_out, foc_p_chg) = eval_program_graph_forward_or_changed(
        &foc_par_prog,
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &init_foc,
    );
    let (exp_foc_out, exp_foc_chg) = csr_forward_or_changed_witness(
        NODE_COUNT,
        &offsets,
        &targets,
        &kind_masks,
        &init_foc,
        ALLOW_MASK,
    );

    assert_eq!(
        foc_s_out, exp_foc_out,
        "Fix: csr_forward_or_changed must match canonical witness on single-step expansion."
    );
    assert_eq!(
        foc_s_chg, exp_foc_chg,
        "Fix: csr_forward_or_changed changed flag must match canonical witness."
    );
    assert_eq!(
        foc_p_out, foc_s_out,
        "Fix: csr_forward_or_changed_parallel must produce identical frontier to serial on single-step expansion."
    );
    assert_eq!(
        foc_p_chg, foc_s_chg,
        "Fix: csr_forward_or_changed_parallel must produce identical changed flag to serial."
    );

    // Both serial and parallel must reach identical transitive closure from node 0
    let mut cur_s = vec![0b0001_u32, 0];
    for _ in 0..NODE_COUNT {
        let (next_s, chg) = eval_program_graph_forward_or_changed(
            &foc_serial_prog,
            NODE_COUNT,
            &offsets,
            &targets,
            &kind_masks,
            &cur_s,
        );
        cur_s = next_s;
        if chg == 0 {
            break;
        }
    }
    let mut cur_p = vec![0b0001_u32, 0];
    for _ in 0..NODE_COUNT {
        let (next_p, chg) = eval_program_graph_forward_or_changed(
            &foc_par_prog,
            NODE_COUNT,
            &offsets,
            &targets,
            &kind_masks,
            &cur_p,
        );
        cur_p = next_p;
        if chg == 0 {
            break;
        }
    }
    assert_eq!(
        cur_p, cur_s,
        "Fix: csr_forward_or_changed serial and parallel must converge to identical closure."
    );
}

#[test]
fn every_queue_entry_point_shares_one_edge_guard_chain() {
    let reference = edge_guard(&queue_forward(), "qt", "_qt_prev");
    for (name, guard) in [
        (
            "csr_queue_strided_forward_traverse",
            edge_guard(&queue_strided(), "qs", "_qs_prev"),
        ),
        (
            "csr_queue_split_low_forward_traverse",
            edge_guard(&queue_split(), "qsl", "_qsl_prev"),
        ),
        (
            "csr_queue_delta_enqueue",
            edge_guard(&queue_delta(QUEUE_CAPACITY), "qd", "qd_old"),
        ),
        (
            "csr_queue_delta_strided_enqueue",
            edge_guard(&queue_delta_strided(QUEUE_CAPACITY), "qds", "qds_old"),
        ),
    ] {
        assert_eq!(
            guard, reference,
            "Fix: {name} must reach its destination bit through the one shared CSR edge guard."
        );
    }
}

#[test]
fn scalar_queue_entry_points_share_one_queue_bound_and_row_lookup() {
    let reference = region(
        &canonicalize(&queue_forward(), "qt"),
        "Ident(\"Q_idx\")",
        "Ident(\"Q_edge_end\")",
    );
    for (name, prefix, program) in [
        ("csr_queue_delta_enqueue", "qd", queue_delta(QUEUE_CAPACITY)),
        ("csr_queue_split_low_forward_traverse", "qsl", queue_split()),
    ] {
        assert_eq!(
            region(
                &canonicalize(&program, prefix),
                "Ident(\"Q_idx\")",
                "Ident(\"Q_edge_end\")",
            ),
            reference,
            "Fix: {name} must take the one shared scalar queue bound check and CSR row lookup."
        );
    }
}

#[test]
fn scalar_queue_entry_points_share_one_edge_walk_loop() {
    assert_eq!(
        region(
            &canonicalize(&queue_delta(QUEUE_CAPACITY), "qd"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"Q_old\")",
        ),
        region(
            &canonicalize(&queue_forward(), "qt"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"_Q_prev\")",
        ),
        "Fix: the scalar queue entry points must walk a queued CSR row through one shared loop."
    );
}

#[test]
fn strided_queue_entry_points_share_one_row_striping_loop() {
    assert_eq!(
        region(
            &canonicalize(&queue_delta_strided(QUEUE_CAPACITY), "qds"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"Q_old\")",
        ),
        region(
            &canonicalize(&queue_strided(), "qs"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"_Q_prev\")",
        ),
        "Fix: the row-strided queue entry points must stripe a CSR row through one shared loop."
    );
}
