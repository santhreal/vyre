//! GPU-IR vs CPU-ref parity for `graph::csr_queue_strided_forward_traverse`, the
//! row-strided load-balanced frontier expansion.
//!
//! A team of `LANES_PER_SOURCE` (32) lanes cooperates on each queued source row,
//! striding over its out-edges; every allowed edge (`kind & allow_mask != 0`)
//! whose destination is in range atomic-ORs the destination bit into the
//! `frontier_out` bitset. Because the output is a bitset written by commutative
//! `atomic_or`, the result is ORDER-INDEPENDENT: the reached-node SET is a
//! deterministic function of the input regardless of lane scheduling, so
//! `reference_eval` parity is well-defined (unlike a queue-APPEND kernel whose
//! output order races). Every shipped test is `cpu_ref`-vs-oracle or a queue
//! proptest; the actual strided-lane IR (queue_idx/edge_lane split, per-team
//! iteration count `ceil(degree/32)`, the edge-lane<degree stride guard, the
//! kind + dst-bound gates, the atomic_or) was never executed. A wrong lane split,
//! a dropped tail iteration, or a missing bound gate all diverge here.
//!
//! Grid: the op atomic-writes, so reference_eval's `force_full_span` covers the
//! max input buffer, but the team layout needs `queue_capacity * 32` lanes; we
//! pass that as an explicit dispatch floor so every (queue_slot, edge_lane) pair
//! fires. Per-lane guards (`queue_idx < queue_len`, `src < node_count`) drop the
//! over-fire.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::csr_queue_strided::{
    csr_queue_strided_forward_traverse, csr_queue_strided_forward_traverse_cpu,
    CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE as LANES_PER_SOURCE,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Drive the real IR through `reference_eval` and return the frontier_out bitset.
/// Buffer binding order: active_queue(0), queue_len(1), edge_offsets(2),
/// edge_targets(3), edge_kind_mask(4), frontier_out(5, the only ReadWrite).
#[allow(clippy::too_many_arguments)]
fn gpu_frontier(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    let edge_count = *edge_offsets
        .last()
        .expect("offsets has node_count+1 entries");
    let queue_capacity = active_queue.len() as u32;
    let program = csr_queue_strided_forward_traverse(
        "active_queue",
        "queue_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "frontier_out",
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    );
    let padded_edges = edge_count.max(1) as usize;
    let mut targets = edge_targets.to_vec();
    targets.resize(padded_edges, 0);
    let mut kind_mask = edge_kind_mask.to_vec();
    kind_mask.resize(padded_edges, 0);
    let words = bitset_words(node_count);

    let floor = queue_capacity * LANES_PER_SOURCE; // one 32-lane team per queue slot
    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(active_queue)),
            Value::from(pack(&[queue_len])),
            Value::from(pack(edge_offsets)),
            Value::from(pack(&targets)),
            Value::from(pack(&kind_mask)),
            Value::from(pack(&vec![0u32; words])),
        ],
        floor,
    )
    .expect("csr_queue_strided reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

/// Random CSR graph + a random active queue (a prefix of `queue_len` valid source
/// nodes) + a random allow_mask. High-degree nodes are included so the strided
/// team must run multiple `ceil(degree/32)` iterations.
fn generated_case(seed: u64) -> (Vec<u32>, u32, Vec<u32>, Vec<u32>, Vec<u32>, u32, u32) {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let node_count = 2 + next() % 64; // 2..=65
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut kind_mask = Vec::new();
    offsets.push(0u32);
    for _ in 0..node_count {
        // Occasionally a high degree (>32) to force multiple strided iterations.
        let degree = if next() % 5 == 0 {
            next() % 80
        } else {
            next() % 6
        };
        for _ in 0..degree {
            targets.push(next() % node_count);
            kind_mask.push(1u32 << (next() % 5));
        }
        offsets.push(targets.len() as u32);
    }
    // Queue: capacity 1..=8, a random number of valid source nodes then padding.
    let queue_capacity = 1 + next() % 8;
    let mut queue = Vec::with_capacity(queue_capacity as usize);
    for _ in 0..queue_capacity {
        queue.push(next() % node_count);
    }
    let queue_len = 1 + next() % queue_capacity; // 1..=capacity
    let allow_mask = 1u32 << (next() % 5) | 1u32 << (next() % 5);
    (
        queue, queue_len, offsets, targets, kind_mask, node_count, allow_mask,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn ir_matches_cpu_ref_over_random_queues(seed in any::<u64>()) {
        let (queue, queue_len, offsets, targets, kind_mask, node_count, allow_mask) =
            generated_case(seed);
        let expected = csr_queue_strided_forward_traverse_cpu(
            &queue, queue_len, &offsets, &targets, &kind_mask, node_count, allow_mask,
        );
        let got = gpu_frontier(
            &queue, queue_len, &offsets, &targets, &kind_mask, node_count, allow_mask,
        );
        prop_assert_eq!(
            got, expected,
            "strided-queue frontier IR diverged: node_count={}, queue={:?}, queue_len={}, allow_mask={:#x}",
            node_count, queue, queue_len, allow_mask
        );
    }
}

/// Deterministic anchors: a high-degree source spanning multiple strided
/// iterations, the queue_len gate (padding slots ignored), and allow_mask
/// filtering.
#[test]
fn ir_matches_cpu_ref_on_boundary_queues() {
    // Node 0 has 40 out-edges (> 32 lanes -> 2 strided iterations), each to a
    // distinct destination 1..=40; node_count 41. Only node 0 is queued.
    let node_count = 41u32;
    let mut offsets = vec![0u32];
    let mut targets = Vec::new();
    let mut kind_mask = Vec::new();
    for d in 1..=40u32 {
        targets.push(d % node_count);
        kind_mask.push(1u32);
    }
    offsets.push(targets.len() as u32); // node 0
    for _ in 1..node_count {
        offsets.push(targets.len() as u32); // other nodes have no edges
    }
    let queue = vec![0u32, 7, 7, 7]; // capacity 4, but queue_len 1 -> only node 0
    let expected = csr_queue_strided_forward_traverse_cpu(
        &queue,
        1,
        &offsets,
        &targets,
        &kind_mask,
        node_count,
        0xFFFF_FFFF,
    );
    // Every destination 1..=40 reached; 40 % 41 stays in range.
    let mut want_bits = 0u32;
    for d in 1..=40u32 {
        assert_eq!(
            expected[(d / 32) as usize] >> (d % 32) & 1,
            1,
            "cpu_ref reaches {d}"
        );
        want_bits += 1;
    }
    assert_eq!(want_bits, 40);
    assert_eq!(
        gpu_frontier(
            &queue,
            1,
            &offsets,
            &targets,
            &kind_mask,
            node_count,
            0xFFFF_FFFF
        ),
        expected,
        "high-degree strided expansion (2 iterations) must match"
    );

    // allow_mask filtering: node 0 -> node 1 via kind bit 2; a mask selecting bit
    // 4 reaches nothing; bit 2 reaches node 1.
    let offsets = vec![0u32, 1, 1];
    let targets = vec![1u32];
    let kind_mask = vec![1u32 << 2];
    let queue = vec![0u32];
    let dropped = csr_queue_strided_forward_traverse_cpu(
        &queue,
        1,
        &offsets,
        &targets,
        &kind_mask,
        2,
        1 << 4,
    );
    assert_eq!(
        dropped,
        vec![0u32],
        "cpu_ref: mask mismatch reaches nothing"
    );
    assert_eq!(
        gpu_frontier(&queue, 1, &offsets, &targets, &kind_mask, 2, 1 << 4),
        dropped
    );
    let fired = csr_queue_strided_forward_traverse_cpu(
        &queue,
        1,
        &offsets,
        &targets,
        &kind_mask,
        2,
        1 << 2,
    );
    assert_eq!(
        fired,
        vec![0b10u32],
        "cpu_ref: matching mask reaches node 1"
    );
    assert_eq!(
        gpu_frontier(&queue, 1, &offsets, &targets, &kind_mask, 2, 1 << 2),
        fired
    );
}
