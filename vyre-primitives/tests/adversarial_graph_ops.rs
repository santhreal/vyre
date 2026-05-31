//! Failure-oriented adversarial integration tests for graph primitives.
//!
//! Coverage: csr_forward_traverse, csr_backward_traverse, toposort,
//! scc_decompose, path_reconstruct  -  hostile boundaries, empty graphs,
//! edge-kind diversity (M8), malformed CSR, cross-word bitsets.
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use vyre_primitives::bitset::bitset_words;
use vyre_primitives::graph::csr_backward_traverse::cpu_ref as bwd_cpu_ref;
use vyre_primitives::graph::csr_forward_traverse::cpu_ref as fwd_cpu_ref;
use vyre_primitives::graph::csr_frontier_queue::{
    frontier_to_queue_cpu, frontier_words_to_queue_parallel,
};
use vyre_primitives::graph::path_reconstruct::cpu_ref as path_cpu_ref;
use vyre_primitives::graph::scc_decompose::cpu_ref as scc_cpu_ref;
use vyre_primitives::graph::toposort::{toposort, ToposortError};
use vyre_reference::value::Value;

fn pack_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn unpack_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
        .collect()
}

// ---------------------------------------------------------------------------
// csr_forward_traverse
// ---------------------------------------------------------------------------

#[test]
fn forward_empty_graph() {
    let got = fwd_cpu_ref(0, &[0], &[], &[], &[], 0xFFFF_FFFF);
    assert!(got.is_empty());
}

#[test]
fn forward_single_node_no_edges() {
    let got = fwd_cpu_ref(1, &[0, 0], &[0], &[0], &[0b0001], 0xFFFF_FFFF);
    assert_eq!(got, vec![0]);
}

#[test]
fn forward_self_loops_only() {
    let got = fwd_cpu_ref(2, &[0, 1, 2], &[0, 1], &[1, 1], &[0b0011], 0xFFFF_FFFF);
    assert_eq!(got, vec![0b0011]);
}

#[test]
fn forward_disconnected_components() {
    let got = fwd_cpu_ref(
        4,
        &[0, 1, 1, 2, 2],
        &[1, 3],
        &[1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(got, vec![0b0010]);
}

#[test]
fn forward_max_node_count_cross_word() {
    let mut offsets = vec![0u32; 66];
    offsets[64] = 0;
    offsets[65] = 1;
    let mut frontier = vec![0u32; 3];
    frontier[2] = 1;
    let got = fwd_cpu_ref(65, &offsets, &[0], &[1], &frontier, 0xFFFF_FFFF);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], 1);
    assert_eq!(got[1], 0);
    assert_eq!(got[2], 0);
}

#[test]
fn forward_edge_mask_filters_all() {
    let got = fwd_cpu_ref(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[0b01, 0b01, 0b01, 0b01],
        &[0b0001],
        0b10,
    );
    assert_eq!(got, vec![0]);
}

#[test]
fn forward_edge_kind_diversity_m8() {
    // DOMINANCE=0x01, ASSIGNMENT=0x02. Mask only DOMINANCE.
    let got = fwd_cpu_ref(4, &[0, 2, 2, 2, 2], &[1, 2], &[0x01, 0x02], &[0b0001], 0x01);
    assert_eq!(
        got,
        vec![0b0010],
        "broken impl ignoring kind_mask would produce 0b0110"
    );
}

// ---------------------------------------------------------------------------
// csr_frontier_queue
// ---------------------------------------------------------------------------

#[test]
fn word_parallel_frontier_queue_matches_cpu_and_ignores_tail_bits() {
    let node_count = 70;
    let queue_capacity = 8;
    let frontier = [
        (1_u32 << 0) | (1_u32 << 1) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 5) | (1_u32 << 31),
    ];
    let (expected_queue, expected_seen) =
        frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
    let program = frontier_words_to_queue_parallel(
        "frontier",
        "queue",
        "queue_len",
        node_count,
        queue_capacity,
    );

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_words(&frontier)),
            Value::from(vec![
                0_u8;
                queue_capacity as usize * std::mem::size_of::<u32>()
            ]),
            Value::from(pack_words(&[0])),
        ],
    )
    .expect("word-level frontier queue materializer should reference-evaluate");

    let mut queue = unpack_words(&outputs[0].to_bytes());
    queue.truncate(expected_queue.len());
    queue.sort_unstable();
    let mut expected_sorted = expected_queue;
    expected_sorted.sort_unstable();

    assert_eq!(unpack_words(&outputs[1].to_bytes()), vec![expected_seen]);
    assert_eq!(
        queue, expected_sorted,
        "word-level materializer must enqueue exactly in-range active nodes"
    );
    assert!(
        !queue.contains(&95),
        "tail bits beyond node_count must not enter the active queue"
    );
}

#[test]
fn word_parallel_frontier_queue_matches_cpu_len_across_2048_generated_frontiers() {
    for case in 0u32..2048 {
        let node_count = 1 + case.wrapping_mul(17) % 127;
        let queue_capacity = 1 + case.wrapping_mul(13) % 48;
        let words = bitset_words(node_count) as usize;
        let mut state = 0x9E37_79B9u32 ^ case.wrapping_mul(0x85EB_CA6B);
        let mut frontier = vec![0u32; words];

        for word in &mut frontier {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *word = match case % 5 {
                0 => 0,
                1 => state & 0x0101_0101,
                2 => state & 0x1111_1111,
                3 => state,
                _ => state | 0x8000_0001,
            };
        }

        let tail_bits = node_count % 32;
        if tail_bits != 0 {
            let in_range_mask = (1u32 << tail_bits) - 1;
            let tail_mask = !in_range_mask;
            if case % 7 == 0 {
                frontier[words - 1] &= !in_range_mask;
                frontier[words - 1] |= tail_mask;
            } else if case % 3 == 0 {
                frontier[words - 1] |= tail_mask & 0xA5A5_A5A5;
            }
        }

        let (expected_queue, expected_seen) =
            frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
        let active_nodes: Vec<u32> = (0..node_count)
            .filter(|&node| {
                let word = frontier[node as usize / 32];
                (word & (1u32 << (node % 32))) != 0
            })
            .collect();
        let program = frontier_words_to_queue_parallel(
            "frontier",
            "queue",
            "queue_len",
            node_count,
            queue_capacity,
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack_words(&frontier)),
                Value::from(vec![
                    0_u8;
                    queue_capacity as usize * std::mem::size_of::<u32>()
                ]),
                Value::from(pack_words(&[0])),
            ],
        )
        .unwrap_or_else(|error| {
            panic!(
                "case {case}: word-level frontier queue materializer failed reference_eval: {error}"
            )
        });

        let mut queue = unpack_words(&outputs[0].to_bytes());
        queue.truncate(expected_queue.len());
        queue.sort_unstable();
        let mut expected_sorted = expected_queue;
        expected_sorted.sort_unstable();

        assert_eq!(
            unpack_words(&outputs[1].to_bytes()),
            vec![expected_seen],
            "case {case}: queue_len must report in-range active nodes"
        );
        if expected_seen as usize <= queue_capacity as usize {
            assert_eq!(
                queue, expected_sorted,
                "case {case}: queue contents must match CPU oracle without overflow"
            );
        } else {
            assert_eq!(
                queue.len(),
                queue_capacity as usize,
                "case {case}: overflow must fill the bounded queue"
            );
            assert!(
                queue.windows(2).all(|pair| pair[0] != pair[1]),
                "case {case}: overflow queue must not duplicate an active node"
            );
            assert!(
                queue
                    .iter()
                    .all(|node| active_nodes.binary_search(node).is_ok()),
                "case {case}: overflow queue must contain only active in-range nodes"
            );
        }
        assert!(
            queue.iter().all(|&node| node < node_count),
            "case {case}: tail bit escaped into queue for node_count={node_count}"
        );
    }
}

// ---------------------------------------------------------------------------
// csr_backward_traverse
// ---------------------------------------------------------------------------

#[test]
fn backward_empty_graph() {
    let got = bwd_cpu_ref(0, &[0], &[], &[], &[], 0xFFFF_FFFF);
    assert!(got.is_empty());
}

#[test]
fn backward_single_node_no_edges() {
    let got = bwd_cpu_ref(1, &[0, 0], &[0], &[0], &[0b0001], 0xFFFF_FFFF);
    assert_eq!(got, vec![0]);
}

#[test]
fn backward_self_loops_only() {
    let got = bwd_cpu_ref(2, &[0, 1, 2], &[0, 1], &[1, 1], &[0b0001], 0xFFFF_FFFF);
    assert_eq!(got, vec![0b0001]);
}

#[test]
fn backward_disconnected_components() {
    let got = bwd_cpu_ref(
        4,
        &[0, 1, 1, 2, 2],
        &[1, 3],
        &[1, 1],
        &[0b1000],
        0xFFFF_FFFF,
    );
    assert_eq!(got, vec![0b0100]);
}

#[test]
fn backward_edge_kind_diversity_m8() {
    let got = bwd_cpu_ref(4, &[0, 2, 2, 2, 2], &[1, 2], &[0x01, 0x02], &[0b0010], 0x01);
    assert_eq!(
        got,
        vec![0b0001],
        "broken impl ignoring kind_mask would produce 0"
    );
}

// ---------------------------------------------------------------------------
// toposort
// ---------------------------------------------------------------------------

#[test]
fn toposort_single_node() {
    assert_eq!(toposort(1, &[]), Ok(vec![0]));
}

#[test]
fn toposort_self_loops_rejected() {
    let err = toposort(3, &[(0, 0), (1, 1), (2, 2)]).expect_err("self-loops are cycles");
    assert!(matches!(err, ToposortError::Cycle { .. }));
}

#[test]
fn toposort_disconnected_components() {
    let got = toposort(4, &[(0, 1), (2, 3)]).unwrap();
    assert_eq!(got.len(), 4);
    let pos = |v: u32| got.iter().position(|&x| x == v).unwrap();
    assert!(pos(1) < pos(0));
    assert!(pos(3) < pos(2));
}

#[test]
fn toposort_large_graph_cycle_diagnostic() {
    let mut edges: Vec<(u32, u32)> = (0..99).map(|i| (i, i + 1)).collect();
    edges.push((99, 50));
    let err = toposort(100, &edges).expect_err("cycle must be detected");
    match err {
        ToposortError::Cycle { node } => {
            assert!((50..=99).contains(&node));
        }
        other => panic!("expected Cycle, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// scc_decompose
// ---------------------------------------------------------------------------

#[test]
fn scc_empty_graph() {
    let out = scc_cpu_ref(0, &[], &[], &[], 0);
    assert!(out.is_empty());
}

#[test]
fn scc_self_loop() {
    let out = scc_cpu_ref(1, &[0b0001], &[0b0001], &[u32::MAX; 1], 0);
    assert_eq!(out, vec![0]);
}

#[test]
fn scc_disconnected_components() {
    let forward = vec![0b0101];
    let backward = vec![0b0101];
    let comp_in = vec![u32::MAX; 4];
    let out = scc_cpu_ref(4, &forward, &backward, &comp_in, 0);
    assert_eq!(out[0], 0);
    assert_eq!(out[1], u32::MAX);
    assert_eq!(out[2], 0);
    assert_eq!(out[3], u32::MAX);
}

#[test]
fn scc_multi_word_cross_boundary() {
    let mut forward = vec![0u32; 3];
    let mut backward = vec![0u32; 3];
    forward[1] = 1; // node 32
    forward[2] = 1; // node 64
    backward[1] = 1;
    backward[2] = 1;
    let comp_in = vec![u32::MAX; 65];
    let out = scc_cpu_ref(65, &forward, &backward, &comp_in, 42);
    assert_eq!(out[32], 42);
    assert_eq!(out[64], 42);
    assert_eq!(out[0], u32::MAX);
    assert_eq!(out[31], u32::MAX);
    assert_eq!(out[33], u32::MAX);
    assert_eq!(out[63], u32::MAX);
}

// ---------------------------------------------------------------------------
// path_reconstruct
// ---------------------------------------------------------------------------

#[test]
fn path_parent_self_loops() {
    let mut scratch = Vec::with_capacity(4);
    let len = path_cpu_ref(&[0, 1], 1, 4, &mut scratch);
    assert_eq!(len, 1);
    assert_eq!(scratch[0], 1);
    assert_eq!(&scratch[1..], &[0, 0, 0]);
}

#[test]
fn path_deep_chain() {
    let parent = &[0, 0, 1, 2, 3];
    let mut scratch = Vec::with_capacity(8);
    let len = path_cpu_ref(parent, 4, 8, &mut scratch);
    assert_eq!(len, 5);
    assert_eq!(&scratch[..5], &[4, 3, 2, 1, 0]);
}

#[test]
fn path_target_not_in_parent() {
    let mut scratch = Vec::with_capacity(4);
    let len = path_cpu_ref(&[0, 0, 1], 5, 4, &mut scratch);
    assert_eq!(len, 1);
    assert_eq!(scratch[0], 5);
}

#[test]
fn path_max_depth_zero() {
    let mut scratch = Vec::with_capacity(4);
    let len = path_cpu_ref(&[0, 0, 1, 2], 3, 0, &mut scratch);
    assert_eq!(len, 0);
    assert!(scratch.is_empty());
}

#[test]
fn path_max_depth_one() {
    let mut scratch = Vec::with_capacity(4);
    let len = path_cpu_ref(&[0, 0, 1, 2], 3, 1, &mut scratch);
    assert_eq!(len, 1);
    assert_eq!(scratch[0], 3);
}
