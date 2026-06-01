//! Adversarial tests for packed frontier queue materialization with fused output clear.

#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use vyre_primitives::graph::csr_frontier_queue::{
    frontier_to_queue_cpu, frontier_words_to_queue_clear_out_parallel,
};
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

#[test]
fn word_parallel_frontier_queue_clear_out_matches_queue_and_zeros_output() {
    let node_count = 70;
    let queue_capacity = 8;
    let frontier = [
        (1_u32 << 0) | (1_u32 << 1) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 5) | (1_u32 << 31),
    ];
    let frontier_out_seed = [u32::MAX, 0xA5A5_A5A5, 0x8000_0001];
    let (expected_queue, expected_seen) =
        frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
    let program = frontier_words_to_queue_clear_out_parallel(
        "frontier",
        "queue",
        "queue_len",
        "frontier_out",
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
            Value::from(pack_words(&frontier_out_seed)),
        ],
    )
    .expect("word-level clear-out frontier queue materializer should reference-evaluate");

    let mut queue = unpack_words(&outputs[0].to_bytes());
    queue.truncate(expected_queue.len());
    queue.sort_unstable();
    let mut expected_sorted = expected_queue;
    expected_sorted.sort_unstable();

    assert_eq!(unpack_words(&outputs[1].to_bytes()), vec![expected_seen]);
    assert_eq!(queue, expected_sorted);
    assert_eq!(unpack_words(&outputs[2].to_bytes()), vec![0, 0, 0]);
}
