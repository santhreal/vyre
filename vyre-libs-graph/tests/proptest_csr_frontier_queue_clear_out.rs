//! The fused output clear in the packed-word frontier queue materializer.
//!
//! `frontier_words_to_queue_clear_out_parallel` folds a full-frontier reset
//! into the packed-word scan, so one program writes three buffers: the queue,
//! the active count, and a zeroed `frontier_out`. Three properties can break
//! independently of each other, and the fused write is the one with no other
//! subject in this crate: the shape check in `emitted_program_shape` reads the
//! declared buffers and never evaluates the program.

#![cfg(feature = "graph")]

use proptest::prelude::*;
use vyre_foundation::ir::Program;
use vyre_libs_graph::graph::csr_frontier_queue::frontier_words_to_queue_clear_out_parallel;
use vyre_primitives::wire::decode_u32_le_bytes_all as unpack_words;
use vyre_primitives::wire::pack_u32_slice as pack_words;
use vyre_reference::composition_witness::frontier_to_queue_witness;
use vyre_reference::value::Value;

/// Read one declared output by name.
///
/// Reading by position lets a builder that reorders its outputs swap the queue
/// for the cleared frontier without failing anything.
fn out_words(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: frontier queue program must declare output `{name}`"));
    unpack_words(&outputs[index].to_bytes())
}

/// Evaluate the clear-out materializer over `frontier`, seeding `frontier_out`.
fn run(
    node_count: u32,
    queue_capacity: u32,
    frontier: &[u32],
    frontier_out_seed: &[u32],
) -> (Program, Vec<Value>) {
    let program = frontier_words_to_queue_clear_out_parallel(
        "frontier",
        "queue",
        "queue_len",
        "frontier_out",
        node_count,
        queue_capacity,
    );
    let inputs = vec![
        Value::from(pack_words(frontier)),
        Value::from(vec![0_u8; queue_capacity as usize * size_of::<u32>()]),
        Value::from(pack_words(&[0])),
        Value::from(pack_words(frontier_out_seed)),
    ];
    let outputs = vyre_reference::ReferenceRequest::standard(&program, &inputs)
        .outputs()
        .expect(
            "Fix: the clear-out frontier queue materializer must evaluate on the reference oracle",
        );
    (program, outputs)
}

/// A word-mixing generator that keeps the top and bottom bits of some words
/// set, so word boundaries and the partial tail word are always exercised.
fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn generated_words(node_count: u32, seed: u32) -> Vec<u32> {
    let words = node_count.div_ceil(32);
    (0..words)
        .map(|word| {
            let mut bits = mix32(seed ^ word.wrapping_mul(0x9e37_79b9));
            if word % 3 == 0 {
                bits |= 1 << 31;
            }
            if word % 5 == 0 {
                bits |= 1;
            }
            bits
        })
        .collect()
}

/// WHY: the fused clear writes `frontier_out` from the same scan that fills the
/// queue. A clear that runs over the wrong span leaves a seeded word standing,
/// and the next traversal step reads a frontier that was already consumed. This
/// pins all three outputs of one dispatch against an independent witness on a
/// case whose node count is not a multiple of the word width.
#[test]
fn the_fused_clear_zeroes_every_frontier_word_the_queue_was_read_from() {
    let node_count = 70;
    let queue_capacity = 8;
    let frontier = [
        (1_u32 << 0) | (1_u32 << 1) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 31),
        (1_u32 << 0) | (1_u32 << 5) | (1_u32 << 31),
    ];
    let frontier_out_seed = [u32::MAX, 0xA5A5_A5A5, 0x8000_0001];
    let (expected_queue, expected_seen) =
        frontier_to_queue_witness(&frontier, node_count, queue_capacity as usize);
    let (program, outputs) = run(node_count, queue_capacity, &frontier, &frontier_out_seed);

    let mut queue = out_words(&program, &outputs, "queue");
    queue.truncate(expected_queue.len());
    queue.sort_unstable();
    let mut expected_sorted = expected_queue;
    expected_sorted.sort_unstable();

    assert_eq!(
        out_words(&program, &outputs, "queue_len"),
        vec![expected_seen],
        "Fix: report every active node below node_count in queue_len, including the ones past queue capacity"
    );
    assert_eq!(
        queue, expected_sorted,
        "Fix: emit the active nodes the witness names, in the capacity the caller declared"
    );
    assert_eq!(
        out_words(&program, &outputs, "frontier_out"),
        vec![0, 0, 0],
        "Fix: clear every frontier word the scan read, not only the ones that held an active bit"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    /// WHY: the queue is filled by a capacity-clamped scan, so three failures
    /// hide behind the same shape: a duplicate node, a node that was never
    /// active, and a count that stops at the capacity instead of at the node
    /// count. Generating the capacity independently of the active count puts
    /// the clamp on both sides of the boundary. The seeded `frontier_out` is
    /// nonzero in every word, so a clear that covers fewer words than the scan
    /// read is visible rather than coincidentally already zero.
    #[test]
    fn the_queue_holds_distinct_active_nodes_and_the_frontier_is_left_clear(
        node_count in 1u32..=257,
        frontier_seed in any::<u32>(),
        out_seed in any::<u32>(),
        capacity_salt in any::<u32>(),
    ) {
        let frontier = generated_words(node_count, frontier_seed);
        let frontier_words = frontier.len();
        let queue_capacity = 1 + capacity_salt % (node_count + 7);
        let frontier_out_seed = generated_words(node_count, out_seed ^ 0xa5a5_5a5a);
        let (all_active, expected_seen) =
            frontier_to_queue_witness(&frontier, node_count, node_count as usize);
        let (program, outputs) =
            run(node_count, queue_capacity, &frontier, &frontier_out_seed);

        let written = expected_seen.min(queue_capacity) as usize;
        let mut actual_queue = out_words(&program, &outputs, "queue")
            .into_iter()
            .take(written)
            .collect::<Vec<_>>();
        let mut unique_actual = actual_queue.clone();
        unique_actual.sort_unstable();
        unique_actual.dedup();
        let mut sorted_active = all_active;
        sorted_active.sort_unstable();

        prop_assert_eq!(
            out_words(&program, &outputs, "queue_len"),
            vec![expected_seen],
            "Fix: count every active node below node_count, not the ones that fit the queue"
        );
        prop_assert_eq!(
            unique_actual.len(),
            actual_queue.len(),
            "Fix: give each active node one queue slot; a repeated node means two lanes claimed the same index"
        );
        for node in &actual_queue {
            prop_assert!(
                sorted_active.binary_search(node).is_ok(),
                "Fix: enqueue only nodes whose frontier bit is set; node {node} is inactive at node_count={node_count}"
            );
        }
        if queue_capacity >= expected_seen {
            actual_queue.sort_unstable();
            prop_assert_eq!(
                actual_queue,
                sorted_active,
                "Fix: enqueue every active node when the capacity admits all of them"
            );
        }
        prop_assert_eq!(
            out_words(&program, &outputs, "frontier_out"),
            vec![0; frontier_words],
            "Fix: clear the whole frontier the scan read; a partial clear leaves a consumed node active for the next step"
        );
    }
}
