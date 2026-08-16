//! `frontier_to_queue` is a cooperative single-workgroup scan, but nothing makes
//! its dispatch single-workgroup, so the driver's own span rule launches it wide
//! and it double-appends.
//!
//! Mechanism. The scan walks `q_src = q_iter * 256 + q_lane` where `q_lane` is
//! the GLOBAL invocation id and `q_iter` runs to `ceil(node_count / 256)`. That
//! covers `[0, node_count)` exactly once only when the grid is one workgroup. At
//! `G` workgroups the lanes of group `g` re-derive `q_src` values that group 0
//! already covered at a later `q_iter`, so every set frontier bit at or above
//! index 256 is appended once PER covering group and `queue_len` is inflated by
//! the same factor.
//!
//! The grid is not the caller's choice by default. `frontier_to_queue` emits
//! `atomic_add` on `queue_len`, and once a program contains any atomic the
//! dispatch span becomes the widest non-shared binding
//! (`vyre-driver` `dispatch_element_count_for_program`, mirrored by the CPU
//! reference interpreter's `force_full_span`). The widest binding here is
//! `active_queue`, whose count is `queue_capacity`, so a capacity that merely
//! matches the node count already produces a multi-workgroup grid. No hostile
//! input and no race is required: workgroup coverage overlaps deterministically,
//! so the wrong answer is reproducible on the CPU reference backend.
//!
//! Same root cause as the lane-gate family in the persistent-fixpoint audit, but
//! a different symptom: here the shared word is inflated by duplicate atomic
//! increments rather than erased by a racing plain-store clear.

#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use vyre_foundation::ir::Program;
use vyre_libs::graph::csr_frontier_queue::{frontier_to_queue, frontier_to_queue_cpu};
use vyre_primitives::wire::decode_u32_le_bytes_all as unpack_words;
use vyre_primitives::wire::pack_u32_slice as pack_words;
use vyre_reference::value::Value;

fn out_words(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: frontier queue program must declare output `{name}`"));
    unpack_words(&outputs[index].to_bytes())
}

/// Build a frontier bitset over `node_count` nodes with exactly `set` bits set.
fn frontier_with(node_count: u32, set: &[u32]) -> Vec<u32> {
    let mut words = vec![0u32; node_count.div_ceil(32) as usize];
    for &node in set {
        words[(node / 32) as usize] |= 1u32 << (node % 32);
    }
    words
}

fn run(program: &Program, frontier: &[u32], queue_capacity: u32) -> (Vec<u32>, Vec<u32>) {
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack_words(frontier)),
            Value::from(vec![
                0_u8;
                queue_capacity as usize * std::mem::size_of::<u32>()
            ]),
            Value::from(pack_words(&[0])),
        ],
    )
    .expect("Fix: frontier_to_queue must reference-evaluate");
    (
        out_words(program, &outputs, "queue"),
        out_words(program, &outputs, "queue_len"),
    )
}

/// A single workgroup covers `[0, node_count)` exactly once, so the scan agrees
/// with the CPU oracle. This is the configuration the builder's doc comment
/// assumes and it must keep working after the fix.
#[test]
fn single_workgroup_span_appends_each_active_node_exactly_once() {
    // node_count and queue_capacity both under the 256-wide workgroup, so the
    // inferred span is one workgroup.
    let node_count = 200;
    let queue_capacity = 200;
    let active = [3_u32, 64, 199];
    let frontier = frontier_with(node_count, &active);

    let (expected_queue, expected_seen) =
        frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
    assert_eq!(
        expected_seen, 3,
        "oracle must see exactly the three seeded nodes"
    );
    assert_eq!(expected_queue, vec![3, 64, 199]);

    let program = frontier_to_queue("frontier", "queue", "queue_len", node_count, queue_capacity);
    let (queue, queue_len) = run(&program, &frontier, queue_capacity);

    assert_eq!(
        queue_len,
        vec![expected_seen],
        "single-workgroup scan must report exactly 3 appended nodes"
    );
    let mut got: Vec<u32> = queue[..expected_seen as usize].to_vec();
    got.sort_unstable();
    assert_eq!(got, vec![3, 64, 199]);
}

/// The defect. With `queue_capacity` at the node count the driver's span rule
/// gives this program two workgroups, and every active node at index >= 256 is
/// appended twice: once by group 0 at `q_iter == 1` and once by group 1 at
/// `q_iter == 0`. `queue_len` is 4 instead of 3 and node 260 appears twice.
///
/// Asserted as exact values, not as a shape: node 260 must appear exactly once
/// and the length must be exactly the oracle's `seen`.
#[test]
fn multi_workgroup_span_must_not_double_append_nodes_above_the_workgroup_width() {
    let node_count = 300;
    let queue_capacity = 300;
    // One node below the workgroup width (covered once, by group 0 at q_iter 0)
    // and two at or above it (covered by group 0 at q_iter 1 AND group 1 at
    // q_iter 0).
    let active = [5_u32, 260, 299];
    let frontier = frontier_with(node_count, &active);

    let (expected_queue, expected_seen) =
        frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
    assert_eq!(expected_seen, 3);
    assert_eq!(expected_queue, vec![5, 260, 299]);

    let program = frontier_to_queue("frontier", "queue", "queue_len", node_count, queue_capacity);
    let (queue, queue_len) = run(&program, &frontier, queue_capacity);

    assert_eq!(
        queue_len,
        vec![expected_seen],
        "Fix: frontier_to_queue counted {:?} appends for 3 active nodes; nodes at or \
         above the 256-wide workgroup are being scanned by more than one workgroup, so \
         atomic_add on queue_len fires once per covering group.",
        queue_len,
    );

    let appended: Vec<u32> = queue[..expected_seen as usize].to_vec();
    for node in active {
        let hits = appended.iter().filter(|&&entry| entry == node).count();
        assert_eq!(
            hits, 1,
            "Fix: node {node} was appended {hits} times, expected exactly once; \
             queue prefix was {appended:?}."
        );
    }
}

/// Coverage must stay exactly-once across the workgroup-width boundary, which is
/// where a scan that strides by a fixed 256 while the grid grows starts
/// overlapping. Sweeps node counts that bracket 256 and 512.
#[test]
fn every_active_node_is_appended_exactly_once_across_workgroup_width_boundaries() {
    for node_count in [255_u32, 256, 257, 511, 512, 513, 600] {
        let queue_capacity = node_count;
        // Seed the first, last, and both sides of each 256 boundary that exists.
        let mut active = vec![0_u32, node_count - 1];
        for boundary in [256_u32, 512] {
            if boundary < node_count {
                active.push(boundary - 1);
                active.push(boundary);
            }
        }
        active.sort_unstable();
        active.dedup();
        let frontier = frontier_with(node_count, &active);

        let (expected_queue, expected_seen) =
            frontier_to_queue_cpu(&frontier, node_count, queue_capacity as usize);
        assert_eq!(expected_seen as usize, active.len());

        let program =
            frontier_to_queue("frontier", "queue", "queue_len", node_count, queue_capacity);
        let (queue, queue_len) = run(&program, &frontier, queue_capacity);

        assert_eq!(
            queue_len,
            vec![expected_seen],
            "Fix: node_count={node_count} reported {queue_len:?} appends for \
             {} active nodes.",
            active.len(),
        );
        let mut got: Vec<u32> = queue[..expected_seen as usize].to_vec();
        got.sort_unstable();
        assert_eq!(
            got, expected_queue,
            "Fix: node_count={node_count} queue contents diverged from the CPU oracle."
        );
    }
}
