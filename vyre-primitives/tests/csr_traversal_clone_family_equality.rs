//! Cross-entry-point equality guard for the CSR traversal clone family.
//!
//! Every public entry point below is a thin wrapper over the one owner module
//! `graph::csr_frontier_step`. Two obligations are pinned here, both permanent:
//!
//! 1. `entry_point_ir_fingerprints_are_byte_identical` pins the canonical wire
//!    fingerprint of each entry point. The goldens were captured before the
//!    shared queue-loop builder existed, so the merge is proven to be a pure
//!    rehome, and any later edit to the shared builder that is not a pure
//!    rehome turns every affected entry point red.
//! 2. The `*_share_one_*` tests assert the wrappers really do share one
//!    implementation: after erasing the per-entry-point variable prefix, the
//!    queue bound check, the row lookup, the row-striping arithmetic, and the
//!    edge-kind/destination guard chain are literally the same node tree. A
//!    change made for one caller only cannot keep these green.

#![cfg(feature = "graph")]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_backward_traverse::csr_backward_traverse;
use vyre_primitives::graph::csr_bidirectional::csr_bidirectional;
use vyre_primitives::graph::csr_forward_or_changed::{
    csr_forward_or_changed, csr_forward_or_changed_parallel,
};
use vyre_primitives::graph::csr_forward_traverse::{
    csr_forward_traverse, csr_forward_traverse_excluding,
};
use vyre_primitives::graph::csr_frontier_queue::csr_queue_forward_traverse;
use vyre_primitives::graph::csr_queue_delta::{
    csr_queue_delta_enqueue, csr_queue_delta_strided_enqueue,
};
use vyre_primitives::graph::csr_queue_split::csr_queue_split_low_forward_traverse;
use vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_traverse;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
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

/// Canonical wire fingerprints captured from the tree before the shared
/// queue-loop builder existed. Regenerate ONLY when a shape change is the
/// intended product of the change under review.
const PRE_MERGE_FINGERPRINTS: &[(&str, &str)] = &[
    (
        "csr_queue_forward_traverse",
        "b5fc1b0e1fdd06a980dbf1cb9022f76a609b0a23337f66b41416a74d273eea0e",
    ),
    (
        "csr_queue_strided_forward_traverse",
        "de74a8dde1eb180ef6184d5200d59ef193963c7a0b747e17f9c862acda15b8b0",
    ),
    (
        "csr_queue_delta_enqueue",
        "b8021c7ce40143d990864b86dc941f1df0041c55245cb707d605a118572b8fb9",
    ),
    (
        "csr_queue_delta_strided_enqueue",
        "8d0ba983153034318b24b2e81ebf542bd8259d4f0843fbe48ac143a3a76a7989",
    ),
    (
        "csr_queue_delta_strided_enqueue.capped",
        "06ccca77ccf12fe22a4a9699827deb7880752fb216e7b929d57d605070b04547",
    ),
    (
        "csr_queue_split_low_forward_traverse",
        "f473c30c4e22bf9219cd4c8886d73c7f28d8c51df9e9eecd8ac3ec9d8788c8d2",
    ),
    (
        "csr_forward_traverse",
        "8fa33250b82f01ea53ec3098285a48d117132f13590e38fb5957f390d5471878",
    ),
    (
        "csr_forward_traverse_excluding",
        "23d2e90e68287dfab6b56c1588c194c32f3624458cecc6aede0b6f89560ae9b3",
    ),
    (
        "csr_backward_traverse",
        "b4941f80aa0e58a01b5e6020b36c4772b73f757c51529d0abfa63473d925a9bf",
    ),
    (
        "csr_bidirectional",
        "180ccef6ba84f423950a61a816f320c684ba18db45823ad087f6127a035967d5",
    ),
    (
        "csr_forward_or_changed",
        "484192aacb4bdcc6cd5db8a476ef0a1f6d7319dd1c8dfd6bb1ca293e66123f66",
    ),
    (
        "csr_forward_or_changed_parallel",
        "a43ffe6af801ae574139daee9c2cda5db6e852b5cef26affe36d056409552679",
    ),
];

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[test]
fn entry_point_ir_fingerprints_are_byte_identical() {
    let actual: Vec<(String, String)> = entry_points()
        .into_iter()
        .map(|(name, program)| (name.to_string(), hex32(program.fingerprint())))
        .collect();
    let expected: Vec<(String, String)> = PRE_MERGE_FINGERPRINTS
        .iter()
        .map(|(name, hash)| ((*name).to_string(), (*hash).to_string()))
        .collect();
    let table = actual
        .iter()
        .map(|(name, hash)| format!("    (\n        \"{name}\",\n        \"{hash}\",\n    ),"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        actual, expected,
        "Fix: a CSR traversal entry point changed its generated IR. Dedup must be a pure rehome; \
         if a shape change is intended, record why in the commit body and replace \
         PRE_MERGE_FINGERPRINTS with:\n{table}"
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
