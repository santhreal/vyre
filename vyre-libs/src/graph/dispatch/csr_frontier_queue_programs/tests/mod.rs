//! Cross-entry-point equality guard for the resident CSR clone family.
//!
//! `csr_frontier_queue_resident`, `csr_frontier_queue_batch_resident` and
//! `adaptive_traverse` run one queue-driven CSR traversal against three
//! resident-buffer protocols. Three obligations are pinned here, all permanent:
//!
//! 1. `resident_family_ir_fingerprints_are_byte_identical` pins the canonical
//!    wire fingerprint of every Program the family builds. The goldens were
//!    captured before the resident Program builders were rehomed onto one
//!    owner, so the merge is proven to be a pure rehome.
//! 2. The `*_build_identical_*` tests assert the three sites really do agree,
//!    Program for Program. A change made for one resident caller only cannot
//!    keep them green.
//! 3. The `*_share_one_*` tests assert the resident traversal reaches its
//!    destination bit through the SAME node tree as the `vyre-primitives`
//!    queue-step builder that PR-04 owns: one edge-guard chain, one queue
//!    bound plus row lookup, one edge-walk loop. Only the resident-buffer
//!    additions may differ.

use crate::graph::csr_bidirectional::plan_csr_bidirectional_step;
use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
use crate::graph::csr_forward_or_changed::plan_csr_forward_or_changed_launch;
use crate::graph::csr_queue_delta::{csr_queue_delta_enqueue, csr_queue_delta_strided_enqueue};
use vyre_foundation::ir::Program;
use vyre_test_support::ir_regions::{canonicalize, edge_guard, region};

use super::{
    resident_csr_queue_atomic_word_scan_program, resident_csr_queue_block_offsets_program,
    resident_csr_queue_clear_frontier_out_program, resident_csr_queue_len_init_program,
    resident_csr_queue_materializer_programs, resident_csr_queue_split_low_program,
    resident_csr_queue_traverse_program, resident_csr_queue_word_counts_program,
    resident_csr_queue_word_prefix_queue_program,
};
use crate::graph::dispatch::csr_frontier_queue_scratch::{
    ResidentCsrQueueMaterializer, ResidentCsrQueueTraverseKind,
};

const NODE_COUNT: u32 = 64;
const EDGE_COUNT: u32 = 7;
const WORDS: u32 = 2;
const QUEUE_CAPACITY: u32 = 8;
const HIGH_QUEUE_CAPACITY: u32 = 4;
const NEXT_QUEUE_CAPACITY: u32 = 16;
const ALLOW_MASK: u32 = 1;

/// Resident buffer names bound by the single-query and batched resident paths.
const RESIDENT_FRONTIER_IN: &str = "frontier";
/// Adaptive traversal stages its own frontier upload under a distinct name.
const ADAPTIVE_FRONTIER_IN: &str = "frontier_in";

// ---------------------------------------------------------------------------
// One named coordinate per resident Program role, over one shared CSR fixture.
// ---------------------------------------------------------------------------

fn traverse_row_serial() -> Program {
    resident_csr_queue_traverse_program(
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        ALLOW_MASK,
        ResidentCsrQueueTraverseKind::RowSerial,
    )
}

fn traverse_row_strided() -> Program {
    resident_csr_queue_traverse_program(
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        ALLOW_MASK,
        ResidentCsrQueueTraverseKind::RowStrided,
    )
}

fn traverse_mixed_split_high() -> Program {
    resident_csr_queue_traverse_program(
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        ALLOW_MASK,
        ResidentCsrQueueTraverseKind::MixedSplit {
            high_queue_capacity: HIGH_QUEUE_CAPACITY,
        },
    )
}

fn split_low() -> Program {
    resident_csr_queue_split_low_program(
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        HIGH_QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

/// Programs built by one resident site, keyed by role.
fn site_programs(frontier_in: &str) -> Vec<(&'static str, Program)> {
    vec![
        ("traverse.row_serial", traverse_row_serial()),
        ("traverse.row_strided", traverse_row_strided()),
        ("traverse.mixed_split_high", traverse_mixed_split_high()),
        ("split_low", split_low()),
        (
            "queue_len_init",
            resident_csr_queue_len_init_program("queue_len"),
        ),
        (
            "high_len_init",
            resident_csr_queue_len_init_program("high_len"),
        ),
        (
            "materialize.atomic_word_scan",
            resident_csr_queue_atomic_word_scan_program(frontier_in, NODE_COUNT, QUEUE_CAPACITY),
        ),
        (
            "materialize.clear_frontier_out",
            resident_csr_queue_clear_frontier_out_program(WORDS),
        ),
        (
            "materialize.word_counts",
            resident_csr_queue_word_counts_program(frontier_in, NODE_COUNT),
        ),
        (
            "materialize.block_offsets",
            resident_csr_queue_block_offsets_program(NODE_COUNT),
        ),
        (
            "materialize.block_offsets_queue",
            resident_csr_queue_word_prefix_queue_program(
                frontier_in,
                NODE_COUNT,
                QUEUE_CAPACITY,
                true,
            ),
        ),
        (
            "materialize.word_prefix_queue",
            resident_csr_queue_word_prefix_queue_program(
                frontier_in,
                NODE_COUNT,
                QUEUE_CAPACITY,
                false,
            ),
        ),
    ]
}

/// Small CSR fixture for the two plan-driven family members.
fn csr_fixture() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(NODE_COUNT as usize + 1);
    for node in 0..=NODE_COUNT {
        offsets.push(node.min(EDGE_COUNT));
    }
    let targets: Vec<u32> = (0..EDGE_COUNT)
        .map(|edge| (edge * 3) % NODE_COUNT)
        .collect();
    let masks = vec![ALLOW_MASK; EDGE_COUNT as usize];
    let frontier = vec![1u32, 0u32];
    (offsets, targets, masks, frontier)
}

fn bidirectional_program() -> Program {
    let (offsets, targets, masks, frontier) = csr_fixture();
    plan_csr_bidirectional_step(
        NODE_COUNT, &offsets, &targets, &masks, &frontier, ALLOW_MASK,
    )
    .expect("Fix: bidirectional fixture must be a valid CSR graph")
    .program()
}

fn forward_or_changed_program(max_iters: u32) -> Program {
    let (offsets, targets, masks, _) = csr_fixture();
    plan_csr_forward_or_changed_launch(CsrClosureInputs {
        graph: CsrGraphView {
            node_count: NODE_COUNT,
            edge_offsets: &offsets,
            edge_targets: &targets,
            edge_kind_mask: &masks,
        },
        allow_mask: ALLOW_MASK,
        max_iters,
    })
    .expect("Fix: forward-or-changed fixture must be a valid CSR graph")
    .program()
    .expect("Fix: forward-or-changed fixture must be representable")
}

/// Every Program the resident CSR clone family builds, one row per site.
fn entry_points() -> Vec<(String, Program)> {
    let mut out = Vec::new();
    for (site, frontier_in) in [
        ("resident", RESIDENT_FRONTIER_IN),
        ("batch", RESIDENT_FRONTIER_IN),
        ("adaptive", ADAPTIVE_FRONTIER_IN),
    ] {
        for (role, program) in site_programs(frontier_in) {
            out.push((format!("{site}.{role}"), program));
        }
    }
    out.push(("csr_bidirectional".to_string(), bidirectional_program()));
    out.push((
        "csr_forward_or_changed.history".to_string(),
        forward_or_changed_program(4),
    ));
    out.push((
        "csr_forward_or_changed.single_slot".to_string(),
        forward_or_changed_program(0),
    ));
    out
}

/// Canonical wire fingerprints captured from the tree before the resident
/// Program builders were rehomed onto one owner. Regenerate ONLY when a shape
/// change is the intended product of the change under review.
///
/// Six rows moved once since capture. `frontier_word_counts_scan_pass_a` and
/// `frontier_word_block_offsets_single_workgroup` swept a Hillis-Steele tree
/// and now sweep a Blelloch tree, which writes O(n) elements over the sweep
/// instead of writing every lane on every round. The three sites read those
/// two builders, so `materialize.word_counts` and `materialize.block_offsets`
/// moved at each of them.
const PRE_MERGE_FINGERPRINTS: &str = "\
resident.traverse.row_serial 768fa8a83852a0efeb13259bbfec7a3d2e578578ec3db1ed5aee00abbeb580ee
resident.traverse.row_strided c1ecb077c7f03ce568070562b0327e3ca5d3505cbeb726275ebc667ab1e9100f
resident.traverse.mixed_split_high c0690c885eb98c883ac62873361d602ac92b44732d036b587415df64175aeec4
resident.split_low b9191307c246dd9949b08b859cbd706af31e7f882254ea64ae9ee7de2129f122
resident.queue_len_init 448b389f3a30fa3e1998d837f219d5e25cf791b61cce84060eb98a3cb4b09604
resident.high_len_init d961c35f055e5e7f87e30007be4570bdd9cb2c3c995bfb7104615ed4ef700d5c
resident.materialize.atomic_word_scan 1b2148f8e6df60adf345dd66b01896e32488544d591265c09ad2171227e80b0d
resident.materialize.clear_frontier_out daac7cd6ed7b19bf8f9a2a0aae6036d6765c6614a6faa14bc4a40f642647dc9f
resident.materialize.word_counts 3282efa4797486f76aac2541da8b6a72130f3f13fc85dc55e98b389a6b9d9fdb
resident.materialize.block_offsets eef97ba97ecb13c5182d1dca23fb63cc9d7a5477ed67b5e63991616ff9e650b5
resident.materialize.block_offsets_queue 3c44dfe044c89db30a01ce0d2c922b383083a1744db51d478e77c888bf5c5f1a
resident.materialize.word_prefix_queue 27b514873870b708be9e8001641da82fff6b15a5f367fc36194cdc994815e12b
batch.traverse.row_serial 768fa8a83852a0efeb13259bbfec7a3d2e578578ec3db1ed5aee00abbeb580ee
batch.traverse.row_strided c1ecb077c7f03ce568070562b0327e3ca5d3505cbeb726275ebc667ab1e9100f
batch.traverse.mixed_split_high c0690c885eb98c883ac62873361d602ac92b44732d036b587415df64175aeec4
batch.split_low b9191307c246dd9949b08b859cbd706af31e7f882254ea64ae9ee7de2129f122
batch.queue_len_init 448b389f3a30fa3e1998d837f219d5e25cf791b61cce84060eb98a3cb4b09604
batch.high_len_init d961c35f055e5e7f87e30007be4570bdd9cb2c3c995bfb7104615ed4ef700d5c
batch.materialize.atomic_word_scan 1b2148f8e6df60adf345dd66b01896e32488544d591265c09ad2171227e80b0d
batch.materialize.clear_frontier_out daac7cd6ed7b19bf8f9a2a0aae6036d6765c6614a6faa14bc4a40f642647dc9f
batch.materialize.word_counts 3282efa4797486f76aac2541da8b6a72130f3f13fc85dc55e98b389a6b9d9fdb
batch.materialize.block_offsets eef97ba97ecb13c5182d1dca23fb63cc9d7a5477ed67b5e63991616ff9e650b5
batch.materialize.block_offsets_queue 3c44dfe044c89db30a01ce0d2c922b383083a1744db51d478e77c888bf5c5f1a
batch.materialize.word_prefix_queue 27b514873870b708be9e8001641da82fff6b15a5f367fc36194cdc994815e12b
adaptive.traverse.row_serial 768fa8a83852a0efeb13259bbfec7a3d2e578578ec3db1ed5aee00abbeb580ee
adaptive.traverse.row_strided c1ecb077c7f03ce568070562b0327e3ca5d3505cbeb726275ebc667ab1e9100f
adaptive.traverse.mixed_split_high c0690c885eb98c883ac62873361d602ac92b44732d036b587415df64175aeec4
adaptive.split_low b9191307c246dd9949b08b859cbd706af31e7f882254ea64ae9ee7de2129f122
adaptive.queue_len_init 448b389f3a30fa3e1998d837f219d5e25cf791b61cce84060eb98a3cb4b09604
adaptive.high_len_init d961c35f055e5e7f87e30007be4570bdd9cb2c3c995bfb7104615ed4ef700d5c
adaptive.materialize.atomic_word_scan 57c07f2d9f7aec5e3268da98a7b7695b0941896cc6e0dc1ca04fe334644d121d
adaptive.materialize.clear_frontier_out daac7cd6ed7b19bf8f9a2a0aae6036d6765c6614a6faa14bc4a40f642647dc9f
adaptive.materialize.word_counts 1ffc1ce926cd4fcb9b640bbe89fb708515d6e489bb3ff8afc5e56b7d687f105a
adaptive.materialize.block_offsets eef97ba97ecb13c5182d1dca23fb63cc9d7a5477ed67b5e63991616ff9e650b5
adaptive.materialize.block_offsets_queue adb70fbfc597839088e023f2624ecfa3d019dc7564315533ba101f5c4d9d4a0b
adaptive.materialize.word_prefix_queue c2fa99828b92fe95addfd5c3ac58ca5559f67396f27feb59b2f3cfa83bd93926
csr_bidirectional 61707f4fd384664b1ea64df86b567826f4b46902fab0b7b20e4f67ec4a04e97c
csr_forward_or_changed.history 9837ba05daabd3f15f9a54fb4ba19b7bcf1195a1363c8226127980c12d4e58c3
csr_forward_or_changed.single_slot 148d72fd265a482a788c6b2346142361ab2741de4705259b5f0b37c95e719cb6\n";

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// One `name hash` line per Program, in `entry_points` order.
fn fingerprint_table(rows: impl Iterator<Item = (String, String)>) -> String {
    rows.map(|(name, hash)| format!("{name} {hash}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn resident_family_ir_fingerprints_are_byte_identical() {
    let actual = fingerprint_table(
        entry_points()
            .into_iter()
            .map(|(name, program)| (name, hex32(program.fingerprint()))),
    );
    if actual == PRE_MERGE_FINGERPRINTS.trim_end() {
        return;
    }
    // Two thirty-row tables compared as one string report the whole blob and
    // truncate it, which names no builder. Pair the rows so the failure says
    // which Program moved.
    let expected: Vec<&str> = PRE_MERGE_FINGERPRINTS.trim_end().lines().collect();
    let observed: Vec<&str> = actual.lines().collect();
    let mut moved = Vec::new();
    for row in 0..expected.len().max(observed.len()) {
        let (before, after) = (expected.get(row), observed.get(row));
        if before != after {
            moved.push(format!(
                "  row {row}: was {}, is {}",
                before.copied().unwrap_or("<absent>"),
                after.copied().unwrap_or("<absent>")
            ));
        }
    }
    panic!(
        "Fix: a resident CSR Program changed its generated IR. Dedup must be a pure rehome; if a \
         shape change is intended, record why in the commit body and replace \
         PRE_MERGE_FINGERPRINTS with the observed table.\n{}\n\nObserved table:\n{actual}",
        moved.join("\n")
    );
}

#[test]
fn resident_sites_build_identical_traverse_programs() {
    let resident = site_programs(RESIDENT_FRONTIER_IN);
    let batch = site_programs(RESIDENT_FRONTIER_IN);
    let adaptive = site_programs(ADAPTIVE_FRONTIER_IN);
    // Every role except the four frontier-input readers is name-for-name equal
    // across all three sites; those four legitimately bind a different upload
    // buffer and are compared per-name in the fingerprint table instead.
    let frontier_reading = [
        "materialize.atomic_word_scan",
        "materialize.word_counts",
        "materialize.block_offsets_queue",
        "materialize.word_prefix_queue",
    ];
    for ((role, a), ((_, b), (_, c))) in resident
        .iter()
        .zip(batch.iter().zip(adaptive.iter()))
        .map(|((role, a), (b, c))| ((*role, a), (b, c)))
    {
        assert_eq!(
            a.fingerprint(),
            b.fingerprint(),
            "Fix: the single-query and batched resident sites must build one {role} Program."
        );
        if frontier_reading.contains(&role) {
            continue;
        }
        assert_eq!(
            a.fingerprint(),
            c.fingerprint(),
            "Fix: adaptive traversal must build the same {role} Program as the resident sites."
        );
    }
}

/// A delta-emit queue step over the SAME resident buffers. It is built by the
/// PR-04 owner and is not part of the resident family, so agreeing with it is
/// evidence of one shared builder rather than of one shared copy.
fn primitive_delta() -> Program {
    csr_queue_delta_enqueue(
        "active_queue",
        "queue_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "frontier_out",
        "next_queue",
        "next_len",
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        NEXT_QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

fn primitive_delta_strided() -> Program {
    csr_queue_delta_strided_enqueue(
        "active_queue",
        "queue_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "frontier_out",
        "next_queue",
        "next_len",
        NODE_COUNT,
        EDGE_COUNT,
        QUEUE_CAPACITY,
        NEXT_QUEUE_CAPACITY,
        ALLOW_MASK,
    )
}

#[test]
fn resident_family_shares_one_edge_guard_chain_with_the_primitive_builder() {
    let reference = edge_guard(&primitive_delta(), "qd", "qd_old");
    for (name, guard) in [
        (
            "traverse.row_serial",
            edge_guard(&traverse_row_serial(), "qt", "_qt_prev"),
        ),
        (
            "traverse.row_strided",
            edge_guard(&traverse_row_strided(), "qs", "_qs_prev"),
        ),
        (
            "traverse.mixed_split_high",
            edge_guard(&traverse_mixed_split_high(), "qs", "_qs_prev"),
        ),
        ("split_low", edge_guard(&split_low(), "qsl", "_qsl_prev")),
    ] {
        assert_eq!(
            guard, reference,
            "Fix: resident {name} must reach its destination bit through the one shared CSR edge \
             guard owned by vyre-primitives."
        );
    }
}

#[test]
fn resident_scalar_traversal_shares_one_queue_bound_and_row_lookup() {
    let reference = region(
        &canonicalize(&primitive_delta(), "qd"),
        "Ident(\"Q_idx\")",
        "Ident(\"Q_edge_end\")",
    );
    for (name, prefix, program) in [
        ("traverse.row_serial", "qt", traverse_row_serial()),
        ("split_low", "qsl", split_low()),
    ] {
        assert_eq!(
            region(
                &canonicalize(&program, prefix),
                "Ident(\"Q_idx\")",
                "Ident(\"Q_edge_end\")",
            ),
            reference,
            "Fix: resident {name} must take the one shared scalar queue bound check and CSR row \
             lookup owned by vyre-primitives."
        );
    }
}

#[test]
fn resident_scalar_traversal_shares_one_edge_walk_loop() {
    assert_eq!(
        region(
            &canonicalize(&traverse_row_serial(), "qt"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"_Q_prev\")",
        ),
        region(
            &canonicalize(&primitive_delta(), "qd"),
            "Ident(\"Q_edge_start\")",
            "Ident(\"Q_old\")",
        ),
        "Fix: resident scalar traversal must walk a queued CSR row through the one shared loop \
         owned by vyre-primitives."
    );
}

#[test]
fn resident_strided_traversal_shares_one_row_striping_loop() {
    let reference = region(
        &canonicalize(&primitive_delta_strided(), "qds"),
        "Ident(\"Q_edge_start\")",
        "Ident(\"Q_old\")",
    );
    for (name, program) in [
        ("traverse.row_strided", traverse_row_strided()),
        ("traverse.mixed_split_high", traverse_mixed_split_high()),
    ] {
        assert_eq!(
            region(
                &canonicalize(&program, "qs"),
                "Ident(\"Q_edge_start\")",
                "Ident(\"_Q_prev\")",
            ),
            reference,
            "Fix: resident {name} must stripe a CSR row through the one shared loop owned by \
             vyre-primitives."
        );
    }
}

#[test]
fn materializer_program_set_matches_its_leaf_builders() {
    for frontier_in in [RESIDENT_FRONTIER_IN, ADAPTIVE_FRONTIER_IN] {
        let atomic = resident_csr_queue_materializer_programs(
            frontier_in,
            NODE_COUNT,
            WORDS,
            QUEUE_CAPACITY,
            ResidentCsrQueueMaterializer::AtomicWordScan,
            false,
        );
        assert!(
            atomic.clear_frontier_out.is_none()
                && atomic.word_counts.is_none()
                && atomic.word_block_offsets.is_none(),
            "Fix: the atomic word scan clears the output frontier itself and runs no prefix scan."
        );
        assert_eq!(
            atomic
                .queue_len_init
                .expect("Fix: the atomic word scan must reset the queue length")
                .fingerprint(),
            resident_csr_queue_len_init_program("queue_len").fingerprint(),
        );
        assert_eq!(
            atomic.queue.fingerprint(),
            resident_csr_queue_atomic_word_scan_program(frontier_in, NODE_COUNT, QUEUE_CAPACITY)
                .fingerprint(),
        );

        for precomputed_block_offsets in [false, true] {
            let prefix = resident_csr_queue_materializer_programs(
                frontier_in,
                NODE_COUNT,
                WORDS,
                QUEUE_CAPACITY,
                ResidentCsrQueueMaterializer::DeterministicWordPrefix,
                precomputed_block_offsets,
            );
            assert!(
                prefix.queue_len_init.is_none(),
                "Fix: the word-prefix scatter writes an exact queue length, so nothing resets it."
            );
            assert_eq!(
                prefix
                    .clear_frontier_out
                    .expect("Fix: the word-prefix path must clear the output frontier")
                    .fingerprint(),
                resident_csr_queue_clear_frontier_out_program(WORDS).fingerprint(),
            );
            assert_eq!(
                prefix
                    .word_counts
                    .expect("Fix: the word-prefix path must popcount frontier words")
                    .fingerprint(),
                resident_csr_queue_word_counts_program(frontier_in, NODE_COUNT).fingerprint(),
            );
            assert_eq!(
                prefix
                    .word_block_offsets
                    .map(|program| program.fingerprint()),
                precomputed_block_offsets
                    .then(|| resident_csr_queue_block_offsets_program(NODE_COUNT).fingerprint()),
                "Fix: a separate block-offset scan runs exactly when the scatter does not sum \
                 block totals inline."
            );
            assert_eq!(
                prefix.queue.fingerprint(),
                resident_csr_queue_word_prefix_queue_program(
                    frontier_in,
                    NODE_COUNT,
                    QUEUE_CAPACITY,
                    precomputed_block_offsets,
                )
                .fingerprint(),
            );
        }
    }
}
