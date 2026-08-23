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
use crate::graph::csr_frontier_queue::scratch::{
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

/// Canonical wire fingerprints for resident CSR programs.
///
/// WHY: Wire revision 8 and schedule-free logical execution markers intentionally
/// change canonical bytes. Semantic parity remains separately tested across the
/// conformance and parity suites.
const CANONICAL_FINGERPRINTS: &str = "\
resident.traverse.row_serial 2a11c2fc01e0c888807492aef3faaa1f70126bb018618882cfcd23b7b1ed171a
resident.traverse.row_strided 831e82ee1cdf970304845950209a0e1b8f561f7d759b426c4bfd25c5a7e66620
resident.traverse.mixed_split_high c69f1a7ab222bfeae0eeba99c1b4b506eaa4ef1dcbd15ed6182939cdf8ba7dd1
resident.split_low 9d451807d27aacb1c100efd89356838d4ec320ed40d33e8917437df280286b7b
resident.queue_len_init 986258c804419b70dbfb9117960ae230e42f8c847b4006092f03e0124a51fbd5
resident.high_len_init 563a712e1f2182905ac5dd3bc967bf773076ab46922b194b1ba78db955f20da7
resident.materialize.atomic_word_scan 461e620138c1e4a3fe7c641027ef555307b8b3a36c497d2b9e1323ec6aeee496
resident.materialize.clear_frontier_out 93020e4b3677888e851b890007fce4818acff65ae20569693519fb5eb0d1c275
resident.materialize.word_counts 123af65b7efea1ead98cc4a25281159e774edb2c18a724e3689f8a6279976043
resident.materialize.block_offsets ed591b08258a12f3b79a6501725efcef78b962756dcccc96e9b32ad023ba9e34
resident.materialize.block_offsets_queue fd3d9c6f6e502c18814282bf01b97f2efc75601f52d8e6e5dda2b8dffa04a59c
resident.materialize.word_prefix_queue ebe6132026f50e919e7fa91778d5406381f9f69699ab8905ec96123d51d9dc7d
batch.traverse.row_serial 2a11c2fc01e0c888807492aef3faaa1f70126bb018618882cfcd23b7b1ed171a
batch.traverse.row_strided 831e82ee1cdf970304845950209a0e1b8f561f7d759b426c4bfd25c5a7e66620
batch.traverse.mixed_split_high c69f1a7ab222bfeae0eeba99c1b4b506eaa4ef1dcbd15ed6182939cdf8ba7dd1
batch.split_low 9d451807d27aacb1c100efd89356838d4ec320ed40d33e8917437df280286b7b
batch.queue_len_init 986258c804419b70dbfb9117960ae230e42f8c847b4006092f03e0124a51fbd5
batch.high_len_init 563a712e1f2182905ac5dd3bc967bf773076ab46922b194b1ba78db955f20da7
batch.materialize.atomic_word_scan 461e620138c1e4a3fe7c641027ef555307b8b3a36c497d2b9e1323ec6aeee496
batch.materialize.clear_frontier_out 93020e4b3677888e851b890007fce4818acff65ae20569693519fb5eb0d1c275
batch.materialize.word_counts 123af65b7efea1ead98cc4a25281159e774edb2c18a724e3689f8a6279976043
batch.materialize.block_offsets ed591b08258a12f3b79a6501725efcef78b962756dcccc96e9b32ad023ba9e34
batch.materialize.block_offsets_queue fd3d9c6f6e502c18814282bf01b97f2efc75601f52d8e6e5dda2b8dffa04a59c
batch.materialize.word_prefix_queue ebe6132026f50e919e7fa91778d5406381f9f69699ab8905ec96123d51d9dc7d
adaptive.traverse.row_serial 2a11c2fc01e0c888807492aef3faaa1f70126bb018618882cfcd23b7b1ed171a
adaptive.traverse.row_strided 831e82ee1cdf970304845950209a0e1b8f561f7d759b426c4bfd25c5a7e66620
adaptive.traverse.mixed_split_high c69f1a7ab222bfeae0eeba99c1b4b506eaa4ef1dcbd15ed6182939cdf8ba7dd1
adaptive.split_low 9d451807d27aacb1c100efd89356838d4ec320ed40d33e8917437df280286b7b
adaptive.queue_len_init 986258c804419b70dbfb9117960ae230e42f8c847b4006092f03e0124a51fbd5
adaptive.high_len_init 563a712e1f2182905ac5dd3bc967bf773076ab46922b194b1ba78db955f20da7
adaptive.materialize.atomic_word_scan 02fba69f422a929d743d52f7041fbce5bc76f9656553ee30b2c1919297320a46
adaptive.materialize.clear_frontier_out 93020e4b3677888e851b890007fce4818acff65ae20569693519fb5eb0d1c275
adaptive.materialize.word_counts 2d9bab8122c5648451a51f4be8071b50427cf1f6734ac260a3054bfb9bdf1c18
adaptive.materialize.block_offsets ed591b08258a12f3b79a6501725efcef78b962756dcccc96e9b32ad023ba9e34
adaptive.materialize.block_offsets_queue eaabc9a71212f4113e1a1627ec868bf3aca209d9d1820cf63f44178fcfd45a06
adaptive.materialize.word_prefix_queue 5e701fad74a1fb3a24969a1745549e3250dd9127a2ca546583c5107683113fca
csr_bidirectional 62ff3a0986dcb30cd4fe59218b34c50d4dea02af09b7ef8060a7c771344eb258
csr_forward_or_changed.history 35876ed996dbf6fbce4391d6786f2fc10b5838c5731ac401cc65aff8ad39becf
csr_forward_or_changed.single_slot ba44655dbbd847c4d033ab4244fa59f3fb9393483601469a2659f83b9b315e96\n";

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
    if actual == CANONICAL_FINGERPRINTS.trim_end() {
        return;
    }
    // Two thirty-row tables compared as one string report the whole blob and
    // truncate it, which names no builder. Pair the rows so the failure says
    // which Program moved.
    let expected: Vec<&str> = CANONICAL_FINGERPRINTS.trim_end().lines().collect();
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
         CANONICAL_FINGERPRINTS with the observed table.\n{}\n\nObserved table:\n{actual}",
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
