//! One module and one registered op id per decode codec.
//!
//! WHY: base64, hex and inflate each had a builder and an `inventory::submit` on
//! both sides of a crate boundary, so a built program carried a registered
//! region nested inside a second registered region naming the same work. A
//! dispatcher walking the tree saw two operations where one ran, the op matrix
//! carried both names, and a caller had two spellings to choose between.
//!
//! Each codec is now one module with one id. The assertions below are written
//! against both halves of that contract, because either half alone passes on the
//! old shape: the surviving id appears exactly once, AND the collapsed id appears
//! zero times. Before the collapse the first assertion held while the second did
//! not, so this suite goes red on the previous behaviour.
//!
//! What this does not catch: an id registered by a module that no builder in this
//! crate reaches. `registry_closure.rs` owns that.

use vyre::ir::{Node, Program};
use vyre_foundation::visit::walk_nodes;
use vyre_libs::decode::base64::base64_decode;
use vyre_libs::decode::hex::hex_decode;
use vyre_libs::decode::inflate::inflate_stored_block;

/// Ids the collapse deleted. A region naming one of these means a second builder
/// for the same codec came back.
const COLLAPSED_IDS: &[&str] = &[
    "vyre-libs::decode::base64_decode",
    "vyre-libs::decode::hex_decode",
    "vyre-libs::decode::inflate_stored",
    "vyre-libs::decode::ziftsieve",
];

fn region_count(program: &Program, expected: &str) -> usize {
    let mut count = 0usize;
    walk_nodes(program, |node| {
        if matches!(node, Node::Region { generator, .. } if generator.as_str() == expected) {
            count += 1;
        }
    });
    count
}

/// Every generator name the program's regions carry, in walk order.
fn region_ids(program: &Program) -> Vec<String> {
    let mut ids = Vec::new();
    walk_nodes(program, |node| {
        if let Node::Region { generator, .. } = node {
            ids.push(generator.as_str().to_string());
        }
    });
    ids
}

fn assert_one_id(program: &Program, expected: &str) {
    assert_eq!(
        region_count(program, expected),
        1,
        "Fix: `{expected}` must name exactly one region, got these regions: {:?}",
        region_ids(program)
    );
    for collapsed in COLLAPSED_IDS {
        assert_eq!(
            region_count(program, collapsed),
            0,
            "Fix: `{collapsed}` was deleted by the codec collapse, so no program \
             may still build a region under it. Regions: {:?}",
            region_ids(program)
        );
    }
    assert_eq!(
        region_ids(program).len(),
        1,
        "Fix: a collapsed codec builds one registered region, not a region inside \
         a region. Regions: {:?}",
        region_ids(program)
    );
}

/// Standalone base64 decode carries one region and the four-buffer ABI.
#[test]
fn base64_decode_builds_one_registered_region() {
    let program = base64_decode("encoded", "decoded", 8);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 4);
    assert_eq!(program.output_buffer_indices(), vec![2, 3]);
    assert_eq!(program.buffers()[0].count(), 8);
    assert_eq!(program.buffers()[2].count(), 6);
    assert_one_id(&program, vyre_libs::decode::base64::OP_ID);
}

/// Standalone hex decode carries one region.
#[test]
fn hex_decode_builds_one_registered_region() {
    let program = hex_decode("encoded", "decoded", 8);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_one_id(&program, vyre_libs::decode::hex::OP_ID);
}

/// Standalone stored-block inflate carries one region and the length sidecar ABI.
#[test]
fn inflate_stored_builds_one_registered_region() {
    let program = inflate_stored_block("encoded", "decoded", 10);

    assert_eq!(program.workgroup_size(), [64, 1, 1]);
    assert_eq!(program.buffers().len(), 3);
    assert_eq!(program.output_buffer_indices(), vec![1, 2]);
    assert_eq!(program.buffers()[0].count(), 10);
    assert_eq!(program.buffers()[1].count(), 10);
    assert_eq!(program.buffers()[2].count(), 1);
    assert_one_id(&program, vyre_libs::decode::inflate::OP_ID);
}

/// Every op id the decode modules register is distinct, so no two modules claim
/// the same operation. Derived from the modules rather than a written-out list,
/// so a codec added later is covered without editing this test.
#[test]
fn every_decode_module_registers_a_distinct_id() {
    let ids = [
        vyre_libs::decode::base64::OP_ID,
        vyre_libs::decode::hex::OP_ID,
        vyre_libs::decode::inflate::OP_ID,
        vyre_libs::decode::ziftsieve::OP_ID,
        vyre_libs::decode::rle_segment_lengths::OP_ID,
    ];
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        before,
        "Fix: two decode modules register the same op id: {ids:?}"
    );
    for id in ids {
        assert!(
            !COLLAPSED_IDS.contains(&id),
            "Fix: `{id}` is an id the codec collapse deleted; a module reintroduced it."
        );
    }
}
