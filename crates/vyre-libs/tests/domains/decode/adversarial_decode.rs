//! Failure-oriented adversarial tests for decode primitives.
//!
//! Focus: hostile boundaries, overflow, invalid offsets, property invariants.
#![cfg(feature = "decode")]

use vyre_libs::decode::base64::*;

#[test]
fn decoded_capacity_hostile_lengths() {
    let max_blocks = u32::MAX / 4;
    let max_expected = max_blocks.saturating_mul(3);
    let cases = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 3),
        (5, 3),
        (7, 3),
        (8, 6),
        (u32::MAX, max_expected),
    ];
    for (input_len, expected) in cases {
        let got = decoded_capacity(input_len);
        assert_eq!(got, expected, "decoded_capacity({input_len}) mismatch");
    }
}

#[test]
fn decoded_capacity_no_panic_on_max() {
    let _ = decoded_capacity(u32::MAX);
}

#[test]
fn base64_decode_program_has_expected_buffers() {
    // The collapsed builder names the table and the decoded-length sidecar
    // itself, so the contract is the count, the order and which ones the program
    // publishes, not four spellings the caller no longer supplies.
    let p = base64_decode("input", "output", 4);
    assert_eq!(p.buffers().len(), 4);
    assert_eq!(p.output_buffer_indices(), vec![2, 3]);
    assert_eq!(p.buffers()[0].count(), 4);
    assert_eq!(p.buffers()[2].count(), decoded_capacity(4));
    assert_eq!(p.buffers()[3].count(), 1);
}

#[test]
fn base64_decode_child_returns_region() {
    let node = base64_decode_child("parent", "input", "table", "output", "decoded_len", 4);
    match node {
        vyre_foundation::ir::Node::Region { generator, .. } => {
            assert_eq!(generator.as_str(), OP_ID);
        }
        other => panic!("expected Region node, got {other:?}"),
    }
}
