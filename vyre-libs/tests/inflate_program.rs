//! Adversarial tests for DEFLATE stored-block inflate program construction.
//!
//! `inflate_stored_block` emits an IR program that must validate, round-trip
//! through the wire format, and declare correct buffer access modes.
//!
//! The builder scopes the caller's buffer names into the decode family and names
//! the inflated-length sidecar itself, so these assertions address the buffers by
//! position and access mode, which is the ABI a dispatcher binds against. A test
//! that spelled the names out would pass on any builder that kept the spelling
//! and broke the order.

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_libs::decode::inflate_stored_block;

#[test]
fn inflate_stored_has_three_buffers() {
    let prog = inflate_stored_block("input", "output", 10);
    assert_eq!(prog.buffers().len(), 3);
}

#[test]
fn inflate_stored_input_is_readonly() {
    let prog = inflate_stored_block("input", "output", 10);
    let buf = &prog.buffers()[0];
    assert_eq!(buf.access(), BufferAccess::ReadOnly);
    assert_eq!(buf.count(), 10);
}

#[test]
fn inflate_stored_output_is_write_only() {
    let prog = inflate_stored_block("input", "output", 10);
    let buf = &prog.buffers()[1];
    assert!(buf.is_output());
    assert_eq!(buf.count(), 10);
}

#[test]
fn inflate_stored_len_is_readwrite() {
    let prog = inflate_stored_block("input", "output", 10);
    let buf = &prog.buffers()[2];
    assert_eq!(buf.access(), BufferAccess::ReadWrite);
    assert_eq!(buf.count(), 1);
}

#[test]
fn inflate_stored_publishes_the_payload_and_the_length() {
    let prog = inflate_stored_block("input", "output", 10);
    assert_eq!(prog.output_buffer_indices(), vec![1, 2]);
}

#[test]
fn inflate_stored_wire_roundtrips() {
    let prog = inflate_stored_block("input", "output", 10);
    let bytes = prog.to_wire().expect("inflate_stored must encode");
    let decoded = Program::from_wire(&bytes).expect("inflate_stored must decode");
    assert!(prog.structural_eq(&decoded));
}

#[test]
fn inflate_stored_workgroup_size_is_64_1_1() {
    let prog = inflate_stored_block("input", "output", 10);
    assert_eq!(prog.workgroup_size(), [64, 1, 1]);
}

#[test]
fn inflate_stored_with_zero_input_len_is_constructible() {
    // The validator may reject this, but construction must succeed.
    let prog = inflate_stored_block("input", "output", 0);
    assert_eq!(prog.buffers()[0].count(), 0);
    assert_eq!(prog.buffers()[1].count(), 0);
}

#[test]
fn inflate_stored_with_max_u32_input_len_is_constructible() {
    let prog = inflate_stored_block("input", "output", u32::MAX);
    assert_eq!(prog.buffers()[0].count(), u32::MAX);
    assert_eq!(prog.buffers()[1].count(), u32::MAX);
}
