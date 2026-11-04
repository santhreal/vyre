//! Protocol/security edge cases for the megakernel ring and control ABI.
//!
//! Covers:
//! - Slot publish bounds (exact boundary, empty ring)
//! - Packed slot overflow (12-word boundary)
//! - Done/epoch/metrics readback with short buffers
//! - Queue packing validation (BatchDescriptor/WindowDescriptor overflow)
//! - No silent CPU fallback (runtime-level explicit GPU mode selection)
//!
//! The u8 opcode-count field is bounded by `packed_slot_256_ops_fails` in
//! `resident_work_queue_protocol_layout_contracts`, which owns the packed-slot
//! payload budget as one set.

use vyre_runtime::resident_work_queue::{
    descriptor::{BatchDescriptor, BuiltinOpcode, SlotDescriptor, SlotOpcode, WindowDescriptor},
    policy::ResidentExecutionMode,
    protocol::{self, control, slot, ARGS_PER_SLOT, STATUS_WORD},
    ResidentWorkQueue,
};
use vyre_runtime::{PipelineError, RingEncodingFault};

use crate::ring_expectations::{assert_ring_fault, missing_word, protocol_missing_word};
use vyre_test_support::le_words::write_word;

// ---------------------------------------------------------------------------
// 1. Slot publish bounds
// ---------------------------------------------------------------------------

#[test]
fn slot_publish_exact_boundary_last_slot_ok_next_fails() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 3, 0, protocol::opcode::NOP, &[])
        .expect("last slot (slot_count - 1) must be publishable");
    let err = ResidentWorkQueue::publish_slot(&mut ring, 4, 0, protocol::opcode::NOP, &[])
        .expect_err("slot_idx == slot_count must be rejected");
    assert_ring_fault(
        &err,
        RingEncodingFault::OutOfBounds,
        "slot_idx == slot_count is out of bounds",
    );
}

#[test]
fn slot_publish_empty_ring_rejects_any_slot() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(0).unwrap();
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &[])
        .expect_err("empty ring must reject any slot publish");
    assert_ring_fault(
        &err,
        RingEncodingFault::OutOfBounds,
        "a zero-slot ring has no publishable slot",
    );
}

// ---------------------------------------------------------------------------
// 2. Packed slot overflow
// ---------------------------------------------------------------------------

#[test]
fn packed_slot_exact_12_word_boundary_succeeds() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let boundary_ops = vec![(1u8, vec![0u32; 5]), (2u8, vec![0u32; 5])];
    ResidentWorkQueue::publish_packed_slot(&mut ring, 0, 0, &boundary_ops)
        .expect("packed slot with exactly 12 words must succeed");
    let base = (STATUS_WORD as usize) * 4;
    let status = u32::from_le_bytes(ring[base..base + 4].try_into().unwrap());
    assert_eq!(status, slot::PUBLISHED);
}

#[test]
fn packed_slot_13_word_boundary_fails() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let over_ops = vec![(1u8, vec![0u32; 5]), (2u8, vec![0u32; 6])];
    let err = ResidentWorkQueue::publish_packed_slot(&mut ring, 0, 0, &over_ops)
        .expect_err("packed slot with 13 words must fail");
    assert!(
        matches!(
            &err,
            PipelineError::RingEncoding {
                fault: RingEncodingFault::Capacity,
                ..
            }
        ),
        "13 packed words against a 12-word budget is a capacity fault, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("12-word") || msg.contains("exceeds") || msg.contains("budget"),
        "error must mention slot argument budget: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 3. Done/epoch/metrics readback with short buffers
// ---------------------------------------------------------------------------

#[test]
fn try_read_done_count_rejects_buffer_missing_word() {
    // DONE_COUNT is at word 1; 4 bytes only covers word 0.
    let short = vec![0u8; 4];
    let err = protocol::try_read_done_count(&short)
        .expect_err("buffer missing DONE_COUNT word must fail");
    let (buffer, word_idx, byte_len) = missing_word(
        &err,
        "a buffer ending before DONE_COUNT must name that word",
    );
    assert_eq!(buffer, "control");
    assert_eq!(word_idx, control::DONE_COUNT as usize);
    assert_eq!(byte_len, 4);
}

#[test]
fn try_read_epoch_rejects_buffer_missing_epoch_word() {
    let short = vec![0u8; (control::EPOCH as usize) * 4];
    let err = protocol::try_read_epoch(&short).expect_err("buffer missing EPOCH word must fail");
    let (buffer, word_idx, byte_len) = missing_word(
        &err,
        "a buffer ending at the epoch word must name that word",
    );
    assert_eq!(buffer, "control");
    assert_eq!(word_idx, control::EPOCH as usize);
    assert_eq!(byte_len, (control::EPOCH as usize) * 4);
}

#[test]
fn try_read_metrics_rejects_short_buffer() {
    let short = vec![0u8; ((control::METRICS_BASE + 1) as usize) * 4];
    let err =
        ResidentWorkQueue::try_read_metrics(&short).expect_err("short metrics buffer must fail");
    let (buffer, word_idx, byte_len) = protocol_missing_word(
        &err,
        "a short metrics window must name the first word it could not read",
    );
    assert_eq!(buffer, "control");
    assert_eq!(
        word_idx,
        (control::METRICS_BASE + 1) as usize,
        "the strict metrics counter walks the window in order, so the first absent word is the one past the buffer"
    );
    assert_eq!(byte_len, ((control::METRICS_BASE + 1) as usize) * 4);
}

#[test]
fn try_read_metrics_accepts_exact_size_buffer() {
    let exact = vec![0u8; ((control::METRICS_BASE + control::METRICS_SLOTS) as usize) * 4];
    let metrics = ResidentWorkQueue::try_read_metrics(&exact)
        .expect("exact-size metrics buffer must succeed");
    assert!(metrics.is_empty());
}

#[test]
fn try_read_done_count_accepts_minimal_buffer() {
    let mut buf = vec![0u8; (control::DONE_COUNT as usize + 1) * 4];
    write_word(&mut buf, control::DONE_COUNT as usize, 42);
    assert_eq!(protocol::try_read_done_count(&buf).unwrap(), 42);
}

#[test]
fn try_read_epoch_accepts_minimal_buffer() {
    let mut buf = vec![0u8; (control::EPOCH as usize + 1) * 4];
    write_word(&mut buf, control::EPOCH as usize, 7);
    assert_eq!(protocol::try_read_epoch(&buf).unwrap(), 7);
}

// ---------------------------------------------------------------------------
// 4. Queue packing validation
// ---------------------------------------------------------------------------

#[test]
fn batch_descriptor_rejects_items_exceeding_ring_capacity() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    let batch = BatchDescriptor::new(
        0,
        vec![
            SlotDescriptor::single(0, SlotOpcode::Builtin(BuiltinOpcode::Nop), vec![]),
            SlotDescriptor::single(0, SlotOpcode::Builtin(BuiltinOpcode::Nop), vec![]),
            SlotDescriptor::single(0, SlotOpcode::Builtin(BuiltinOpcode::Nop), vec![]),
        ],
    );
    let err = batch
        .publish_into(&mut ring)
        .expect_err("batch exceeding ring must fail");
    assert_ring_fault(
        &err,
        RingEncodingFault::OutOfBounds,
        "a batch wider than the ring reaches past its end",
    );
}

#[test]
fn batch_publish_rejects_u32_slot_index_overflow() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    let batch_items = vec![
        (protocol::opcode::STORE_U32, vec![10, 20]),
        (protocol::opcode::NOP, vec![]),
    ];
    let err = ResidentWorkQueue::batch_publish(&mut ring, u32::MAX - 1, 0, &batch_items, 0)
        .expect_err("batch publish wrapping u32::MAX must fail on OOB ring");
    assert_ring_fault(
        &err,
        RingEncodingFault::Overflow,
        "start_slot near u32::MAX plus the fence slot overflows u32",
    );
}

#[test]
fn window_descriptor_rejects_prefixed_arg_overflow() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    // WindowDescriptor prefixes [ticket, class_tag] (2 words) to the payload.
    // A payload of ARGS_PER_SLOT args makes the total 2 + ARGS_PER_SLOT > ARGS_PER_SLOT.
    let window = WindowDescriptor::new(
        0,
        0,
        SlotOpcode::Builtin(BuiltinOpcode::Nop),
        77,
        vec![vec![0u32; ARGS_PER_SLOT as usize]],
        vec![],
    );
    let err = window
        .publish_into(&mut ring)
        .expect_err("prefixed args exceeding budget must fail");
    assert_ring_fault(
        &err,
        RingEncodingFault::Capacity,
        "ticket and class prefix plus a full arg slot exceeds the budget",
    );
}

// ---------------------------------------------------------------------------
// 5. No silent CPU fallback
// ---------------------------------------------------------------------------

#[test]
fn execution_mode_variants_are_always_gpu() {
    // Interpreter and Jit are both GPU execution modes; there is no CPU variant.
    for mode in [
        ResidentExecutionMode::Interpreter,
        ResidentExecutionMode::Jit,
    ] {
        let name = format!("{mode:?}").to_lowercase();
        assert!(
            !name.contains("cpu"),
            "execution mode {mode:?} must never imply CPU fallback"
        );
    }
}
