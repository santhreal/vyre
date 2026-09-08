//! Unbounded allocation rejection for megakernel buffer encoders.
//!
//! Verifies that fallible encoder entrypoints reject inputs that would
//! overflow the protocol byte-length calculations *before* allocating
//! host-side Vec<u8> buffers.

use crate::ring_expectations::assert_ring_fault;
use vyre_runtime::resident_work_queue::{
    io::ResidentIoQueue,
    protocol::{self, control, debug, ProtocolError, SLOT_WORDS},
    ResidentWorkQueue,
};
use vyre_runtime::{PipelineError, RingEncodingFault};

#[test]
fn try_encode_empty_ring_rejects_overflowing_slot_count() {
    let too_many = (u32::MAX / SLOT_WORDS) + 1;
    let err = ResidentWorkQueue::try_encode_empty_ring(too_many)
        .expect_err("overflowing slot count must be rejected before allocation");
    // `ring_encode_capacity` checks the allocation cap first, so a slot count
    // past the protocol cap is reported as a ring byte-length overflow.
    let PipelineError::Protocol(ProtocolError::ByteLengthOverflow { buffer, fix }) = err else {
        panic!("a ring byte-length overflow must surface the typed protocol fault, got {err:?}")
    };
    assert_eq!(buffer, "ring");
    assert!(
        fix.contains("ring shards"),
        "the remediation must state how to get under the cap: {fix}"
    );
}

#[test]
fn try_encode_empty_debug_log_rejects_overflowing_record_capacity() {
    let too_many = (u32::MAX / debug::RECORD_WORDS) + 1;
    let err = ResidentWorkQueue::try_encode_empty_debug_log(too_many)
        .expect_err("overflowing record capacity must be rejected before allocation");
    let PipelineError::Protocol(ProtocolError::ByteLengthOverflow { buffer, .. }) = err else {
        panic!(
            "a debug-log byte-length overflow must surface the typed protocol fault, got {err:?}"
        )
    };
    assert_eq!(buffer, "debug_log");
}

#[test]
fn try_encode_control_rejects_overflow_at_observable_boundary() {
    let overflow_observable = u32::MAX - control::OBSERVABLE_BASE + 1;
    let err = ResidentWorkQueue::try_encode_control(false, 1, overflow_observable)
        .expect_err("observable word offset overflow must be rejected");
    let PipelineError::Protocol(ProtocolError::ByteLengthOverflow { buffer, .. }) = err else {
        panic!("a control byte-length overflow must surface the typed protocol fault, got {err:?}")
    };
    assert_eq!(buffer, "control");
}

#[test]
fn try_encode_empty_io_queue_rejects_u32_max_before_alloc() {
    let err = vyre_runtime::resident_work_queue::io::try_encode_empty_io_queue(u32::MAX)
        .expect_err("u32::MAX io queue must be rejected before allocation");
    assert_ring_fault(
        &err,
        RingEncodingFault::Capacity,
        "u32::MAX slots exceeds the compiled 64-slot poll window",
    );
}

#[test]
fn megakernel_io_queue_new_rejects_u32_max() {
    let err = ResidentIoQueue::new(u32::MAX)
        .expect_err("u32::MAX io queue must be rejected before allocation");
    assert_ring_fault(
        &err,
        RingEncodingFault::Capacity,
        "u32::MAX slots exceeds the compiled 64-slot poll window",
    );
}

#[test]
fn batch_publish_rejects_slot_index_overflow_before_allocating_extra_ring() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let err = ResidentWorkQueue::batch_publish(
        &mut ring,
        u32::MAX,
        0,
        &[(protocol::opcode::NOP, vec![])],
        0,
    )
    .expect_err("batch publish slot-index overflow must be rejected");
    assert_ring_fault(
        &err,
        RingEncodingFault::Overflow,
        "start_slot u32::MAX plus the fence slot overflows u32",
    );
}
