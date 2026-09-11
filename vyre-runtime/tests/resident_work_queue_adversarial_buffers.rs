//! Adversarial buffer contracts: malformed ring/control/debug buffers and
//! hostile publish-slot boundary conditions.

use vyre_runtime::resident_work_queue::{
    protocol::{self, debug, slot, ProtocolError, ARGS_PER_SLOT, SLOT_WORDS},
    telemetry::RingTelemetry,
    ResidentWorkQueue, RingSlotTransition,
};
use vyre_runtime::{PipelineError, RingEncodingFault};

use crate::ring_expectations::{assert_publish_rejected_by_status, assert_ring_fault};
use vyre_test_support::le_words::write_word;

// ---------------------------------------------------------------------------
// 1. Malformed ring buffers
// ---------------------------------------------------------------------------

#[test]
fn publish_slot_rejects_ring_one_byte_under_slot_multiple() {
    let mut ring = vec![0u8; (SLOT_WORDS as usize * 4) - 1];
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &[])
        .expect_err("ring one byte under slot multiple must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::Geometry,
        "a ring one byte under a slot multiple is malformed geometry",
    );
}

#[test]
fn publish_slot_rejects_ring_one_byte_over_slot_multiple() {
    let mut ring = vec![0u8; (SLOT_WORDS as usize * 4) + 1];
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &[])
        .expect_err("ring one byte over slot multiple must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::Geometry,
        "a ring one byte over a slot multiple is malformed geometry",
    );
}

#[test]
fn batch_publish_rejects_truncated_ring_length() {
    let mut ring = vec![0u8; (SLOT_WORDS as usize * 4) / 2];
    let err =
        ResidentWorkQueue::batch_publish(&mut ring, 0, 0, &[(protocol::opcode::NOP, vec![])], 0)
            .expect_err("batch publish on truncated ring must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::Geometry,
        "a half-slot ring is malformed geometry",
    );
}

#[test]
fn publish_packed_slot_rejects_ring_with_non_slot_multiple_length() {
    let mut ring = vec![0u8; (SLOT_WORDS as usize * 4) + 3];
    let err = ResidentWorkQueue::publish_packed_slot(
        &mut ring,
        0,
        0,
        &[(protocol::opcode::NOP as u8, vec![])],
    )
    .expect_err("packed slot on non-slot-multiple ring must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::Geometry,
        "a non-slot-multiple ring is malformed geometry",
    );
}

#[test]
fn strict_ring_telemetry_rejects_ring_one_byte_under_slot_multiple() {
    let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    ring.pop();
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("ring one byte under slot multiple must reject strict decode");
    let PipelineError::Backend(message) = err else {
        panic!("a partial trailing slot must reject as a ring alignment fault, got {err:?}")
    };
    assert!(
        message.contains(&format!(
            "ring snapshot has {} bytes, not a multiple of slot size {}",
            (SLOT_WORDS as usize * 4) * 2 - 1,
            SLOT_WORDS as usize * 4
        )),
        "the fault must report the byte length it received and the slot width it required: {message}"
    );
}

#[test]
fn strict_ring_telemetry_rejects_ring_one_byte_over_slot_multiple() {
    let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    ring.push(0xAA);
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("ring one byte over slot multiple must reject strict decode");
    let PipelineError::Backend(message) = err else {
        panic!("a trailing partial byte must reject as a ring alignment fault, got {err:?}")
    };
    assert!(
        message.contains(&format!(
            "ring snapshot has {} bytes, not a multiple of slot size {}",
            (SLOT_WORDS as usize * 4) * 2 + 1,
            SLOT_WORDS as usize * 4
        )),
        "the fault must report the byte length it received and the slot width it required: {message}"
    );
}

// ---------------------------------------------------------------------------
// 2. Malformed control / debug buffers
// ---------------------------------------------------------------------------

#[test]
fn strict_ring_telemetry_rejects_control_one_byte_over_word_boundary() {
    let mut control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    control.push(0xBB);
    let ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("control one byte over word boundary must reject strict decode");
    // Control is validated before ring geometry, so a well-formed ring cannot
    // mask this fault.
    let PipelineError::Backend(message) = err else {
        panic!("a control buffer past a word boundary must reject as a control snapshot fault, got {err:?}")
    };
    let min_control =
        protocol::control_byte_len(0).expect("the minimum control length must be representable");
    assert!(
        message.contains(&format!(
            "control snapshot has {} bytes, expected at least {min_control} and 4-byte alignment",
            min_control + 1
        )),
        "the fault must report the byte length it received and the minimum it required: {message}"
    );
}

#[test]
fn encode_empty_debug_log_with_zero_capacity_produces_minimal_buffer() {
    let log = ResidentWorkQueue::encode_empty_debug_log(0).unwrap();
    let expected = (debug::RECORDS_BASE as usize) * 4;
    assert_eq!(
        log.len(),
        expected,
        "zero-capacity debug log must be exactly RECORDS_BASE words"
    );
    let records = ResidentWorkQueue::read_debug_log(&log);
    assert!(records.is_empty());
}

#[test]
fn try_encode_empty_debug_log_rejects_overflow_capacity() {
    let err =
        protocol::try_encode_empty_debug_log(u32::MAX).expect_err("u32::MAX records must overflow");
    let ProtocolError::ByteLengthOverflow { buffer, .. } = err else {
        panic!("an oversized record capacity must reject for byte-length overflow, got {err:?}")
    };
    assert_eq!(buffer, "debug_log");
}

// ---------------------------------------------------------------------------
// 3. Publish-slot bounds (adversarial edge cases)
// ---------------------------------------------------------------------------

#[test]
fn publish_slot_accepts_exact_args_budget() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let args = vec![0xDEAD_BEEF; ARGS_PER_SLOT as usize];
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &args)
        .expect("exact args budget must be accepted");
    let status = u32::from_le_bytes(ring[..4].try_into().unwrap());
    assert_eq!(status, slot::PUBLISHED);
}

#[test]
fn publish_slot_rejects_args_one_over_budget() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let args = vec![0u32; ARGS_PER_SLOT as usize + 1];
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &args)
        .expect_err("one arg over budget must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::Capacity,
        "one arg over the per-slot budget is a capacity fault",
    );
}

#[test]
fn publish_slot_rejects_slot_count_exactly_at_boundary() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 3, 0, protocol::opcode::NOP, &[])
        .expect("last valid slot must accept");
    let err = ResidentWorkQueue::publish_slot(&mut ring, 4, 0, protocol::opcode::NOP, &[])
        .expect_err("slot_idx == slot_count must reject");
    assert_ring_fault(
        &err,
        RingEncodingFault::OutOfBounds,
        "slot_idx == slot_count is out of bounds",
    );
}

/// A slot whose status word is already inflight must never be re-published, and
/// the rejection must name the status it actually found plus the two statuses
/// that are publishable. The hostile set is derived from `slot::STATUSES` rather
/// than listed, so a new status is hostile by default until it is classified.
#[test]
fn publish_slot_rejects_on_hostile_inflight_status_garbage() {
    let publishable = [slot::EMPTY, slot::DONE];
    let hostile: Vec<(u32, &str)> = slot::STATUSES
        .iter()
        .copied()
        .filter(|(status, _)| !publishable.contains(status))
        .collect();
    assert_eq!(
        hostile.len(),
        slot::STATUSES.len() - publishable.len(),
        "every status is either publishable or hostile"
    );

    assert_eq!(
        RingSlotTransition::Publish.permitted(),
        publishable.as_slice(),
        "publish is legal from EMPTY and DONE only"
    );

    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    for (hostile_status, name) in hostile {
        write_word(&mut ring, protocol::STATUS_WORD as usize, hostile_status);
        let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &[])
            .expect_err(&format!("hostile status {name} must block re-publish"));
        assert_publish_rejected_by_status(
            &err,
            hostile_status,
            &format!("a publish from hostile status {name}"),
        );
    }
}

#[test]
fn publish_slot_recycles_done_slot_and_clears_stale_opcode() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::STORE_U32, &[1, 2, 3])
        .unwrap();
    write_word(&mut ring, protocol::STATUS_WORD as usize, slot::DONE);
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, protocol::opcode::NOP, &[9]).unwrap();
    let words: Vec<u32> = ring
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(words[protocol::OPCODE_WORD as usize], protocol::opcode::NOP);
    assert_eq!(words[protocol::ARG0_WORD as usize], 9);
    assert_eq!(words[protocol::ARG0_WORD as usize + 1], 0);
}

#[test]
fn encode_control_with_zero_tenants_and_zero_observables_is_minimal() {
    let ctrl = ResidentWorkQueue::encode_control(false, 0, 0).unwrap();
    let min = protocol::control_byte_len(0).expect("control length must fit");
    assert_eq!(ctrl.len(), min);
    assert_eq!(ResidentWorkQueue::read_done_count(&ctrl), 0);
    assert_eq!(ResidentWorkQueue::read_epoch(&ctrl), 0);
}
