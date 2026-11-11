//! Adversarial metrics-decode sanity: hostile control buffers, truncation
//! boundaries, misalignment, and region-non-alias contracts.

use vyre_runtime::resident_work_queue::telemetry::{ControlSnapshot, RingTelemetry};
use vyre_runtime::resident_work_queue::{
    protocol::{control, control_byte_len, ProtocolError},
    ResidentWorkQueue,
};
use vyre_runtime::PipelineError;

use crate::ring_expectations::protocol_missing_word;
use vyre_test_support::le_words::write_word;

// ---------------------------------------------------------------------------
// 1. Metrics truncation / misalignment boundaries
// ---------------------------------------------------------------------------

#[test]
fn try_read_metrics_rejects_buffer_one_word_short_of_full_window() {
    let words = (control::METRICS_BASE + control::METRICS_SLOTS - 1) as usize;
    let short = vec![0u8; words * 4];
    let err = ResidentWorkQueue::try_read_metrics(&short)
        .expect_err("buffer one word short of metrics window must reject");
    let (buffer, word_idx, byte_len) = protocol_missing_word(
        &err,
        "a truncated metrics window must name the word it could not read",
    );
    assert_eq!(buffer, "control");
    assert_eq!(
        word_idx,
        (control::METRICS_BASE + control::METRICS_SLOTS - 1) as usize,
        "the fault must name the first metrics word past the buffer"
    );
    assert_eq!(byte_len, words * 4);
}

#[test]
fn try_read_metrics_rejects_misaligned_buffer() {
    let mut buf = vec![0u8; ((control::METRICS_BASE + control::METRICS_SLOTS) as usize) * 4];
    buf.push(0xAA);
    let err = ResidentWorkQueue::try_read_metrics(&buf)
        .expect_err("misaligned metrics buffer must reject");
    let PipelineError::Protocol(ProtocolError::MisalignedByteLength {
        buffer, byte_len, ..
    }) = err
    else {
        panic!("a misaligned buffer must reject for alignment, not for truncation, got {err:?}")
    };
    assert_eq!(buffer, "control");
    assert_eq!(
        byte_len,
        ((control::METRICS_BASE + control::METRICS_SLOTS) as usize) * 4 + 1,
        "the fault must report the byte length that was not a whole word count"
    );
}

#[test]
fn read_metrics_gracefully_truncates_on_short_buffer() {
    // Provide enough bytes for 5 metrics but not the full 32.
    let mut buf = vec![0u8; ((control::METRICS_BASE + 5) as usize) * 4];
    for i in 0..5 {
        write_word(&mut buf, (control::METRICS_BASE + i) as usize, i + 1);
    }
    let metrics = ResidentWorkQueue::read_metrics(&buf);
    assert_eq!(metrics.len(), 5);
    for i in 0..5 {
        assert!(metrics.contains(&(i, (i + 1))));
    }
}

// ---------------------------------------------------------------------------
// 2. Hostile / extreme metric values
// ---------------------------------------------------------------------------

#[test]
fn read_metrics_preserves_u32_max_counts() {
    let mut buf = vec![0u8; ((control::METRICS_BASE + control::METRICS_SLOTS) as usize) * 4];
    for i in 0..control::METRICS_SLOTS {
        write_word(&mut buf, (control::METRICS_BASE + i) as usize, u32::MAX);
    }
    let metrics = ResidentWorkQueue::read_metrics(&buf);
    assert_eq!(metrics.len(), control::METRICS_SLOTS as usize);
    for i in 0..control::METRICS_SLOTS {
        assert!(metrics.contains(&(i, u32::MAX)));
    }
}

#[test]
fn control_snapshot_decode_preserves_max_done_count_and_epoch() {
    let mut buf = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    write_word(&mut buf, control::DONE_COUNT as usize, u32::MAX);
    write_word(&mut buf, control::EPOCH as usize, u32::MAX);
    let snapshot = ControlSnapshot::try_decode(&buf).expect("control snapshot decode");
    assert_eq!(snapshot.done_count, u32::MAX);
    assert_eq!(snapshot.epoch, u32::MAX);
    assert!(!snapshot.shutdown);
}

#[test]
fn control_snapshot_decode_skips_zero_metrics() {
    let mut buf = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    write_word(&mut buf, (control::METRICS_BASE + 3) as usize, 99);
    let snapshot = ControlSnapshot::try_decode(&buf).expect("control snapshot decode");
    assert_eq!(snapshot.metrics, vec![(3, 99)]);
}

// ---------------------------------------------------------------------------
// 3. Region non-alias sanity
// ---------------------------------------------------------------------------

#[test]
fn metrics_slot_31_does_not_alias_epoch_word() {
    let mut buf = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    let last_metric = control::METRICS_BASE + control::METRICS_SLOTS - 1;
    assert!(
        last_metric < control::EPOCH,
        "last metric slot must be strictly before epoch word"
    );
    write_word(&mut buf, last_metric as usize, 0xDEAD_BEEF);
    write_word(&mut buf, control::EPOCH as usize, 0x1111_2222);
    let snapshot = ControlSnapshot::try_decode(&buf).expect("control snapshot decode");
    assert!(snapshot
        .metrics
        .contains(&(control::METRICS_SLOTS - 1, 0xDEAD_BEEF)));
    assert_eq!(snapshot.epoch, 0x1111_2222);
}

#[test]
fn metrics_decode_sanity_on_hostile_control_with_mixed_values() {
    let mut control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    for i in 0..control::METRICS_SLOTS {
        write_word(
            &mut control,
            (control::METRICS_BASE + i) as usize,
            (i + 1).wrapping_mul(0x1111_1111),
        );
    }
    let ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let telemetry = RingTelemetry::try_decode(&control, &ring).expect("telemetry decode");
    assert_eq!(
        telemetry.control.metrics.len(),
        control::METRICS_SLOTS as usize,
        "every non-zero hostile metric must be captured"
    );
    // Slot 0 with value 0 may or may not appear depending on decode; verify explicitly.
    let m = &telemetry.control.metrics;
    assert!(m.iter().any(|(idx, _)| *idx == 1));
    assert!(m.iter().any(|(idx, _)| *idx == control::METRICS_SLOTS - 1));
}

#[test]
fn strict_ring_telemetry_rejects_control_shorter_than_metrics_window() {
    let control = vec![0u8; (control::METRICS_BASE as usize) * 4];
    let ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("control shorter than fixed metrics window must reject");
    let PipelineError::Backend(message) = err else {
        panic!("a control buffer short of the metrics window must reject as a control snapshot fault, got {err:?}")
    };
    let min_control =
        control_byte_len(0).expect("the minimum control length must be representable");
    assert!(
        message.contains(&format!(
            "control snapshot has {} bytes, expected at least {min_control} and 4-byte alignment",
            (control::METRICS_BASE as usize) * 4
        )),
        "the fault must report the byte length it received and the minimum it required: {message}"
    );
}
