//! Adversarial metrics-decode sanity: hostile control buffers, truncation
//! boundaries, misalignment, and region-non-alias contracts.

use vyre_runtime::resident_work_queue::{
    protocol::{control, ProtocolError},
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
    let metrics = ResidentWorkQueue::read_metrics(&buf);
    assert!(metrics.contains(&(control::METRICS_SLOTS - 1, 0xDEAD_BEEF)));
    assert!(
        !metrics.contains(&(control::METRICS_SLOTS - 1, 0x1111_2222)),
        "the epoch word must not decode as a metric"
    );
}
