use super::*;

#[test]
fn decode_empty_ring_counts_slots() {
    let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    let ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();
    let telemetry = RingTelemetry::decode(&control, &ring);
    assert_eq!(telemetry.occupancy.empty, 4);
    assert_eq!(telemetry.occupancy.published, 0);
    assert_eq!(telemetry.slots.len(), 4);
    assert!(telemetry.windows.is_empty());
}

#[test]
fn strict_decode_rejects_trailing_partial_slot() {
    let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    let mut ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
    ring.push(0);
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("Fix: strict telemetry must reject malformed ring snapshots");
    assert!(matches!(err, PipelineError::Backend(_)));
}

#[test]
fn strict_decode_rejects_misaligned_control_snapshot() {
    let mut control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    control.push(0xFF);
    let ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
    let err = RingTelemetry::try_decode(&control, &ring)
        .expect_err("Fix: strict telemetry must reject malformed control snapshots");
    assert!(matches!(err, PipelineError::Backend(_)));
}

/// The control decode error path is the only thing standing between a truncated
/// device readback and a snapshot that reads as a healthy idle kernel, so the
/// error must both name the malformed buffer and carry the corrective action.
#[test]
fn control_try_decode_rejects_short_snapshot_without_panic() {
    let err = ControlSnapshot::try_decode(&[])
        .expect_err("Fix: strict control telemetry decode must reject missing control words");
    let message = err.to_string();
    assert!(
        message.contains("control snapshot"),
        "Fix: strict control decode errors must explain the malformed control buffer: {err}"
    );
    assert!(
        message.contains("Fix: capture the full control buffer"),
        "Fix: strict control decode errors must carry the corrective action: {err}"
    );
}

/// `try_decode_into` is the caller-owned-storage twin, and it must reject the
/// same truncated buffer with the same corrective action instead of leaving a
/// half-written snapshot behind.
#[test]
fn control_try_decode_into_rejects_short_snapshot_and_leaves_output_untouched() {
    let mut control = ResidentWorkQueue::try_encode_control(false, 3, 5).unwrap();
    let done_count_offset = (control::DONE_COUNT as usize) * 4;
    control[done_count_offset..done_count_offset + 4].copy_from_slice(&41u32.to_le_bytes());
    let mut out = ControlSnapshot::try_decode(&control)
        .expect("Fix: a well-formed control buffer must decode");
    assert_eq!(out.done_count, 41);

    let err = ControlSnapshot::try_decode_into(&[], &mut out)
        .expect_err("Fix: strict control telemetry decode_into must reject missing control words");
    assert!(
        err.to_string()
            .contains("Fix: capture the full control buffer"),
        "Fix: strict control decode_into errors must carry the corrective action: {err}"
    );
    assert_eq!(
        out.done_count, 41,
        "Fix: a rejected control buffer must not overwrite the caller's previous snapshot"
    );
}

#[test]
fn strict_decode_into_rejects_trailing_partial_slot_without_mutating_output() {
    let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    let mut ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
    ring.push(0);
    let mut telemetry = RingTelemetry::default();
    let mut scratch = TelemetryDecodeScratch::new();

    let err = RingTelemetry::try_decode_with_window_opcodes_into(
        &control,
        &ring,
        &[],
        &mut telemetry,
        &mut scratch,
    )
    .expect_err("Fix: strict telemetry decode_into must reject partial ring slots");

    assert!(
        err.to_string().contains("whole ring slots"),
        "Fix: strict telemetry decode_into errors must explain partial ring slots: {err}"
    );
    assert!(telemetry.slots.is_empty());
}

#[test]
fn decode_published_slot_reads_prefix() {
    let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 1, 9, opcode::ATOMIC_ADD, &[5, 7, 11]).unwrap();
    let telemetry = RingTelemetry::decode(&control, &ring);
    let slot = &telemetry.slots[1];
    assert_eq!(slot.status, RingStatus::Published);
    assert_eq!(slot.tenant_id, 9);
    assert_eq!(slot.opcode, opcode::ATOMIC_ADD);
    assert_eq!(slot.args_prefix, [5, 7, 11]);
}
