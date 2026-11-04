//! Async dispatch observability contracts for synthetic in-flight states.
//!
//! Verifies that telemetry decoders accurately reflect mixed active,
//! terminal, and faulted slot states without requiring a live GPU.

#![cfg(feature = "megakernel-batch")]

use vyre_runtime::resident_work_queue::{
    descriptor::{SlotOpcode, WindowDescriptor},
    policy::{ResidentExecutionMode, ResidentLaunchRequest},
    protocol::{self, control, slot},
    telemetry::RingTelemetry,
    ResidentWorkQueue,
};

fn write_slot_status(ring: &mut [u8], slot_idx: u32, status: u32) {
    let base = (slot_idx as usize) * (protocol::SLOT_WORDS as usize) * 4;
    ring[base..base + 4].copy_from_slice(&status.to_le_bytes());
}

#[test]
fn telemetry_decode_counts_mixed_inflight_states() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(8).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::NOP, &[]).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 1, 2, protocol::opcode::STORE_U32, &[1, 2]).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 2, 3, protocol::opcode::ATOMIC_ADD, &[3, 4])
        .unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 3, 4, protocol::opcode::LOAD_U32, &[5, 6]).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 4, 5, protocol::opcode::COMPARE_SWAP, &[7, 8, 9])
        .unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 5, 6, protocol::opcode::MEMCPY, &[10, 11, 12])
        .unwrap();

    write_slot_status(&mut ring, 2, slot::CLAIMED);
    write_slot_status(&mut ring, 3, slot::DONE);
    write_slot_status(&mut ring, 4, slot::WAIT_IO);
    write_slot_status(&mut ring, 5, slot::YIELD);
    write_slot_status(&mut ring, 6, slot::REQUEUE);
    write_slot_status(&mut ring, 7, slot::FAULT);

    let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    let telemetry =
        RingTelemetry::try_decode(&control, &ring).expect("valid telemetry must decode");

    assert_eq!(telemetry.occupancy.published, 2); // slots 0, 1
    assert_eq!(telemetry.occupancy.claimed, 1); // slot 2
    assert_eq!(telemetry.occupancy.done, 1); // slot 3
    assert_eq!(telemetry.occupancy.wait_io, 1); // slot 4
    assert_eq!(telemetry.occupancy.yield_count, 1); // slot 5
    assert_eq!(telemetry.occupancy.requeue, 1); // slot 6
    assert_eq!(telemetry.occupancy.fault, 1); // slot 7
    assert_eq!(telemetry.occupancy.empty, 0);
}

#[test]
fn active_slots_for_opcode_filters_only_inflight() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    let op = 0xBEEF;
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, op, &[1]).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 1, 0, op, &[2]).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 2, 0, op, &[3]).unwrap();
    write_slot_status(&mut ring, 2, slot::DONE);

    let telemetry = RingTelemetry::try_decode(
        &ResidentWorkQueue::encode_control(false, 1, 0).unwrap(),
        &ring,
    )
    .expect("valid telemetry must decode");
    let active = telemetry
        .try_active_slots_for_opcode(op)
        .expect("active telemetry slots must collect");
    assert_eq!(active.len(), 2, "only PUBLISHED slots count as active");
    assert_eq!(active[0].slot_idx, 0);
    assert_eq!(active[1].slot_idx, 1);
}

#[test]
fn active_windows_excludes_fully_terminal_windows() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(3).unwrap();
    let window_opcode = 0xF103;
    let window = WindowDescriptor::new(
        0,
        7,
        SlotOpcode::Custom(window_opcode),
        42,
        vec![vec![10], vec![20]],
        vec![vec![30]],
    );
    window.publish_into(&mut ring).unwrap();

    write_slot_status(&mut ring, 0, slot::DONE);

    let telemetry = RingTelemetry::try_decode_with_window_opcodes(
        &ResidentWorkQueue::encode_control(false, 1, 0).unwrap(),
        &ring,
        &[window_opcode],
    )
    .expect("valid window telemetry must decode");
    assert_eq!(telemetry.windows.len(), 1);
    assert!(
        telemetry.windows[0].is_active(),
        "partially done window is still active"
    );
    assert_eq!(
        telemetry
            .try_active_windows()
            .expect("active telemetry windows must collect")
            .len(),
        1
    );

    write_slot_status(&mut ring, 1, slot::DONE);
    write_slot_status(&mut ring, 2, slot::DONE);
    let telemetry2 = RingTelemetry::try_decode_with_window_opcodes(
        &ResidentWorkQueue::encode_control(false, 1, 0).unwrap(),
        &ring,
        &[window_opcode],
    )
    .expect("valid window telemetry must decode");
    assert!(
        !telemetry2.windows[0].is_active(),
        "fully done window is not active"
    );
    assert!(telemetry2
        .try_active_windows()
        .expect("active telemetry windows must collect")
        .is_empty());
}

#[test]
fn priority_accounting_reflects_requeue_pressure() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    for i in 0..4 {
        ResidentWorkQueue::publish_slot(&mut ring, i, 0, protocol::opcode::NOP, &[]).unwrap();
        write_slot_status(&mut ring, i, slot::REQUEUE);
    }
    let telemetry = RingTelemetry::try_decode(
        &ResidentWorkQueue::encode_control(false, 1, 0).unwrap(),
        &ring,
    )
    .expect("valid telemetry must decode");
    let accounting = telemetry.priority_accounting();
    assert_eq!(accounting.requeue_count, 4);
}

#[test]
fn recommend_launch_from_mixed_pressure_ring_selects_jit() {
    let mut control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    for i in 0..8u32 {
        let off = ((control::METRICS_BASE + i) as usize) * 4;
        control[off..off + 4].copy_from_slice(&1u32.to_le_bytes());
    }
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    write_slot_status(&mut ring, 0, slot::REQUEUE);
    write_slot_status(&mut ring, 1, slot::YIELD);

    let telemetry =
        RingTelemetry::try_decode(&control, &ring).expect("valid telemetry must decode");
    let rec = telemetry
        .recommend_launch(ResidentLaunchRequest::direct(4096, 64, 256))
        .expect("telemetry must produce launch recommendation");
    assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
    assert!(rec.promote_hot_opcodes);
    assert!(rec.age_priority_work);
}
