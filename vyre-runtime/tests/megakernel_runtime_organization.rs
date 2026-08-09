//! Resident work-queue protocol and host-mirror contracts.

use vyre_runtime::resident_work_queue::{
    protocol, ResidentQueueBuffers, ResidentQueueReadback, ResidentWorkQueue,
};


#[test]
fn readback_rejects_truncated_ring_before_telemetry() {
    let control = ResidentWorkQueue::try_encode_control(false, 1, 0).unwrap();
    let ring = vec![0_u8; 4];
    let debug = ResidentWorkQueue::try_encode_empty_debug_log(protocol::debug::RECORD_CAPACITY).unwrap();
    let io = vyre_runtime::resident_work_queue::io::try_encode_empty_io_queue(
        vyre_runtime::resident_work_queue::io::IO_SLOT_COUNT,
    )
    .unwrap();

    let error = ResidentQueueReadback::from_outputs(vec![control, ring, debug, io], 2)
        .expect_err("truncated ring readback must fail before telemetry decode");
    assert!(error.to_string().contains("readback ring"));
}

#[test]
fn resident_buffers_preserve_multitenant_slot_headers() {
    let mut resident = ResidentQueueBuffers::new(4, 8, 2).unwrap();
    resident
        .publish_slot(0, 3, protocol::opcode::STORE_U32, &[11, 12])
        .unwrap();
    resident
        .publish_slot(1, 7, protocol::opcode::ATOMIC_ADD, &[13, 14])
        .unwrap();

    let slot_bytes = protocol::SLOT_WORDS as usize * 4;
    let tenant0 = u32::from_le_bytes(
        resident.ring_bytes()
            [protocol::TENANT_WORD as usize * 4..protocol::TENANT_WORD as usize * 4 + 4]
            .try_into()
            .unwrap(),
    );
    let slot1_base = slot_bytes;
    let tenant1 = u32::from_le_bytes(
        resident.ring_bytes()[slot1_base + protocol::TENANT_WORD as usize * 4
            ..slot1_base + protocol::TENANT_WORD as usize * 4 + 4]
            .try_into()
            .unwrap(),
    );

    assert_eq!(tenant0, 3);
    assert_eq!(tenant1, 7);
}
