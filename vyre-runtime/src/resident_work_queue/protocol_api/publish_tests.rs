use super::*;
use crate::resident_work_queue::protocol::{
    ARG0_WORD, ARGS_PER_SLOT, OPCODE_WORD, PRIORITY_WORD, STATUS_WORD, TENANT_WORD,
};

#[test]
fn publish_slot_writes_and_reads_back() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 0, 42, protocol::opcode::STORE_U32, &[100, 200])
        .unwrap();

    // Verify status is PUBLISHED.
    let status = read_word(&ring, 0, STATUS_WORD as usize);
    assert_eq!(status, slot::PUBLISHED);

    // Verify opcode.
    let op = read_word(&ring, 0, OPCODE_WORD as usize);
    assert_eq!(op, protocol::opcode::STORE_U32);

    // Verify tenant.
    let tenant = read_word(&ring, 0, TENANT_WORD as usize);
    assert_eq!(tenant, 42);

    let priority = read_word(&ring, 0, PRIORITY_WORD as usize);
    assert_eq!(priority, scheduler::priority::NORMAL);

    // Verify args.
    let a0 = read_word(&ring, 0, ARG0_WORD as usize);
    let a1 = read_word(&ring, 0, ARG0_WORD as usize + 1);
    assert_eq!(a0, 100);
    assert_eq!(a1, 200);
}

#[test]
fn publish_slot_rejects_inflight_slot() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    // Publish once (now status = PUBLISHED).
    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1]).unwrap();
    // Try to publish again  -  slot is PUBLISHED (not EMPTY/DONE).
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[2])
        .expect_err("must reject publishing to an in-flight slot");
    match err {
        PipelineError::IllegalSlotTransition {
            transition,
            permitted,
            current_status,
            ..
        } => {
            assert_eq!(transition, "publish");
            assert_eq!(permitted, [slot::EMPTY, slot::DONE].as_slice());
            assert_eq!(current_status, slot::PUBLISHED);
        }
        other => panic!("expected an illegal publish transition, got {other:?}"),
    }
}

#[test]
fn publish_slot_rejects_out_of_bounds() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    let err = ResidentWorkQueue::publish_slot(&mut ring, 99, 1, protocol::opcode::STORE_U32, &[1])
        .expect_err("must reject slot_idx beyond ring capacity");
    let PipelineError::RingEncoding { fault, .. } = err else {
        panic!("a slot index past the ring slot count must report a ring encode fault, got {err:?}")
    };
    assert_eq!(fault, RingEncodingFault::OutOfBounds);
}

#[test]
fn publish_slot_rejects_too_many_args() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    let too_many = vec![0u32; ARGS_PER_SLOT as usize + 1];
    let err =
        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &too_many)
            .expect_err("must reject args exceeding ARGS_PER_SLOT");
    let PipelineError::RingEncoding { fault, .. } = err else {
        panic!("one argument past the per-slot budget must report a ring encode fault, got {err:?}")
    };
    assert_eq!(fault, RingEncodingFault::Capacity);
}

#[test]
fn publish_slot_allows_republish_after_done() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    // Publish, then manually mark as DONE.
    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1]).unwrap();
    write_word(&mut ring, 0, STATUS_WORD as usize, slot::DONE);
    // Should succeed  -  DONE slots are recyclable.
    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::ATOMIC_ADD, &[2]).unwrap();
    let op = read_word(&ring, 0, OPCODE_WORD as usize);
    assert_eq!(op, protocol::opcode::ATOMIC_ADD);
}

#[test]
fn ring_slot_transition_state_machine_accepts_legal_lifecycle() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1]).unwrap();

    let previous =
        ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Claim)
            .expect("Fix: PUBLISHED slots must be claimable");
    assert_eq!(previous, slot::PUBLISHED);
    assert_eq!(read_word(&ring, 0, STATUS_WORD as usize), slot::CLAIMED);

    let previous =
        ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Done)
            .expect("Fix: CLAIMED slots must complete to DONE");
    assert_eq!(previous, slot::CLAIMED);
    assert_eq!(read_word(&ring, 0, STATUS_WORD as usize), slot::DONE);

    ResidentWorkQueue::publish_slot(&mut ring, 1, 1, protocol::opcode::STORE_U32, &[2]).unwrap();
    ResidentWorkQueue::transition_slot_status(&mut ring, 1, RingSlotTransition::Cancel)
        .expect("Fix: unclaimed published slots must be cancellable");
    assert_eq!(read_word(&ring, 1, STATUS_WORD as usize), slot::EMPTY);

    ResidentWorkQueue::publish_slot(&mut ring, 2, 1, protocol::opcode::STORE_U32, &[3]).unwrap();
    ResidentWorkQueue::transition_slot_status(&mut ring, 2, RingSlotTransition::Fault)
        .expect("Fix: in-flight published slots must transition to FAULT");
    assert_eq!(read_word(&ring, 2, STATUS_WORD as usize), slot::FAULT);
}

#[test]
fn ring_slot_transition_state_machine_rejects_illegal_edges_without_mutation() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();

    let err = ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Done)
        .expect_err("EMPTY slots cannot complete");
    let PipelineError::IllegalSlotTransition {
        transition,
        permitted,
        current_status,
        ..
    } = &err
    else {
        panic!("an EMPTY slot must reject a done transition as illegal, got {err:?}")
    };
    assert_eq!(*transition, RingSlotTransition::Done.label());
    assert_eq!(*permitted, RingSlotTransition::Done.permitted());
    assert_eq!(*current_status, slot::EMPTY);
    // The permitted set is asserted above; this pins how it renders, which
    // is the only place a caller reads a status word by name.
    assert!(
        err.to_string().contains("done requires CLAIMED"),
        "Fix: the rejection must render the required source state by name, got: {err}"
    );
    assert_eq!(read_word(&ring, 0, STATUS_WORD as usize), slot::EMPTY);

    ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1]).unwrap();
    ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Claim).unwrap();
    let before = ring.clone();
    let err = ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Cancel)
        .expect_err("CLAIMED slots are worker-owned and cannot be cancelled by host");
    match err {
        PipelineError::IllegalSlotTransition {
            transition,
            permitted,
            current_status,
            fix,
        } => {
            assert_eq!(transition, "cancel");
            assert!(
                !permitted.contains(&slot::CLAIMED),
                "Fix: cancel must never permit a lane-owned CLAIMED slot"
            );
            assert_eq!(current_status, slot::CLAIMED);
            assert!(
                fix.contains("owned by a lane"),
                "Fix: the remediation must name the ownership boundary, got: {fix}"
            );
        }
        other => panic!("expected an illegal cancel transition, got {other:?}"),
    }
    assert_eq!(ring, before);

    let err = ResidentWorkQueue::transition_slot_status(&mut ring, 1, RingSlotTransition::Publish)
        .expect_err("status-only publish is forbidden");
    let PipelineError::RingEncoding { fault, fix } = err else {
        panic!("a status-only publish must report a ring encode fault, got {err:?}")
    };
    assert_eq!(fault, RingEncodingFault::Protocol);
    assert!(
        fix.contains("publish_slot"),
        "Fix: the remediation must direct callers to the payload-safe API, got: {fix}"
    );
    assert_eq!(read_word(&ring, 1, STATUS_WORD as usize), slot::EMPTY);
}

#[test]
fn batch_publish_writes_items_plus_fence() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(8).unwrap();
    let items: Vec<(u32, Vec<u32>)> = vec![
        (protocol::opcode::STORE_U32, vec![10, 20]),
        (protocol::opcode::ATOMIC_ADD, vec![30, 40]),
    ];
    let slots_used = ResidentWorkQueue::batch_publish(&mut ring, 0, 1, &items, 99).unwrap();
    // 2 items + 1 fence = 3 slots consumed.
    assert_eq!(slots_used, 3);

    // Last slot should be BATCH_FENCE.
    let fence_op = read_word(&ring, 2, OPCODE_WORD as usize);
    assert_eq!(fence_op, protocol::opcode::BATCH_FENCE);
}

#[test]
fn batch_publish_rejects_fence_collision_without_partial_publish() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    write_word(&mut ring, 1, STATUS_WORD as usize, slot::PUBLISHED);
    let before = ring.clone();
    let items: Vec<(u32, Vec<u32>)> = vec![(protocol::opcode::STORE_U32, vec![10, 20])];

    let result = ResidentWorkQueue::batch_publish(&mut ring, 0, 1, &items, 99);

    assert!(result.is_err(), "fence collision must reject the batch");
    assert_eq!(ring, before, "rejection must not publish earlier slots");
}

#[test]
fn packed_slot_publish_roundtrips() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    let ops: Vec<(u8, Vec<u32>)> = vec![
        (protocol::opcode::STORE_U32 as u8, vec![10, 20]),
        (protocol::opcode::ATOMIC_ADD as u8, vec![30]),
    ];
    ResidentWorkQueue::publish_packed_slot(&mut ring, 0, 1, &ops).unwrap();

    let status = read_word(&ring, 0, STATUS_WORD as usize);
    assert_eq!(status, slot::PUBLISHED);

    let op = read_word(&ring, 0, OPCODE_WORD as usize);
    assert_eq!(op, protocol::opcode::PACKED_SLOT);
}

#[test]
fn packed_slot_rejects_overflow() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    // Each op gets 3 arg words, so 5 ops × 3 args = 15 words > 12 budget.
    let ops: Vec<(u8, Vec<u32>)> = (0..5).map(|i| (i as u8, vec![1, 2, 3])).collect();
    let err = ResidentWorkQueue::publish_packed_slot(&mut ring, 0, 1, &ops)
        .expect_err("must reject packed slot exceeding arg budget");
    assert!(
        err.to_string()
            .contains("exceeds the 12-word slot argument budget"),
        "unexpected error: {err}"
    );
}
// Helper: read a u32 word from a ring buffer at (slot_idx, word_idx).
fn read_word(ring: &[u8], slot_idx: usize, word_idx: usize) -> u32 {
    let base = slot_idx * SLOT_WORDS as usize * 4;
    let off = base + word_idx * 4;
    u32::from_le_bytes([ring[off], ring[off + 1], ring[off + 2], ring[off + 3]])
}

// Helper: read a native u32 word from a ring-word buffer at (slot_idx, word_idx).
fn read_word_words(ring: &[u32], slot_idx: usize, word_idx: usize) -> u32 {
    ring[slot_idx * SLOT_WORDS as usize + word_idx]
}

// Helper: write a u32 word into a ring buffer at (slot_idx, word_idx).
fn write_word(ring: &mut [u8], slot_idx: usize, word_idx: usize, value: u32) {
    let base = slot_idx * SLOT_WORDS as usize * 4;
    let off = base + word_idx * 4;
    ring[off..off + 4].copy_from_slice(&value.to_le_bytes());
}
mod publish_contracts {
    use super::*;

    #[test]
    fn encode_work_items_ring_into_publishes_contiguous_slots() {
        let items = [
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 10,
                output_handle: 20,
                param: 30,
            },
            ResidentWorkItem {
                op_handle: protocol::opcode::ATOMIC_ADD,
                input_handle: 40,
                output_handle: 50,
                param: 60,
            },
        ];
        let mut ring = vec![0xAA; 4096];

        ResidentWorkQueue::encode_work_items_ring_into(4, 7, &items, &mut ring).unwrap();

        assert_eq!(read_word(&ring, 0, STATUS_WORD as usize), slot::PUBLISHED);
        assert_eq!(
            read_word(&ring, 0, OPCODE_WORD as usize),
            protocol::opcode::STORE_U32
        );
        assert_eq!(read_word(&ring, 0, TENANT_WORD as usize), 7);
        assert_eq!(
            read_word(&ring, 0, PRIORITY_WORD as usize),
            scheduler::priority::NORMAL
        );
        assert_eq!(read_word(&ring, 0, ARG0_WORD as usize), 10);
        assert_eq!(read_word(&ring, 0, ARG0_WORD as usize + 1), 20);
        assert_eq!(read_word(&ring, 0, ARG0_WORD as usize + 2), 30);
        assert_eq!(read_word(&ring, 1, STATUS_WORD as usize), slot::PUBLISHED);
        assert_eq!(
            read_word(&ring, 1, OPCODE_WORD as usize),
            protocol::opcode::ATOMIC_ADD
        );
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize), 40);
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize + 1), 50);
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize + 2), 60);
        assert_eq!(read_word(&ring, 2, STATUS_WORD as usize), slot::EMPTY);
    }

    #[test]
    fn encode_work_items_ring_words_into_matches_byte_encoder() {
        let items = [
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 10,
                output_handle: 20,
                param: 30,
            },
            ResidentWorkItem {
                op_handle: protocol::opcode::ATOMIC_ADD,
                input_handle: 40,
                output_handle: 50,
                param: 60,
            },
        ];
        let mut bytes = Vec::new();
        let mut words = Vec::new();

        ResidentWorkQueue::encode_work_items_ring_into(4, 7, &items, &mut bytes).unwrap();
        ResidentWorkQueue::encode_work_items_ring_words_into(4, 7, &items, &mut words).unwrap();

        assert_eq!(bytemuck::cast_slice::<u32, u8>(&words), bytes.as_slice());
    }

    #[test]
    fn encode_work_items_ring_words_into_reuses_buffer_by_clearing_status_words() {
        let first = [
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 10,
                output_handle: 20,
                param: 30,
            },
            ResidentWorkItem {
                op_handle: protocol::opcode::ATOMIC_ADD,
                input_handle: 40,
                output_handle: 50,
                param: 60,
            },
        ];
        let second = [ResidentWorkItem {
            op_handle: protocol::opcode::STORE_U32,
            input_handle: 70,
            output_handle: 80,
            param: 90,
        }];
        let mut words = Vec::new();

        ResidentWorkQueue::encode_work_items_ring_words_into(4, 7, &first, &mut words).unwrap();
        ResidentWorkQueue::encode_work_items_ring_words_into(4, 7, &second, &mut words).unwrap();

        assert_eq!(
            read_word_words(&words, 0, STATUS_WORD as usize),
            slot::PUBLISHED
        );
        assert_eq!(read_word_words(&words, 0, ARG0_WORD as usize), 70);
        assert_eq!(read_word_words(&words, 0, ARG0_WORD as usize + 1), 80);
        assert_eq!(read_word_words(&words, 0, ARG0_WORD as usize + 2), 90);
        assert_eq!(
            read_word_words(&words, 1, STATUS_WORD as usize),
            slot::EMPTY
        );
        assert_eq!(
            read_word_words(&words, 2, STATUS_WORD as usize),
            slot::EMPTY
        );
        assert_eq!(
            read_word_words(&words, 3, STATUS_WORD as usize),
            slot::EMPTY
        );
    }

    #[test]
    fn publish_work_items_updates_window_without_resetting_unrelated_slots() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
        write_word(&mut ring, 0, ARG0_WORD as usize, 0xDEAD_BEEF);
        write_word(&mut ring, 3, ARG0_WORD as usize, 0xABCD_EF01);
        let items = [
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 10,
                output_handle: 20,
                param: 30,
            },
            ResidentWorkItem {
                op_handle: protocol::opcode::ATOMIC_ADD,
                input_handle: 40,
                output_handle: 50,
                param: 60,
            },
        ];

        let published = ResidentWorkQueue::publish_work_items(&mut ring, 1, 7, &items).unwrap();

        assert_eq!(published, 2);
        assert_eq!(read_word(&ring, 0, ARG0_WORD as usize), 0xDEAD_BEEF);
        assert_eq!(read_word(&ring, 3, ARG0_WORD as usize), 0xABCD_EF01);
        assert_eq!(read_word(&ring, 1, STATUS_WORD as usize), slot::PUBLISHED);
        assert_eq!(
            read_word(&ring, 1, OPCODE_WORD as usize),
            protocol::opcode::STORE_U32
        );
        assert_eq!(read_word(&ring, 1, TENANT_WORD as usize), 7);
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize), 10);
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize + 1), 20);
        assert_eq!(read_word(&ring, 1, ARG0_WORD as usize + 2), 30);
        assert_eq!(read_word(&ring, 2, STATUS_WORD as usize), slot::PUBLISHED);
        assert_eq!(
            read_word(&ring, 2, OPCODE_WORD as usize),
            protocol::opcode::ATOMIC_ADD
        );
        assert_eq!(read_word(&ring, 2, ARG0_WORD as usize), 40);
        assert_eq!(read_word(&ring, 2, ARG0_WORD as usize + 1), 50);
        assert_eq!(read_word(&ring, 2, ARG0_WORD as usize + 2), 60);
    }

    #[test]
    fn publish_work_items_rejects_inflight_window_without_mutating() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
        write_word(&mut ring, 1, STATUS_WORD as usize, slot::CLAIMED);
        let before = ring.clone();
        let items = [ResidentWorkItem {
            op_handle: protocol::opcode::STORE_U32,
            input_handle: 10,
            output_handle: 20,
            param: 30,
        }];

        let error = ResidentWorkQueue::publish_work_items(&mut ring, 1, 7, &items)
            .expect_err("in-flight target slots must be rejected before mutation");

        let PipelineError::IllegalSlotTransition {
            transition,
            permitted,
            current_status,
            ..
        } = error
        else {
            panic!("an in-flight target slot must be an illegal transition, got {error:?}")
        };
        assert_eq!(transition, RingSlotTransition::Publish.label());
        assert_eq!(permitted, RingSlotTransition::Publish.permitted());
        assert_eq!(current_status, slot::CLAIMED);
        assert_eq!(ring, before);
    }

    #[test]
    fn encode_work_items_ring_into_rejects_oversized_queue_without_mutating() {
        let items = [
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 1,
                output_handle: 2,
                param: 3,
            },
            ResidentWorkItem {
                op_handle: protocol::opcode::STORE_U32,
                input_handle: 4,
                output_handle: 5,
                param: 6,
            },
        ];
        let mut ring = vec![0xAA; 8];

        let result = ResidentWorkQueue::encode_work_items_ring_into(1, 0, &items, &mut ring);

        assert!(result.is_err(), "oversized queue must be rejected");
        assert_eq!(ring, vec![0xAA; 8], "rejection must not mutate ring");
    }

    #[test]
    fn encode_work_items_ring_into_rejects_bad_opcode_without_mutating() {
        let items = [ResidentWorkItem {
            op_handle: protocol::opcode::RESERVED_MAX_RANGE_MIN,
            input_handle: 1,
            output_handle: 2,
            param: 3,
        }];
        let mut ring = vec![0xAA; 8];

        let result = ResidentWorkQueue::encode_work_items_ring_into(1, 0, &items, &mut ring);

        assert!(result.is_err(), "invalid opcode must be rejected");
        assert_eq!(ring, vec![0xAA; 8], "rejection must not mutate ring");
    }
}
