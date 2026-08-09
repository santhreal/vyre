// Core megakernel construction and host protocol contracts.

use vyre_foundation::ir::{Node, Program};
use vyre_runtime::resident_work_queue::protocol::{control, debug, opcode as opcodes, slot};
use vyre_runtime::resident_work_queue::scheduler;
use vyre_runtime::resident_work_queue::*;
use vyre_runtime::PipelineError;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedSlotHostView {
    opcode_count: u8,
    entries: Vec<(u8, u8)>,
    packed_args: Vec<u32>,
}

fn decode_packed_slot_words(words: &[u32]) -> PackedSlotHostView {
    let mut bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let opcode_count = bytes.first().copied().unwrap_or(0);
    let total_arg_words = usize::from(bytes.get(1).copied().unwrap_or(0));
    let metadata_bytes = 2usize + usize::from(opcode_count).saturating_mul(2);
    let metadata_words = metadata_bytes.div_ceil(4);
    bytes.resize(metadata_words * 4, 0);
    let mut entries = Vec::with_capacity(opcode_count as usize);
    for index in 0..usize::from(opcode_count) {
        let byte_index = 2 + index * 2;
        entries.push((bytes[byte_index], bytes[byte_index + 1]));
    }
    // Slice off the actual packed arg words (byte 1 of the metadata
    // records the count); trailing slot memory is stale / padding.
    let end = metadata_words
        .saturating_add(total_arg_words)
        .min(words.len());
    PackedSlotHostView {
        opcode_count,
        entries,
        packed_args: words[metadata_words..end].to_vec(),
    }
}

#[test]
fn program_has_four_buffers_and_a_forever_body() {
    let program = build_program();
    assert_eq!(program.buffers().len(), 4);
    assert_eq!(program.buffers()[0].name(), "control");
    assert_eq!(program.buffers()[1].name(), "ring_buffer");
    assert_eq!(program.buffers()[2].name(), "debug_log");
    assert_eq!(program.buffers()[3].name(), "io_queue");
    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert_eq!(program.entry().len(), 1);
    match &program.entry()[0] {
        Node::Region { body, .. } => assert!(
            body.iter().any(|node| matches!(node, Node::Loop { .. })),
            "Fix: megakernel program must retain a persistent loop even when setup nodes are hoisted before it"
        ),
        other => {
            panic!("Fix: megakernel program must be wrapped in one runnable region, got {other:?}")
        }
    }
}

#[test]
fn program_passes_validation() {
    let program = build_program();
    let errors = vyre_foundation::validate::validate(&program);
    assert!(errors.is_empty(), "validation failed: {errors:?}");
}

#[test]
fn program_round_trips_through_wire_format() {
    let program = build_program();
    let bytes = program.to_wire().expect("serialize");
    let decoded = Program::from_wire(&bytes).expect("deserialize");
    assert_eq!(decoded.workgroup_size(), program.workgroup_size());
    assert_eq!(decoded.buffers().len(), program.buffers().len());
}

#[test]
fn sharded_program_uses_requested_workgroup_size() {
    let prog = build_program_sharded(128, &[]);
    assert_eq!(prog.workgroup_size(), [128, 1, 1]);
    let errs = vyre_foundation::validate::validate(&prog);
    assert!(errs.is_empty(), "sharded validation failed: {errs:?}");
}

#[test]
fn sharded_slots_size_ring_buffer_from_slot_count() {
    let prog = build_program_sharded_slots(256, 4096, &[]);
    assert_eq!(prog.workgroup_size(), [256, 1, 1]);
    let ring = prog
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "ring_buffer")
        .expect("ring buffer declared");
    assert_eq!(ring.count(), 4096 * SLOT_WORDS);
}

#[test]
fn custom_opcode_handler_wires_into_program() {
    use vyre_foundation::ir::Expr;
    let handler = OpcodeHandler {
        opcode: 0x4000_0000,
        body: vec![Node::let_bind(
            "custom_prev",
            Expr::atomic_exchange(
                "control",
                Expr::var("arg1"),
                Expr::mul(Expr::var("arg0"), Expr::u32(2)),
            ),
        )],
    };
    let prog = build_program_sharded(64, std::slice::from_ref(&handler));
    let errs = vyre_foundation::validate::validate(&prog);
    assert!(errs.is_empty(), "custom opcode validation failed: {errs:?}");
}

#[test]
fn jit_payload_has_slot_opcode_and_arg_bindings() {
    use vyre_foundation::ir::Expr;

    let payload = vec![Node::let_bind(
        "jit_arg_sum",
        Expr::add(Expr::var("arg0"), Expr::var("arg2")),
    )];
    let prog = build_program_jit(64, &payload);
    let errs = vyre_foundation::validate::validate(&prog);

    assert!(
        errs.is_empty(),
        "JIT payloads must see the same slot argument bindings as interpreted opcode handlers: {errs:?}"
    );
}

#[test]
fn encode_control_sets_shutdown_and_tenant_base() {
    let ctrl = ResidentWorkQueue::encode_control(true, 4, 8).unwrap();
    let shutdown = u32::from_le_bytes(ctrl[0..4].try_into().unwrap());
    let done_count = u32::from_le_bytes(ctrl[4..8].try_into().unwrap());
    let tenant_base = u32::from_le_bytes(ctrl[8..12].try_into().unwrap());
    assert_eq!(shutdown, 1);
    assert_eq!(done_count, 0);
    assert_eq!(tenant_base, control::TENANT_BASE + 1);
    let tt_off = (control::TENANT_BASE as usize + 1) * 4;
    for i in 0..4 {
        let w = u32::from_le_bytes(ctrl[tt_off + i * 4..tt_off + i * 4 + 4].try_into().unwrap());
        assert_eq!(w, !0u32, "tenant {i} must default to all-lanes-allowed");
    }
}

#[test]
fn encode_control_covers_epoch_and_priority_offsets() {
    let ctrl = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
    let min_len = protocol::control_byte_len(0).expect("control length must fit");
    assert_eq!(ctrl.len(), min_len);
    assert!(ctrl.len() >= (control::EPOCH as usize + 1) * 4);
    assert!(ctrl.len() >= (scheduler::PRIORITY_OFFSETS_BASE as usize + 6) * 4);
}

#[test]
fn publish_slot_writes_status_last_and_respects_backpressure() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 1, 0, opcodes::STORE_U32, &[42, 7]).unwrap();
    let base = (SLOT_WORDS as usize) * 4;
    let status = u32::from_le_bytes(ring[base..base + 4].try_into().unwrap());
    let op = u32::from_le_bytes(ring[base + 4..base + 8].try_into().unwrap());
    assert_eq!(status, slot::PUBLISHED);
    assert_eq!(op, opcodes::STORE_U32);
    let err = ResidentWorkQueue::publish_slot(&mut ring, 1, 0, opcodes::NOP, &[]).unwrap_err();
    assert!(matches!(err, PipelineError::QueueFull { .. }));
    ring[base..base + 4].copy_from_slice(&slot::DONE.to_le_bytes());
    ResidentWorkQueue::publish_slot(&mut ring, 1, 0, opcodes::NOP, &[]).unwrap();
}

#[test]
fn publish_slot_rejects_malformed_ring_lengths() {
    let mut ring = vec![0u8; (SLOT_WORDS as usize * 4) + 1];
    let err = ResidentWorkQueue::publish_slot(&mut ring, 0, 0, opcodes::NOP, &[]).unwrap_err();
    assert!(matches!(err, PipelineError::QueueFull { .. }));
}

#[test]
fn republishing_done_slot_clears_stale_args() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, opcodes::STORE_U32, &[1, 2, 3]).unwrap();
    ring[..4].copy_from_slice(&slot::DONE.to_le_bytes());
    ResidentWorkQueue::publish_slot(&mut ring, 0, 0, opcodes::NOP, &[9]).unwrap();
    let words = ring
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(words[ARG0_WORD as usize], 9);
    assert_eq!(words[ARG0_WORD as usize + 1], 0);
    assert_eq!(words[ARG0_WORD as usize + 2], 0);
}

#[test]
fn fallible_ring_encoder_rejects_u32_word_overflow() {
    let too_many_slots = (u32::MAX / SLOT_WORDS) + 1;
    let err = ResidentWorkQueue::try_encode_empty_ring(too_many_slots).unwrap_err();
    assert!(matches!(err, PipelineError::QueueFull { .. }));
}

#[test]
fn publish_slot_out_of_bounds_returns_queue_full() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
    let err = ResidentWorkQueue::publish_slot(&mut ring, 5, 0, opcodes::NOP, &[]).unwrap_err();
    assert!(matches!(err, PipelineError::QueueFull { .. }));
}

#[test]
fn publish_slot_too_many_args_returns_queue_full() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(1).unwrap();
    let too_many = vec![0u32; (ARGS_PER_SLOT + 1) as usize];
    let err =
        ResidentWorkQueue::publish_slot(&mut ring, 0, 0, opcodes::NOP, &too_many).unwrap_err();
    assert!(matches!(err, PipelineError::QueueFull { .. }));
}

#[test]
fn read_done_count_decodes_little_endian() {
    let mut ctrl = ResidentWorkQueue::encode_control(false, 0, 0).unwrap();
    let off = (control::DONE_COUNT as usize) * 4;
    ctrl[off..off + 4].copy_from_slice(&42u32.to_le_bytes());
    assert_eq!(ResidentWorkQueue::read_done_count(&ctrl), 42);
}

#[test]
fn read_debug_log_decodes_printf_records() {
    let mut log = ResidentWorkQueue::encode_empty_debug_log(3).unwrap();
    let cursor = 8u32;
    log[0..4].copy_from_slice(&cursor.to_le_bytes());
    let rec_off = (debug::RECORDS_BASE as usize) * 4;
    for (i, w) in [11u32, 12, 13, 14, 21, 22, 23, 24].iter().enumerate() {
        log[rec_off + i * 4..rec_off + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let records = ResidentWorkQueue::read_debug_log(&log);
    assert_eq!(records.len(), 2);
    assert_eq!(
        records[0],
        DebugRecord {
            fmt_id: 11,
            args: [12, 13, 14]
        }
    );
    assert_eq!(
        records[1],
        DebugRecord {
            fmt_id: 21,
            args: [22, 23, 24]
        }
    );
}

#[test]
fn opcode_shutdown_uses_u32_max_to_avoid_zero_collision() {
    assert_ne!(opcodes::SHUTDOWN, 0);
    assert_eq!(opcodes::SHUTDOWN, u32::MAX);
}

#[test]
fn slot_states_are_distinct() {
    assert_ne!(slot::EMPTY, slot::PUBLISHED);
    assert_ne!(slot::PUBLISHED, slot::CLAIMED);
    assert_ne!(slot::CLAIMED, slot::DONE);
    assert_eq!(slot::EMPTY, 0);
}

// --- V6.4 new tests ---

#[test]
fn new_opcodes_are_distinct() {
    let codes = [
        opcodes::NOP,
        opcodes::STORE_U32,
        opcodes::ATOMIC_ADD,
        opcodes::LOAD_U32,
        opcodes::COMPARE_SWAP,
        opcodes::MEMCPY,
        opcodes::DFA_STEP,
        opcodes::BATCH_FENCE,
        opcodes::PACKED_SLOT,
        opcodes::PRINTF,
        opcodes::SHUTDOWN,
    ];
    for (i, a) in codes.iter().enumerate() {
        for (j, b) in codes.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "opcode collision: index {i} == index {j}");
            }
        }
    }
}

#[test]
fn batch_publish_fills_slots_and_adds_fence() {
    let mut ring = ResidentWorkQueue::encode_empty_ring(8).unwrap();
    let items = vec![
        (opcodes::STORE_U32, vec![10, 32]),
        (opcodes::STORE_U32, vec![20, 33]),
        (opcodes::ATOMIC_ADD, vec![1, 34]),
    ];
    let consumed = ResidentWorkQueue::batch_publish(&mut ring, 0, 0, &items, 0xBEEF).unwrap();
    // 3 work items + 1 fence = 4 slots
    assert_eq!(consumed, 4);

    // Verify the fence slot
    let fence_base = 3 * (SLOT_WORDS as usize) * 4;
    let fence_op = u32::from_le_bytes(ring[fence_base + 4..fence_base + 8].try_into().unwrap());
    assert_eq!(fence_op, opcodes::BATCH_FENCE);
}

#[test]
fn read_epoch_decodes_control() {
    let mut ctrl = vec![0u8; (control::EPOCH as usize + 2) * 4];
    let off = (control::EPOCH as usize) * 4;
    ctrl[off..off + 4].copy_from_slice(&7u32.to_le_bytes());
    assert_eq!(ResidentWorkQueue::read_epoch(&ctrl), 7);
}

#[test]
fn read_observable_returns_zero_for_unset() {
    let ctrl = vec![0u8; (control::OBSERVABLE_BASE as usize + 10) * 4];
    assert_eq!(ResidentWorkQueue::read_observable(&ctrl, 5), 0);
}
