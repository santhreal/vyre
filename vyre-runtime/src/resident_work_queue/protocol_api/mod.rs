//! Host protocol API wrappers for megakernel control/ring buffers.

mod publish;
pub use publish::RingSlotTransition;

use crate::PipelineError;

use super::protocol::{self, DebugRecord};
use super::ResidentWorkQueue;

macro_rules! protocol_counter_readers {
    () => {
        /// Strictly decode the kernel's `done_count` from a control buffer.
        ///
        /// # Errors
        ///
        /// Returns [`PipelineError`] when the control buffer is malformed or too
        /// short to contain the done counter.
        pub fn try_read_done_count(control_bytes: &[u8]) -> Result<u32, PipelineError> {
            map_protocol_counter(protocol::try_read_done_count(control_bytes))
        }

        /// Strictly read the epoch counter from a control buffer.
        ///
        /// # Errors
        ///
        /// Returns [`PipelineError`] when the control buffer is malformed or too
        /// short to contain the epoch counter.
        pub fn try_read_epoch(control_bytes: &[u8]) -> Result<u32, PipelineError> {
            map_protocol_counter(protocol::try_read_epoch(control_bytes))
        }
    };
}

macro_rules! empty_protocol_encoder_into {
    ($name:ident, $capacity:ident, $encoder:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`PipelineError::QueueFull`] when the requested capacity
        /// cannot fit in process address space.
        pub fn $name($capacity: u32, dst: &mut Vec<u8>) -> Result<(), PipelineError> {
            $encoder($capacity, dst).map_err(protocol_error)
        }
    };
}

impl ResidentWorkQueue {
    /// Byte length of a control buffer for `observable_slots`.
    #[must_use]
    pub fn control_byte_len(observable_slots: u32) -> Option<usize> {
        protocol::control_byte_len(observable_slots)
    }

    /// Byte length of a ring buffer for `slot_count`.
    #[must_use]
    pub fn ring_byte_len(slot_count: u32) -> Option<usize> {
        protocol::ring_byte_len(slot_count)
    }

    /// Byte length of a debug-log buffer for `record_capacity`.
    #[must_use]
    pub fn debug_log_byte_len(record_capacity: u32) -> Option<usize> {
        protocol::debug_log_byte_len(record_capacity)
    }

    /// Default debug-log record capacity owned by the runtime protocol.
    #[must_use]
    pub fn debug_record_capacity() -> u32 {
        protocol::debug::RECORD_CAPACITY
    }

    /// Encode a control-buffer payload.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the requested observable region
    /// cannot fit in process address space.
    pub fn encode_control(
        shutdown: bool,
        tenant_count: u32,
        observable_slots: u32,
    ) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_control(shutdown, tenant_count, observable_slots).map_err(protocol_error)
    }

    /// Fallible control-buffer encoder for callers accepting untrusted sizing.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the requested observable region
    /// cannot fit in process address space.
    pub fn try_encode_control(
        shutdown: bool,
        tenant_count: u32,
        observable_slots: u32,
    ) -> Result<Vec<u8>, PipelineError> {
        Self::encode_control(shutdown, tenant_count, observable_slots)
    }

    /// Fallible control-buffer encoder into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the requested observable region
    /// cannot fit in process address space.
    pub fn try_encode_control_into(
        shutdown: bool,
        tenant_count: u32,
        observable_slots: u32,
        dst: &mut Vec<u8>,
    ) -> Result<(), PipelineError> {
        protocol::try_encode_control_into(shutdown, tenant_count, observable_slots, dst)
            .map_err(protocol_error)
    }

    /// Encode an empty ring buffer with `slot_count` slots.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when `slot_count * SLOT_WORDS * 4`
    /// overflows.
    pub fn encode_empty_ring(slot_count: u32) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_empty_ring(slot_count).map_err(protocol_error)
    }

    /// Fallible ring-buffer encoder for callers accepting untrusted slot counts.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when `slot_count * SLOT_WORDS * 4`
    /// overflows.
    pub fn try_encode_empty_ring(slot_count: u32) -> Result<Vec<u8>, PipelineError> {
        Self::encode_empty_ring(slot_count)
    }

    empty_protocol_encoder_into!(
        try_encode_empty_ring_into,
        slot_count,
        protocol::try_encode_empty_ring_into,
        "Fallible ring-buffer encoder into caller-owned storage."
    );

    /// Encode an empty PRINTF channel buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the record capacity overflows.
    pub fn encode_empty_debug_log(record_capacity: u32) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_empty_debug_log(record_capacity).map_err(protocol_error)
    }

    /// Fallible debug-log encoder for callers accepting untrusted capacities.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the record capacity overflows.
    pub fn try_encode_empty_debug_log(record_capacity: u32) -> Result<Vec<u8>, PipelineError> {
        Self::encode_empty_debug_log(record_capacity)
    }

    empty_protocol_encoder_into!(
        try_encode_empty_debug_log_into,
        record_capacity,
        protocol::try_encode_empty_debug_log_into,
        "Fallible debug-log encoder into caller-owned storage."
    );

    /// Decode the kernel's `done_count` from a control buffer.
    #[must_use]
    pub fn read_done_count(control_bytes: &[u8]) -> u32 {
        protocol::read_done_count(control_bytes)
    }

    protocol_counter_readers!();

    /// Strictly count DONE slots in a ring-buffer readback.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the ring readback is malformed or too
    /// short for `item_count` complete protocol slots.
    pub fn try_count_done_ring_slots(
        ring_bytes: &[u8],
        item_count: usize,
    ) -> Result<u64, PipelineError> {
        protocol::try_count_done_ring_slots(ring_bytes, item_count).map_err(protocol_error)
    }

    /// Decode PRINTF records out of the debug-log buffer.
    #[must_use]
    pub fn read_debug_log(debug_bytes: &[u8]) -> Vec<DebugRecord> {
        protocol::read_debug_log(debug_bytes)
    }

    /// Decode PRINTF records into caller-owned storage.
    pub fn read_debug_log_into(debug_bytes: &[u8], out: &mut Vec<DebugRecord>) {
        protocol::read_debug_log_into(debug_bytes, out);
    }

    /// Strictly decode PRINTF records out of the debug-log buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the debug-log buffer is malformed or the
    /// cursor points at a partial record.
    pub fn try_read_debug_log(debug_bytes: &[u8]) -> Result<Vec<DebugRecord>, PipelineError> {
        protocol::try_read_debug_log(debug_bytes).map_err(protocol_error)
    }

    /// Strictly decode PRINTF records into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the debug-log buffer is malformed or the
    /// cursor points at a partial record.
    pub fn try_read_debug_log_into(
        debug_bytes: &[u8],
        out: &mut Vec<DebugRecord>,
    ) -> Result<(), PipelineError> {
        protocol::try_read_debug_log_into(debug_bytes, out).map_err(protocol_error)
    }

    /// Read the epoch counter from a control buffer. The epoch
    /// increments on each `BATCH_FENCE` execution  -  the host polls
    /// this to detect batch completion without scanning the ring.
    #[must_use]
    pub fn read_epoch(control_bytes: &[u8]) -> u32 {
        protocol::read_epoch(control_bytes)
    }

    /// Read an observable result word from a control buffer.
    /// Opcodes like `LOAD_U32`, `COMPARE_SWAP`, and `BATCH_FENCE`
    /// write results here.
    #[must_use]
    pub fn read_observable(control_bytes: &[u8], index: u32) -> u32 {
        protocol::read_observable(control_bytes, index)
    }

    /// Strictly read an observable result word from a control buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the buffer is malformed or the
    /// observable index is outside the supplied readback.
    pub fn try_read_observable(control_bytes: &[u8], index: u32) -> Result<u32, PipelineError> {
        protocol::try_read_observable(control_bytes, index).map_err(protocol_error)
    }

    /// Read per-opcode metrics counters from a control buffer.
    /// Returns a map of `opcode_id → execution_count` for any
    /// non-zero counters.
    #[must_use]
    pub fn read_metrics(control_bytes: &[u8]) -> Vec<(u32, u32)> {
        protocol::read_metrics(control_bytes)
    }

    /// Read per-opcode metrics counters into caller-owned storage.
    pub fn read_metrics_into(control_bytes: &[u8], out: &mut Vec<(u32, u32)>) {
        protocol::read_metrics_into(control_bytes, out);
    }

    /// Strictly read per-opcode metrics counters from a control buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the buffer is malformed or too short for
    /// the fixed metrics window.
    pub fn try_read_metrics(control_bytes: &[u8]) -> Result<Vec<(u32, u32)>, PipelineError> {
        protocol::try_read_metrics(control_bytes).map_err(protocol_error)
    }

    /// Strictly read per-opcode metrics counters into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the buffer is malformed or too short for
    /// the fixed metrics window.
    pub fn try_read_metrics_into(
        control_bytes: &[u8],
        out: &mut Vec<(u32, u32)>,
    ) -> Result<(), PipelineError> {
        protocol::try_read_metrics_into(control_bytes, out).map_err(protocol_error)
    }
}

fn map_protocol_counter(
    result: Result<u32, super::protocol::ProtocolError>,
) -> Result<u32, PipelineError> {
    result.map_err(protocol_error)
}

fn protocol_error(error: protocol::ProtocolError) -> PipelineError {
    match error {
        protocol::ProtocolError::ByteLengthOverflow { fix, .. } => PipelineError::QueueFull {
            queue: "submission",
            fix,
        },
        other => PipelineError::Backend(other.to_string()),
    }
}

pub(super) fn validate_control_bytes(control_bytes: &[u8]) -> Result<(), PipelineError> {
    let min = protocol::control_byte_len(0).ok_or_else(|| {
        PipelineError::Backend(
            "megakernel minimum control-buffer length overflowed usize. Fix: keep CONTROL_MIN_WORDS within host address limits."
                .to_string(),
        )
    })?;
    if control_bytes.len() < min || control_bytes.len() % 4 != 0 {
        return Err(PipelineError::Backend(format!(
            "megakernel control buffer has {} bytes, expected at least {min} bytes and 4-byte alignment. Fix: build it with Megakernel::encode_control.",
            control_bytes.len()
        )));
    }
    Ok(())
}

pub(super) fn validate_debug_log_bytes(debug_log_bytes: &[u8]) -> Result<(), PipelineError> {
    let expected = protocol::debug_log_byte_len(protocol::debug::RECORD_CAPACITY)
        .ok_or(PipelineError::QueueFull {
            queue: "submission",
            fix: "debug-log minimum length overflowed usize; keep debug ABI constants within host limits",
        })?;
    if debug_log_bytes.len() != expected {
        return Err(PipelineError::Backend(format!(
            "megakernel debug-log buffer has {} bytes, expected exactly {expected} bytes for {} PRINTF records. Fix: build it with Megakernel::encode_empty_debug_log(protocol::debug::RECORD_CAPACITY).",
            debug_log_bytes.len(),
            protocol::debug::RECORD_CAPACITY
        )));
    }
    Ok(())
}

// Inline: `protocol_api` is a private module reachable only through the
// `RingSlotTransition` re-export, so its encode and publish helpers are
// unreachable from an integration test.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_work_queue::planner::ResidentWorkItem;
    use crate::resident_work_queue::protocol::{
        slot, ARG0_WORD, ARGS_PER_SLOT, OPCODE_WORD, PRIORITY_WORD, SLOT_WORDS, STATUS_WORD,
        TENANT_WORD,
    };
    use crate::resident_work_queue::scheduler;

    #[test]
    fn encode_control_produces_aligned_buffer() {
        let buf = ResidentWorkQueue::encode_control(false, 1, 4).unwrap();
        assert!(
            buf.len() % 4 == 0,
            "control buffer must be u32-word aligned"
        );
        assert!(
            !buf.is_empty(),
            "control buffer must have at least the fixed header"
        );
    }

    #[test]
    fn encode_control_with_shutdown_sets_flag() {
        let buf = ResidentWorkQueue::encode_control(true, 1, 0).unwrap();
        // The shutdown word should be non-zero.
        let shutdown_word = u32::from_le_bytes([
            buf[protocol::control::SHUTDOWN as usize * 4],
            buf[protocol::control::SHUTDOWN as usize * 4 + 1],
            buf[protocol::control::SHUTDOWN as usize * 4 + 2],
            buf[protocol::control::SHUTDOWN as usize * 4 + 3],
        ]);
        assert_ne!(shutdown_word, 0, "shutdown flag must be set");
    }

    #[test]
    fn try_encode_control_delegates_to_encode_control() {
        let a = ResidentWorkQueue::encode_control(false, 2, 8).unwrap();
        let b = ResidentWorkQueue::try_encode_control(false, 2, 8).unwrap();
        assert_eq!(a, b, "try_encode_control must produce identical output");
    }

    #[test]
    fn encode_into_reuses_and_zeroes_protocol_buffers() {
        let mut control = vec![0xAA; 4096];
        let control_capacity = control.capacity();
        ResidentWorkQueue::try_encode_control_into(false, 2, 8, &mut control).unwrap();
        assert_eq!(control.capacity(), control_capacity);
        assert_eq!(
            control,
            ResidentWorkQueue::try_encode_control(false, 2, 8).unwrap()
        );

        let mut ring = vec![0xAA; 4096];
        let ring_capacity = ring.capacity();
        ResidentWorkQueue::try_encode_empty_ring_into(4, &mut ring).unwrap();
        assert_eq!(ring.capacity(), ring_capacity);
        assert_eq!(ring, ResidentWorkQueue::try_encode_empty_ring(4).unwrap());

        let mut debug_log = vec![0xAA; 4096];
        let debug_capacity = debug_log.capacity();
        ResidentWorkQueue::try_encode_empty_debug_log_into(4, &mut debug_log).unwrap();
        assert_eq!(debug_log.capacity(), debug_capacity);
        assert_eq!(
            debug_log,
            ResidentWorkQueue::try_encode_empty_debug_log(4).unwrap()
        );
    }

    #[test]
    fn encode_empty_ring_respects_slot_count() {
        let buf = ResidentWorkQueue::encode_empty_ring(16).unwrap();
        let expected_bytes = 16 * SLOT_WORDS as usize * 4;
        assert_eq!(
            buf.len(),
            expected_bytes,
            "ring must be slot_count * SLOT_WORDS * 4 bytes"
        );
    }

    #[test]
    fn encode_empty_ring_zero_slots() {
        let buf = ResidentWorkQueue::encode_empty_ring(0).unwrap();
        assert!(buf.is_empty(), "0 slots must produce empty buffer");
    }

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
        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1])
            .unwrap();
        // Try to publish again  -  slot is PUBLISHED (not EMPTY/DONE).
        let err =
            ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[2])
                .expect_err("must reject publishing to an in-flight slot");
        assert!(
            err.to_string().contains("not publishable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn publish_slot_rejects_out_of_bounds() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
        let err =
            ResidentWorkQueue::publish_slot(&mut ring, 99, 1, protocol::opcode::STORE_U32, &[1])
                .expect_err("must reject slot_idx beyond ring capacity");
        assert!(
            err.to_string().contains("slot_idx exceeds ring slot count"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn publish_slot_rejects_too_many_args() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();
        let too_many = vec![0u32; ARGS_PER_SLOT as usize + 1];
        let err = ResidentWorkQueue::publish_slot(
            &mut ring,
            0,
            1,
            protocol::opcode::STORE_U32,
            &too_many,
        )
        .expect_err("must reject args exceeding ARGS_PER_SLOT");
        assert!(
            err.to_string().contains("too many args for one slot"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn publish_slot_allows_republish_after_done() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
        // Publish, then manually mark as DONE.
        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1])
            .unwrap();
        write_word(&mut ring, 0, STATUS_WORD as usize, slot::DONE);
        // Should succeed  -  DONE slots are recyclable.
        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::ATOMIC_ADD, &[2])
            .unwrap();
        let op = read_word(&ring, 0, OPCODE_WORD as usize);
        assert_eq!(op, protocol::opcode::ATOMIC_ADD);
    }

    #[test]
    fn ring_slot_transition_state_machine_accepts_legal_lifecycle() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(4).unwrap();
        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1])
            .unwrap();

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

        ResidentWorkQueue::publish_slot(&mut ring, 1, 1, protocol::opcode::STORE_U32, &[2])
            .unwrap();
        ResidentWorkQueue::transition_slot_status(&mut ring, 1, RingSlotTransition::Cancel)
            .expect("Fix: unclaimed published slots must be cancellable");
        assert_eq!(read_word(&ring, 1, STATUS_WORD as usize), slot::EMPTY);

        ResidentWorkQueue::publish_slot(&mut ring, 2, 1, protocol::opcode::STORE_U32, &[3])
            .unwrap();
        ResidentWorkQueue::transition_slot_status(&mut ring, 2, RingSlotTransition::Fault)
            .expect("Fix: in-flight published slots must transition to FAULT");
        assert_eq!(read_word(&ring, 2, STATUS_WORD as usize), slot::FAULT);
    }

    #[test]
    fn ring_slot_transition_state_machine_rejects_illegal_edges_without_mutation() {
        let mut ring = ResidentWorkQueue::encode_empty_ring(2).unwrap();

        let err = ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Done)
            .expect_err("EMPTY slots cannot complete");
        assert!(
            err.to_string().contains("done requires CLAIMED"),
            "Fix: illegal transition error must name the required source state, got: {err}"
        );
        assert_eq!(read_word(&ring, 0, STATUS_WORD as usize), slot::EMPTY);

        ResidentWorkQueue::publish_slot(&mut ring, 0, 1, protocol::opcode::STORE_U32, &[1])
            .unwrap();
        ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Claim).unwrap();
        let before = ring.clone();
        let err =
            ResidentWorkQueue::transition_slot_status(&mut ring, 0, RingSlotTransition::Cancel)
                .expect_err("CLAIMED slots are worker-owned and cannot be cancelled by host");
        assert!(
            err.to_string().contains("cancel requires an unclaimed"),
            "Fix: illegal cancel error must name ownership boundary, got: {err}"
        );
        assert_eq!(ring, before);

        let err =
            ResidentWorkQueue::transition_slot_status(&mut ring, 1, RingSlotTransition::Publish)
                .expect_err("status-only publish is forbidden");
        assert!(
            err.to_string().contains("publish_slot"),
            "Fix: status-only publish rejection must direct callers to payload-safe APIs, got: {err}"
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
    fn read_done_count_starts_at_zero() {
        let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
        assert_eq!(ResidentWorkQueue::read_done_count(&control), 0);
    }

    #[test]
    fn read_epoch_starts_at_zero() {
        let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
        assert_eq!(ResidentWorkQueue::read_epoch(&control), 0);
    }

    #[test]
    fn encode_empty_debug_log_round_trips() {
        let log = ResidentWorkQueue::encode_empty_debug_log(32).unwrap();
        let records = ResidentWorkQueue::read_debug_log(&log);
        assert!(
            records.is_empty(),
            "fresh debug log must contain zero records"
        );
    }

    #[test]
    fn read_metrics_on_fresh_control_returns_empty() {
        let control = ResidentWorkQueue::encode_control(false, 1, 4).unwrap();
        let metrics = ResidentWorkQueue::read_metrics(&control);
        assert!(
            metrics.is_empty(),
            "fresh control buffer must have no non-zero metric counters"
        );
    }

    #[test]
    fn validate_control_bytes_rejects_too_short() {
        let err =
            validate_control_bytes(&[0u8; 4]).expect_err("must reject undersized control buffer");
        assert!(
            err.to_string().contains("expected at least"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_control_bytes_rejects_misaligned() {
        let err = validate_control_bytes(&[0u8; 101])
            .expect_err("must reject non-4-byte-aligned control buffer");
        assert!(
            err.to_string().contains("4-byte alignment"),
            "unexpected error: {err}"
        );
    }

    #[test]

    fn validate_control_bytes_accepts_valid() {
        let control = ResidentWorkQueue::encode_control(false, 1, 0).unwrap();
        validate_control_bytes(&control).expect("Fix: valid control buffer must pass validation");
    }

    #[test]
    fn validate_debug_log_bytes_rejects_wrong_size() {
        let err =
            validate_debug_log_bytes(&[0u8; 4]).expect_err("must reject undersized debug log");
        assert!(
            err.to_string().contains("expected exactly"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_debug_log_bytes_accepts_valid() {
        let log =
            ResidentWorkQueue::encode_empty_debug_log(protocol::debug::RECORD_CAPACITY).unwrap();
        validate_debug_log_bytes(&log).expect("Fix: valid debug log must pass validation");
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
            ResidentWorkQueue::encode_work_items_ring_words_into(4, 7, &second, &mut words)
                .unwrap();

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

            assert!(error.to_string().contains("not publishable"));
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
}
