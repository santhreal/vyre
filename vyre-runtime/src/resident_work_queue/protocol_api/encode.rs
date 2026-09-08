//! Host-side encoders for the megakernel control, ring and debug-log buffers.
//!
//! Every entry here writes a buffer the host owns and the kernel reads, or
//! reports the byte length such a buffer needs.

use crate::resident_work_queue::protocol;
use crate::resident_work_queue::ResidentWorkQueue;
use crate::PipelineError;

macro_rules! empty_protocol_encoder_into {
    ($name:ident, $capacity:ident, $encoder:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`PipelineError::Protocol`] when the requested capacity
        /// cannot fit in process address space.
        pub fn $name($capacity: u32, dst: &mut Vec<u8>) -> Result<(), PipelineError> {
            $encoder($capacity, dst).map_err(PipelineError::Protocol)
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
    /// Returns [`PipelineError::Protocol`] when the requested observable region
    /// cannot fit in process address space.
    pub fn encode_control(
        shutdown: bool,
        tenant_count: u32,
        observable_slots: u32,
    ) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_control(shutdown, tenant_count, observable_slots)
            .map_err(PipelineError::Protocol)
    }

    /// Fallible control-buffer encoder for callers accepting untrusted sizing.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Protocol`] when the requested observable region
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
    /// Returns [`PipelineError::Protocol`] when the requested observable region
    /// cannot fit in process address space.
    pub fn try_encode_control_into(
        shutdown: bool,
        tenant_count: u32,
        observable_slots: u32,
        dst: &mut Vec<u8>,
    ) -> Result<(), PipelineError> {
        protocol::try_encode_control_into(shutdown, tenant_count, observable_slots, dst)
            .map_err(PipelineError::Protocol)
    }

    /// Encode an empty ring buffer with `slot_count` slots.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Protocol`] when `slot_count * SLOT_WORDS * 4`
    /// overflows.
    pub fn encode_empty_ring(slot_count: u32) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_empty_ring(slot_count).map_err(PipelineError::Protocol)
    }

    /// Fallible ring-buffer encoder for callers accepting untrusted slot counts.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Protocol`] when `slot_count * SLOT_WORDS * 4`
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
    /// Returns [`PipelineError::Protocol`] when the record capacity overflows.
    pub fn encode_empty_debug_log(record_capacity: u32) -> Result<Vec<u8>, PipelineError> {
        protocol::encode_empty_debug_log(record_capacity).map_err(PipelineError::Protocol)
    }

    /// Fallible debug-log encoder for callers accepting untrusted capacities.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Protocol`] when the record capacity overflows.
    pub fn try_encode_empty_debug_log(record_capacity: u32) -> Result<Vec<u8>, PipelineError> {
        Self::encode_empty_debug_log(record_capacity)
    }

    empty_protocol_encoder_into!(
        try_encode_empty_debug_log_into,
        record_capacity,
        protocol::try_encode_empty_debug_log_into,
        "Fallible debug-log encoder into caller-owned storage."
    );
}

// Inline: `protocol_api` is a private module, so its encoders are unreachable
// from an integration test.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident_work_queue::protocol::SLOT_WORDS;

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
    fn encode_empty_debug_log_round_trips() {
        let log = ResidentWorkQueue::encode_empty_debug_log(32).unwrap();
        let records = ResidentWorkQueue::read_debug_log(&log);
        assert!(
            records.is_empty(),
            "fresh debug log must contain zero records"
        );
    }
}
