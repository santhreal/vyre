//! Readers for the buffers the megakernel writes, and the shape checks a
//! readback passes before it is read.
//!
//! A reader here decodes a buffer the kernel owns. The two validators state
//! what a readback must look like before any of them touches it.

use crate::resident_work_queue::protocol::{self, DebugRecord};
use crate::resident_work_queue::ResidentWorkQueue;
use crate::{PipelineError, RingEncodingFault};

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

impl ResidentWorkQueue {
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
        protocol::try_count_done_ring_slots(ring_bytes, item_count).map_err(PipelineError::Protocol)
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
        protocol::try_read_debug_log(debug_bytes).map_err(PipelineError::Protocol)
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
        protocol::try_read_debug_log_into(debug_bytes, out).map_err(PipelineError::Protocol)
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
        protocol::try_read_observable(control_bytes, index).map_err(PipelineError::Protocol)
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
        protocol::try_read_metrics(control_bytes).map_err(PipelineError::Protocol)
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
        protocol::try_read_metrics_into(control_bytes, out).map_err(PipelineError::Protocol)
    }
}

fn map_protocol_counter(
    result: Result<u32, protocol::ProtocolError>,
) -> Result<u32, PipelineError> {
    result.map_err(PipelineError::Protocol)
}

pub(in crate::resident_work_queue) fn validate_control_bytes(
    control_bytes: &[u8],
) -> Result<(), PipelineError> {
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

pub(in crate::resident_work_queue) fn validate_debug_log_bytes(
    debug_log_bytes: &[u8],
) -> Result<(), PipelineError> {
    let expected = protocol::debug_log_byte_len(protocol::debug::RECORD_CAPACITY)
        .ok_or(PipelineError::RingEncoding {
            fault: RingEncodingFault::Overflow,
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

// Inline: `protocol_api` is a private module, so its readers and validators
// are unreachable from an integration test.
#[cfg(test)]
mod tests {
    use super::*;

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
        let PipelineError::Backend(message) = err else {
            panic!("an undersized control buffer must reject as a validation fault, got {err:?}")
        };
        let min = protocol::control_byte_len(0)
            .expect("the minimum control length must be representable");
        assert!(
            message.contains(&format!("has 4 bytes, expected at least {min} bytes")),
            "the fault must report the byte length it received and the minimum it required: {message}"
        );
    }

    #[test]
    fn validate_control_bytes_rejects_misaligned() {
        let err = validate_control_bytes(&[0u8; 101])
            .expect_err("must reject non-4-byte-aligned control buffer");
        let PipelineError::Backend(message) = err else {
            panic!("a misaligned control buffer must reject as a validation fault, got {err:?}")
        };
        // Length and alignment share one rejection, so a 101-byte buffer is
        // reported by the same fault as an undersized one.
        assert!(
            message.contains("has 101 bytes") && message.contains("4-byte alignment"),
            "the fault must report the byte length it received and the alignment it required: {message}"
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
        let PipelineError::Backend(message) = err else {
            panic!("an undersized debug log must reject as a validation fault, got {err:?}")
        };
        let expected = protocol::debug_log_byte_len(protocol::debug::RECORD_CAPACITY)
            .expect("the compiled debug-log length must be representable");
        assert!(
            message.contains(&format!("has 4 bytes, expected exactly {expected} bytes")),
            "the fault must report the byte length it received and the exact one it required: {message}"
        );
    }

    #[test]
    fn validate_debug_log_bytes_accepts_valid() {
        let log =
            ResidentWorkQueue::encode_empty_debug_log(protocol::debug::RECORD_CAPACITY).unwrap();
        validate_debug_log_bytes(&log).expect("Fix: valid debug log must pass validation");
    }
}
