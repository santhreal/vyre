//! Readers for the buffers the megakernel writes.

use crate::resident_work_queue::protocol::{self, DebugRecord};
use crate::resident_work_queue::ResidentWorkQueue;
use crate::PipelineError;

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

// Inline: `protocol_api` is a private module, so its readers are
// unreachable from an integration test.
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
}
