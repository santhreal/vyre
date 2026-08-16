//! Typed host readback view for persistent megakernel outputs.

use super::io;
use super::protocol;
use super::protocol_api::{validate_control_bytes, validate_debug_log_bytes};
use crate::PipelineError;

/// Decoded megakernel output buffers in ABI order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentQueueReadback {
    /// Control buffer bytes after dispatch.
    pub control_bytes: Vec<u8>,
    /// Ring buffer bytes after dispatch.
    pub ring_bytes: Vec<u8>,
    /// Debug-log buffer bytes after dispatch.
    pub debug_log_bytes: Vec<u8>,
    /// IO queue bytes after dispatch.
    pub io_queue_bytes: Vec<u8>,
}

/// Host-visible byte volume for one strict megakernel readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidentReadbackCounters {
    /// Bytes copied back for the control buffer.
    pub control_bytes: usize,
    /// Bytes copied back for the ring buffer.
    pub ring_bytes: usize,
    /// Bytes copied back for the debug log.
    pub debug_log_bytes: usize,
    /// Bytes copied back for the IO queue.
    pub io_queue_bytes: usize,
    /// Total host-visible readback bytes.
    pub total_bytes: usize,
}

impl ResidentQueueReadback {
    /// Decode the backend output vector produced by a persistent artifact submission.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Backend`] when output count or protocol buffer
    /// shapes do not match the persistent megakernel ABI.
    pub fn from_outputs(outputs: Vec<Vec<u8>>, slot_count: u32) -> Result<Self, PipelineError> {
        Self::validate_output_refs(&outputs, slot_count)?;
        let [control, ring, debug_log, io_queue] =
            <[Vec<u8>; 4]>::try_from(outputs).map_err(|outputs| {
                PipelineError::Backend(format!(
                    "megakernel readback returned {} buffers after validation, expected 4. Fix: keep output ownership immutable between validation and decode.",
                    outputs.len()
                ))
            })?;
        Ok(Self {
            control_bytes: control,
            ring_bytes: ring,
            debug_log_bytes: debug_log,
            io_queue_bytes: io_queue,
        })
    }

    /// Decode backend outputs into caller-owned readback storage.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Backend`] when output count or protocol buffer
    /// shapes do not match the persistent megakernel ABI.
    pub fn from_outputs_into(
        mut outputs: Vec<Vec<u8>>,
        slot_count: u32,
        out: &mut Self,
    ) -> Result<(), PipelineError> {
        Self::drain_outputs_into(&mut outputs, slot_count, out)
    }

    /// Decode backend outputs into caller-owned readback storage while
    /// preserving the outer output-vector allocation for the next dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Backend`] when output count or protocol buffer
    /// shapes do not match the persistent megakernel ABI.
    pub fn drain_outputs_into(
        outputs: &mut [Vec<u8>],
        slot_count: u32,
        out: &mut Self,
    ) -> Result<(), PipelineError> {
        Self::validate_output_refs(outputs, slot_count)?;
        if outputs.len() != 4 {
            return Err(PipelineError::Backend(format!(
                "megakernel readback returned {} buffers after validation, expected 4. Fix: keep output ownership immutable during drain.",
                outputs.len()
            )));
        }
        std::mem::swap(&mut out.control_bytes, &mut outputs[0]);
        std::mem::swap(&mut out.ring_bytes, &mut outputs[1]);
        std::mem::swap(&mut out.debug_log_bytes, &mut outputs[2]);
        std::mem::swap(&mut out.io_queue_bytes, &mut outputs[3]);
        Ok(())
    }

    /// Number of slots described by this readback ring.
    ///
    /// # Errors
    ///
    /// Returns when the ring length is not a whole number of slot records.
    pub fn slot_count(&self) -> Result<u32, PipelineError> {
        let slot_words = usize::try_from(protocol::SLOT_WORDS).map_err(|_| {
            PipelineError::Backend(
                "megakernel SLOT_WORDS overflowed usize. Fix: reduce SLOT_WORDS.".to_string(),
            )
        })?;
        let slot_bytes = slot_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                PipelineError::Backend(
                    "megakernel slot byte width overflowed usize. Fix: reduce SLOT_WORDS."
                        .to_string(),
                )
            })?;
        if self.ring_bytes.len() % slot_bytes != 0 {
            return Err(PipelineError::Backend(format!(
                "megakernel readback ring has {} bytes, not a multiple of {slot_bytes}. Fix: rebuild the ring with Megakernel::encode_empty_ring.",
                self.ring_bytes.len()
            )));
        }
        u32::try_from(self.ring_bytes.len() / slot_bytes).map_err(|_| {
            PipelineError::Backend(
                "megakernel readback slot count overflowed u32. Fix: split the ring into smaller shards."
                    .to_string(),
            )
        })
    }

    /// Host-visible readback byte counters for B.21 telemetry.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Backend`] when the sum of the four buffer
    /// lengths overflows `usize`. In practice this cannot happen on any real
    /// hardware (four protocol buffers that together exceed `usize::MAX` bytes
    /// are physically impossible), but the error is surfaced loudly so that a
    /// pathological or mocked readback cannot silently report `usize::MAX` as a
    /// byte count and mislead telemetry consumers.
    pub fn counters(&self) -> Result<ResidentReadbackCounters, PipelineError> {
        let control_bytes = self.control_bytes.len();
        let ring_bytes = self.ring_bytes.len();
        let debug_log_bytes = self.debug_log_bytes.len();
        let io_queue_bytes = self.io_queue_bytes.len();
        let total_bytes = control_bytes
            .checked_add(ring_bytes)
            .and_then(|s| s.checked_add(debug_log_bytes))
            .and_then(|s| s.checked_add(io_queue_bytes))
            .ok_or_else(|| {
                PipelineError::Backend(format!(
                    "megakernel readback total bytes overflowed usize \
                     (control={control_bytes} ring={ring_bytes} \
                     debug_log={debug_log_bytes} io_queue={io_queue_bytes}). \
                     Fix: split the readback into smaller shards."
                ))
            })?;
        Ok(ResidentReadbackCounters {
            control_bytes,
            ring_bytes,
            debug_log_bytes,
            io_queue_bytes,
            total_bytes,
        })
    }

    fn validate_output_refs(outputs: &[Vec<u8>], slot_count: u32) -> Result<(), PipelineError> {
        let [control, ring, debug_log, io_queue] = outputs else {
            return Err(PipelineError::Backend(format!(
                "megakernel readback returned {} buffers, expected 4. Fix: keep builder output declarations aligned with control/ring/debug/io ABI order.",
                outputs.len()
            )));
        };
        validate_control_bytes(control)?;
        validate_debug_log_bytes(debug_log)?;
        io::validate_io_queue_bytes(io_queue)?;
        let expected_ring_bytes = protocol::ring_byte_len(slot_count).ok_or_else(|| {
            PipelineError::Backend(
                "megakernel ring byte length overflowed usize during readback validation. Fix: split the ring into smaller shards."
                    .to_string(),
            )
        })?;
        if ring.len() != expected_ring_bytes {
            return Err(PipelineError::Backend(format!(
                "megakernel readback ring has {} bytes, expected {expected_ring_bytes}. Fix: read back the full ring buffer for the compiled slot count.",
                ring.len()
            )));
        }
        Ok(())
    }
}
