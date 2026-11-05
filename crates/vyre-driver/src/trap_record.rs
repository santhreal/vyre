//! Device-side trap record layout, shared by every backend that reports one.
//!
//! A trap is a device-side refusal: the kernel reached a guard the program says
//! is unreachable, so the launch produced no valid result. The device cannot
//! return an error, so it writes one record into a small sidecar buffer and the
//! host reads it after the launch. A backend that skips that read reports the
//! launch as successful and the caller reads uninitialized output as an answer,
//! which is the failure mode this module exists to make impossible to reproduce
//! twice.
//!
//! The layout is fixed here rather than per backend so a record written by one
//! target decodes with one reader, and so a new target inherits the layout
//! instead of inventing a fourth spelling of it.

use crate::BackendError;

/// Bytes one trap record occupies: four little-endian u32 words.
///
/// Word 0 is the claim flag, set from 0 to 1 by one atomic compare-and-swap so
/// exactly one trapping lane writes the remaining words. Word 1 is the address
/// operand the trapping op carried (an element index or byte offset, not a device
/// pointer). Word 2 is the trap tag code from
/// `vyre_lower::descriptor_trap_tags`, 1-based so 0 stays available for "no
/// trap". Word 3 is the axis-0 global invocation id of the lane that claimed the
/// record.
pub const TRAP_RECORD_BYTES: usize = 16;

/// One device-side trap, decoded from a sidecar readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrapRecord {
    /// Address operand the trapping op carried.
    pub address: u32,
    /// 1-based trap tag code, decoded against the program's tag table.
    pub tag_code: u32,
    /// Axis-0 global invocation id of the lane that claimed the record. Read this
    /// against the launch's element count: a lane at or past the count means the
    /// guard ran outside the declared range.
    pub lane: u32,
}

/// Tag text to report when a code has no entry in the program's tag table.
///
/// A code with no tag is a defect (the emitter and the table came from different
/// descriptors), but the trap itself is still real, so the refusal reports the
/// code rather than swallowing the trap over a missing string.
pub const UNKNOWN_TRAP_TAG: &str = "unknown trap tag code";

/// Decode a trap sidecar readback.
///
/// `Ok(None)` means no lane trapped. `Ok(Some(record))` means one did, and the
/// caller must refuse the launch.
///
/// # Errors
///
/// Returns an error when `bytes` is shorter than [`TRAP_RECORD_BYTES`]. A short
/// readback cannot be read as "no trap": the flag word may be the part that was
/// not transferred, so the only safe answer is to refuse.
pub fn decode_trap_record(bytes: &[u8]) -> Result<Option<TrapRecord>, BackendError> {
    let Some(record) = bytes.get(..TRAP_RECORD_BYTES) else {
        return Err(BackendError::new(format!(
            "trap sidecar readback returned {} bytes but a trap record is {TRAP_RECORD_BYTES} bytes. Fix: allocate and read back the sidecar as vyre_lower::TRAP_SIDECAR_WORDS u32 words.",
            bytes.len()
        )));
    };
    let word = |index: usize| -> u32 {
        let start = index * 4;
        u32::from_le_bytes([
            record[start],
            record[start + 1],
            record[start + 2],
            record[start + 3],
        ])
    };
    if word(0) == 0 {
        return Ok(None);
    }
    Ok(Some(TrapRecord {
        address: word(1),
        tag_code: word(2),
        lane: word(3),
    }))
}

impl TrapRecord {
    /// Render this record as the body of a backend's refusal message.
    ///
    /// `tag_for_code` resolves the tag text; every backend carries its tag table
    /// differently (parsed from module text on one target, held beside the
    /// pipeline on another), so resolution is the caller's and the wording is not.
    pub fn describe(&self, tag_for_code: impl FnOnce(u32) -> Option<String>) -> String {
        let tag = tag_for_code(self.tag_code).unwrap_or_else(|| UNKNOWN_TRAP_TAG.to_owned());
        format!(
            "address={}, tag_code={}, lane={}, tag=`{tag}`.",
            self.address, self.tag_code, self.lane
        )
    }
}
