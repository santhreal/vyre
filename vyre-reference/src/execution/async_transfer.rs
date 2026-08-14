//! Async byte-copy semantics shared by both reference node executors.
//!
//! `AsyncLoad` and `AsyncStore` queue a byte copy that `AsyncWait` observes.
//! Which end of the copy the declared offset applies to, how a size expression
//! becomes a host byte count, and what happens to a span that runs off the end
//! of a buffer are one decision each, and both the statement evaluator and the
//! hashmap interpreter reach them here. The copy itself goes through
//! [`Buffer::read_window`](crate::oob::Buffer::read_window) and
//! [`Buffer::write_window`](crate::oob::Buffer::write_window), which own the
//! poison policy, so neither executor can launder a poisoned reference buffer
//! into a golden value.

use std::sync::Arc;

use crate::oob::Buffer;
use crate::value::Value;
use crate::ReferenceError;

/// One queued byte copy awaiting its `AsyncWait`.
pub(crate) enum AsyncTransfer {
    /// Copy `payload` into `destination` starting at byte offset `start`.
    Copy {
        destination: Arc<str>,
        start: usize,
        payload: Vec<u8>,
    },
}

impl AsyncTransfer {
    /// An `AsyncLoad` reads the SOURCE at the declared offset and lands the
    /// bytes at the head of the destination.
    pub(crate) fn load(destination: &str, payload: Vec<u8>) -> Self {
        Self::Copy {
            destination: Arc::from(destination),
            start: 0,
            payload,
        }
    }

    /// An `AsyncStore` reads the SOURCE from its head and lands the bytes at the
    /// declared offset in the destination.
    pub(crate) fn store(destination: &str, start: usize, payload: Vec<u8>) -> Self {
        Self::Copy {
            destination: Arc::from(destination),
            start,
            payload,
        }
    }

    /// Buffer this transfer writes into.
    pub(crate) fn destination(&self) -> &str {
        match self {
            Self::Copy { destination, .. } => destination,
        }
    }

    /// Apply the queued copy to its already-resolved destination buffer.
    ///
    /// # Panics
    /// Panics when the buffer's byte lock is poisoned; see
    /// [`Buffer::read_window`](crate::oob::Buffer::read_window).
    pub(crate) fn apply_to(&self, buffer: &Buffer) {
        match self {
            Self::Copy { start, payload, .. } => buffer.write_window(*start, payload),
        }
    }
}

/// Convert an evaluated offset or size into a host byte count.
///
/// `label` names the operand so a program that passes a negative or oversized
/// count is told which one it was.
pub(crate) fn byte_count(value: &Value, label: &str) -> Result<usize, ReferenceError> {
    usize::try_from(value.try_as_u64().ok_or_else(|| {
        ReferenceError::new(format!(
            "{label} cannot be represented as u64. Fix: use an in-range non-negative byte count."
        ))
    })?)
    .map_err(|_| {
        ReferenceError::new(format!(
            "{label} exceeds host usize. Fix: reduce the async transfer span."
        ))
    })
}
