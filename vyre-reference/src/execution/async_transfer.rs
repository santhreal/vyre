//! Async byte-copy semantics shared by both reference node executors.
//!
//! `AsyncLoad` and `AsyncStore` queue a byte copy that `AsyncWait` observes.
//! Which end of the copy the declared offset applies to, how a size expression
//! becomes a host byte count, what happens to a span that runs off the end
//! of a buffer, and whether a set of in-flight tags is legal at all are one
//! decision each, and both the statement evaluator and the hashmap interpreter
//! reach them here. The copy itself goes through
//! [`Buffer::read_window`](crate::oob::Buffer::read_window) and
//! [`Buffer::write_window`](crate::oob::Buffer::write_window), which own the
//! poison policy, so neither executor can launder a poisoned reference buffer
//! into a golden value.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::oob::Buffer;
use crate::value::Value;
use crate::workgroup::InvocationIds;
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

/// Every async tag one invocation has started and not yet waited on.
///
/// Three rules decide whether a tag set is legal: a tag may not be started
/// while it is already in flight, a wait must have something to wait on, and an
/// invocation may not reach its end with a transfer still queued. Both
/// reference executors reach all three here, so the reference tree cannot hold
/// two verdicts for one program.
pub(crate) struct PendingAsyncTransfers {
    in_flight: FxHashMap<Arc<str>, AsyncTransfer>,
}

impl PendingAsyncTransfers {
    pub(crate) fn new() -> Self {
        Self {
            in_flight: FxHashMap::default(),
        }
    }

    /// Queue `transfer` under `tag`.
    pub(crate) fn begin(
        &mut self,
        tag: &str,
        transfer: AsyncTransfer,
    ) -> Result<(), ReferenceError> {
        if self.in_flight.contains_key(tag) {
            return Err(ReferenceError::new(format!(
                "async transfer tag `{tag}` was started more than once before a matching wait. Fix: reuse the tag only after AsyncWait completes."
            )));
        }
        self.in_flight.insert(Arc::from(tag), transfer);
        Ok(())
    }

    /// Take the transfer queued under `tag`.
    pub(crate) fn finish(&mut self, tag: &str) -> Result<AsyncTransfer, ReferenceError> {
        self.in_flight.remove(tag).ok_or_else(|| ReferenceError::new(format!(
            "async wait for tag `{tag}` has no matching async transfer. Fix: emit AsyncLoad or AsyncStore before AsyncWait."
        )))
    }

    /// Refuse an invocation that reached its end with a transfer still queued.
    ///
    /// A transfer nobody waited on means the invocation's result depends on
    /// bytes nobody synchronized, so this fails closed rather than accepting a
    /// value the GPU is free to compute differently.
    ///
    /// The pending set is unordered, so the tag named is the lexicographically
    /// smallest one: a program that leaves several tags in flight reports the
    /// same tag on every run.
    pub(crate) fn assert_drained(&self, ids: InvocationIds) -> Result<(), ReferenceError> {
        match self.in_flight.keys().min() {
            None => Ok(()),
            Some(tag) => Err(ReferenceError::new(format!(
                "invocation {ids:?} completed with async transfer tag `{tag}` still pending. Fix: add AsyncWait for every AsyncLoad/AsyncStore tag before Return or end-of-program."
            ))),
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
