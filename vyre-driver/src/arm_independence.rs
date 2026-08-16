//! D2 substrate: independent-arm detection for queue-parallel dispatch.
//!
//! Two megakernel arms can execute on independent native streams /
//! portable queues if and only if their writes are disjoint and neither
//! reads what the other writes. The dispatcher uses this to fan out
//! N concurrent stream submissions instead of serialising on one
//! queue  -  wins on backends with multiple async-engine count > 1
//! (every modern discrete GPU + every portable adapter).
//!
//! This module owns the *decision*: given two `BindingPlan`s
//! (or any structure that names input/output binding slots), are
//! they safe to dispatch concurrently? Pure analysis, no Program
//! walking  -  the caller passes the already-derived input/output
//! sets.

use smallvec::SmallVec;

/// Small sorted binding-slot set optimized for dispatch-policy hot paths.
///
/// Dispatch summaries usually contain fewer than eight bindings. Keeping
/// slots inline avoids allocating two tree nodes per arm while preserving
/// deterministic equality/debug output and O(log n) membership checks.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BindingSlotSet {
    slots: SmallVec<[u32; 8]>,
}

impl BindingSlotSet {
    /// Insert `slot`, preserving sorted unique order.
    pub fn insert(&mut self, slot: u32) -> bool {
        match self.slots.binary_search(&slot) {
            Ok(_) => false,
            Err(pos) => {
                self.slots.insert(pos, slot);
                true
            }
        }
    }

    /// True when `slot` is present.
    #[must_use]
    pub fn contains(&self, slot: &u32) -> bool {
        self.slots.binary_search(slot).is_ok()
    }

    /// True when this set and `other` share at least one slot.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        let (small, large) = if self.slots.len() <= other.slots.len() {
            (self, other)
        } else {
            (other, self)
        };
        small.slots.iter().any(|slot| large.contains(slot))
    }
}

impl FromIterator<u32> for BindingSlotSet {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut set = Self::default();
        for slot in iter {
            set.insert(slot);
        }
        set
    }
}

/// Input/output set summary for one arm. Two of these are compared
/// pairwise to decide whether the arms can launch on independent
/// streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmBindingSummary {
    /// Slots this arm reads (LoadGlobal/LoadShared/LoadConstant/
    /// AsyncLoad/Atomic-input/BufferLength source operand).
    pub reads: BindingSlotSet,
    /// Slots this arm writes (StoreGlobal/StoreShared/AsyncStore/
    /// Atomic-target).
    pub writes: BindingSlotSet,
}

impl ArmBindingSummary {
    /// Empty summary (no reads, no writes). Useful as a starting
    /// accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for ArmBindingSummary {
    fn default() -> Self {
        Self {
            reads: BindingSlotSet::default(),
            writes: BindingSlotSet::default(),
        }
    }
}

/// Verdict from [`can_dispatch_concurrently`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmIndependenceVerdict {
    /// Two arms touch disjoint resources; safe to launch on
    /// independent streams.
    Independent,
    /// Two arms share at least one binding slot in a way that
    /// would race on concurrent dispatch. Caller must serialise
    /// (single stream, or stream A then sync then stream B).
    SerializeRequired {
        /// Why serialisation is required  -  names the offending
        /// access pattern so telemetry and diagnostics can attribute
        /// the missed concurrency.
        reason: ArmConflict,
    },
}

/// Reason two arms cannot launch concurrently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmConflict {
    /// Both arms write the same slot  -  the second write would race
    /// with the first.
    WriteWriteConflict,
    /// Arm A writes a slot that arm B reads  -  read-after-write.
    /// B would see either the old or new value depending on stream
    /// order; observable nondeterminism.
    ReadAfterWrite,
    /// Arm A reads a slot that arm B writes  -  write-after-read.
    /// Symmetric of the above; equally observable.
    WriteAfterRead,
}

/// Decide whether two arms (described by their `ArmBindingSummary`s)
/// can dispatch concurrently on independent streams. Pure set
/// arithmetic  -  no IR walk, no Program clone, and no heap allocation
/// for the common <= 8 binding-slot case.
///
/// Read-only ↔ read-only on the same slot is always safe and does
/// NOT count as a conflict.
#[must_use]
pub fn can_dispatch_concurrently(
    a: &ArmBindingSummary,
    b: &ArmBindingSummary,
) -> ArmIndependenceVerdict {
    if a.writes.intersects(&b.writes) {
        return ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::WriteWriteConflict,
        };
    }
    if a.writes.intersects(&b.reads) {
        return ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::ReadAfterWrite,
        };
    }
    if a.reads.intersects(&b.writes) {
        return ArmIndependenceVerdict::SerializeRequired {
            reason: ArmConflict::WriteAfterRead,
        };
    }
    ArmIndependenceVerdict::Independent
}
