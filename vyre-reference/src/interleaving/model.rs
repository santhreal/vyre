//! Configuration, access records, and the findings a race exploration reports.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{
    AsyncTransactionLifecycle, AtomicOrdering, MemoryScope, StorageDomain,
};

/// Configuration for bounded interleaving exploration.
#[derive(Clone, Debug)]
pub struct InterleavingConfig {
    /// Workgroup dimensions `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Grid dimensions `[x, y, z]`.
    pub grid_size: [u32; 3],
    /// Maximum distinct scheduling permutations to explore.
    pub max_interleavings: usize,
    /// Maximum execution steps per invocation before termination.
    pub step_bound: usize,
}

impl Default for InterleavingConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [2, 1, 1],
            grid_size: [1, 1, 1],
            max_interleavings: 16,
            step_bound: 1024,
        }
    }
}

/// Result report from bounded interleaving exploration.
#[derive(Clone, Debug)]
pub struct InterleavingReport {
    /// Number of distinct scheduling interleavings explored.
    pub explored_schedules: usize,
    /// Whether all explored interleavings proved race-free.
    pub race_free: bool,
    /// Final output buffer values verified across interleavings.
    pub final_outputs: FxHashMap<String, Vec<u32>>,
}

/// Access kind on a buffer location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAccessKind {
    /// Non-atomic read.
    Read,
    /// Non-atomic write.
    Write,
    /// Atomic read-modify-write with ordering and scope.
    Atomic {
        /// Atomic ordering.
        ordering: AtomicOrdering,
        /// Memory scope.
        scope: MemoryScope,
    },
    /// Asynchronous DMA / copy transaction.
    AsyncTransfer {
        /// Transaction lifecycle.
        lifecycle: AsyncTransactionLifecycle,
    },
}

/// A recorded memory access by a specific invocation.
#[derive(Clone, Debug)]
pub struct MemoryAccessRecord {
    /// Invocation coordinates `[x, y, z]`.
    pub invocation: [u32; 3],
    /// Workgroup coordinates `[x, y, z]` the invocation belongs to.
    pub workgroup: [u32; 3],
    /// Memory access kind.
    pub kind: MemoryAccessKind,
    /// Barrier phase / generation when access occurred.
    pub barrier_phase: u64,
    /// Grid fence generation when the access occurred.
    pub grid_phase: u64,
    /// Memory visibility scope of the operation.
    pub memory_scope: MemoryScope,
    /// Storage domain accessed.
    pub storage_domain: StorageDomain,
    /// Order this access was recorded in, within one explored schedule.
    pub sequence: u64,
}

/// One hazard a bounded exploration reported.
///
/// The two variants are distinct classes. An [`UnsynchronizedAccess`] is a
/// conflict inside one explored order: two invocations reached one location in
/// the same synchronization phase with no barrier, fence, or atomic between
/// them. A [`ScheduleDisagreement`] is a conflict between two explored orders:
/// the same dispatch produced different output bytes depending on which order
/// the lanes were stepped in. A program can carry either without the other, so
/// neither one subsumes the other.
///
/// [`UnsynchronizedAccess`]: RaceFinding::UnsynchronizedAccess
/// [`ScheduleDisagreement`]: RaceFinding::ScheduleDisagreement
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RaceFinding {
    /// Two explored step orders produced different bytes for one output.
    ScheduleDisagreement {
        /// Debug rendering of the first step order.
        first_order: String,
        /// Debug rendering of the step order that disagreed with it.
        second_order: String,
        /// Position of the first output value that differs.
        output_index: usize,
    },
    /// Two invocations conflicted on one location in one synchronization phase.
    UnsynchronizedAccess {
        /// Buffer the conflict occurred on.
        buffer: String,
        /// Element index within that buffer.
        index: u64,
        /// Invocation that reached the location first, and its access kind.
        first: [u32; 3],
        /// Access kind of the first invocation.
        first_kind: MemoryAccessKind,
        /// Invocation that conflicted with it.
        second: [u32; 3],
        /// Access kind of the second invocation.
        second_kind: MemoryAccessKind,
        /// Barrier phase both accesses fell in.
        barrier_phase: u64,
    },
}

impl std::fmt::Display for RaceFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScheduleDisagreement {
                first_order,
                second_order,
                output_index,
            } => write!(
                f,
                "schedule exploration disagreed: output {output_index} differs between step order \
                 {first_order} and {second_order}. Fix: give every shared output slot a single \
                 writer, or write it through a commutative atomic, so the result does not depend \
                 on the order the lanes were stepped in."
            ),
            Self::UnsynchronizedAccess {
                buffer,
                index,
                first,
                first_kind,
                second,
                second_kind,
                barrier_phase,
            } => write!(
                f,
                "Data race detected on buffer `{buffer}` at index {index}: unsynchronized access \
                 by invocation {first:?} ({first_kind:?}) races with access by invocation \
                 {second:?} ({second_kind:?}) in barrier phase {barrier_phase}"
            ),
        }
    }
}

/// Result of a bounded race exploration over one submitted program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaceExplorationReport {
    /// Number of deterministic step orders the exploration executed.
    pub orders_explored: usize,
    /// Every distinct hazard the exploration reported, first occurrence order.
    pub findings: Vec<RaceFinding>,
    /// Interpreter steps the whole exploration charged against the budget.
    pub steps_executed: u64,
}

impl RaceExplorationReport {
    /// True when no explored order reported a hazard.
    #[must_use]
    pub fn is_race_free(&self) -> bool {
        self.findings.is_empty()
    }
}
