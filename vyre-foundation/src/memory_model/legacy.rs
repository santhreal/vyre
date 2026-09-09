//! Legacy combined memory ordering model retained for transitional compatibility.
//!
//! Note: There is intentionally NO `Default` implementation for `MemoryOrdering`.
//! Synchronization intent must be explicitly chosen.

use super::atomic::AtomicOrdering;
use super::scope::{ExecutionScope, MemoryScope};

/// Memory ordering attached to atomic and barrier operations.
///
/// This combined type conflated atomic ordering, memory visibility, and execution barrier.
/// Modern code should prefer separate closed types: [`AtomicOrdering`], [`MemoryScope`],
/// and [`ExecutionScope`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryOrdering {
    /// No synchronization beyond atomicity of the operation.
    Relaxed,
    /// Subsequent reads observe writes released by another participant.
    Acquire,
    /// Prior writes become visible to acquiring participants.
    Release,
    /// Acquire and release semantics in one operation.
    AcqRel,
    /// Single total order across sequentially consistent operations
    /// within the issuing thread's workgroup.
    SeqCst,
    /// Cross-grid synchronization. Every thread in the dispatch waits
    /// here, and every prior write is globally visible after the
    /// barrier returns.
    GridSync,
}

impl MemoryOrdering {
    /// Every memory ordering variant in the legacy model.
    pub const ALL: [Self; 6] = [
        Self::Relaxed,
        Self::Acquire,
        Self::Release,
        Self::AcqRel,
        Self::SeqCst,
        Self::GridSync,
    ];

    /// Stable wire tag for this ordering.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Relaxed => 0,
            Self::Acquire => 1,
            Self::Release => 2,
            Self::AcqRel => 3,
            Self::SeqCst => 4,
            Self::GridSync => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an actionable error when `tag` is not assigned to a memory
    /// ordering in this schema.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Relaxed),
            1 => Ok(Self::Acquire),
            2 => Ok(Self::Release),
            3 => Ok(Self::AcqRel),
            4 => Ok(Self::SeqCst),
            5 => Ok(Self::GridSync),
            other => Err(format!(
                "InvalidDiscriminant: memory ordering tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Whether this ordering is valid for an atomic RMW operation.
    /// `GridSync` is barrier-only and not a valid atomic ordering.
    #[must_use]
    #[inline]
    pub const fn is_valid_for_atomic_rmw(self) -> bool {
        matches!(
            self,
            Self::Relaxed | Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst
        )
    }

    /// Whether this ordering is valid for a barrier.
    #[must_use]
    #[inline]
    pub const fn is_valid_for_barrier(self) -> bool {
        matches!(
            self,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst | Self::GridSync
        )
    }

    /// Whether this ordering requires cross-grid synchronization.
    #[must_use]
    #[inline]
    pub const fn requires_grid_sync(self) -> bool {
        matches!(self, Self::GridSync)
    }

    /// Convert to canonical closed [`AtomicOrdering`] if valid for atomic operations.
    #[must_use]
    pub const fn to_atomic_ordering(self) -> Option<AtomicOrdering> {
        match self {
            Self::Relaxed => Some(AtomicOrdering::Relaxed),
            Self::Acquire => Some(AtomicOrdering::Acquire),
            Self::Release => Some(AtomicOrdering::Release),
            Self::AcqRel => Some(AtomicOrdering::AcqRel),
            Self::SeqCst => Some(AtomicOrdering::SeqCst),
            Self::GridSync => None,
        }
    }

    /// Primary execution synchronization scope implied by this ordering.
    #[must_use]
    pub const fn execution_scope(self) -> ExecutionScope {
        match self {
            Self::Relaxed => ExecutionScope::Thread,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst => ExecutionScope::Workgroup,
            Self::GridSync => ExecutionScope::Grid,
        }
    }

    /// Primary memory coherence scope implied by this ordering.
    #[must_use]
    pub const fn memory_scope(self) -> MemoryScope {
        match self {
            Self::Relaxed => MemoryScope::Thread,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst => MemoryScope::Workgroup,
            Self::GridSync => MemoryScope::Device,
        }
    }

    /// Join two orderings to the weakest ordering that satisfies both.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        use MemoryOrdering::{AcqRel, Acquire, GridSync, Relaxed, Release, SeqCst};
        match (self, other) {
            (GridSync, _) | (_, GridSync) => GridSync,
            (SeqCst, _) | (_, SeqCst) => SeqCst,
            (AcqRel, _) | (_, AcqRel) | (Acquire, Release) | (Release, Acquire) => AcqRel,
            (Acquire, Acquire) => Acquire,
            (Release, Release) => Release,
            (Relaxed, ordering) | (ordering, Relaxed) => ordering,
        }
    }
}
