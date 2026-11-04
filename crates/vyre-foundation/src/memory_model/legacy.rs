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

    /// Wire tag reserved for the grid-scope barrier, the one variant this
    /// model adds to the closed atomic set.
    const GRID_SYNC_WIRE_TAG: u8 = 5;

    /// Stable wire tag for this ordering.
    ///
    /// The five atomic variants carry the tag [`AtomicOrdering`] assigns them.
    /// Both models encode into one wire schema, so a second tag table here
    /// would be free to drift away from the one that decodes.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self.to_atomic_ordering() {
            Some(ordering) => ordering.wire_tag(),
            None => Self::GRID_SYNC_WIRE_TAG,
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
        if tag == Self::GRID_SYNC_WIRE_TAG {
            return Ok(Self::GridSync);
        }
        AtomicOrdering::from_wire_tag(tag)
            .map(Self::from_atomic_ordering)
            .map_err(|_| format!(
                "InvalidDiscriminant: memory ordering tag {tag} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            ))
    }

    /// Widen a closed atomic ordering into the legacy combined model.
    #[must_use]
    #[inline]
    pub const fn from_atomic_ordering(ordering: AtomicOrdering) -> Self {
        match ordering {
            AtomicOrdering::Relaxed => Self::Relaxed,
            AtomicOrdering::Acquire => Self::Acquire,
            AtomicOrdering::Release => Self::Release,
            AtomicOrdering::AcqRel => Self::AcqRel,
            AtomicOrdering::SeqCst => Self::SeqCst,
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
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst => {
                ExecutionScope::Workgroup
            }
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
    ///
    /// Grid synchronization absorbs everything. Below it the lattice is the
    /// closed atomic one, so [`AtomicOrdering::join`] decides it.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        match (self.to_atomic_ordering(), other.to_atomic_ordering()) {
            (Some(left), Some(right)) => Self::from_atomic_ordering(left.join(right)),
            _ => Self::GridSync,
        }
    }
}
