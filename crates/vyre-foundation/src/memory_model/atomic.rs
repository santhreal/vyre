//! Closed atomic memory ordering for read-modify-write and load/store operations.

/// Closed atomic memory ordering for atomic read-modify-write and load/store operations.
///
/// Synchronization intent must be explicitly stated. There is no default ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AtomicOrdering {
    /// No synchronization beyond single-location atomicity.
    Relaxed,
    /// Acquires prior writes released by another thread.
    Acquire,
    /// Releases prior writes to acquiring threads.
    Release,
    /// Both acquire and release semantics on the same memory location.
    AcqRel,
    /// Sequentially consistent total order across all threads in the relevant memory scope.
    SeqCst,
}

impl AtomicOrdering {
    /// Every atomic ordering variant in the closed contract.
    pub const ALL: [Self; 5] = [
        Self::Relaxed,
        Self::Acquire,
        Self::Release,
        Self::AcqRel,
        Self::SeqCst,
    ];

    /// Stable wire tag for atomic ordering.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Relaxed => 0,
            Self::Acquire => 1,
            Self::Release => 2,
            Self::AcqRel => 3,
            Self::SeqCst => 4,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `AtomicOrdering`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Relaxed),
            1 => Ok(Self::Acquire),
            2 => Ok(Self::Release),
            3 => Ok(Self::AcqRel),
            4 => Ok(Self::SeqCst),
            other => Err(format!(
                "InvalidDiscriminant: atomic ordering tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this atomic ordering.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Relaxed => "relaxed",
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::SeqCst => "seq_cst",
        }
    }

    /// Whether this ordering incorporates acquire synchronization.
    #[must_use]
    #[inline]
    pub const fn is_acquire(self) -> bool {
        match self {
            Self::Acquire | Self::AcqRel | Self::SeqCst => true,
            Self::Relaxed | Self::Release => false,
        }
    }

    /// Whether this ordering incorporates release synchronization.
    #[must_use]
    #[inline]
    pub const fn is_release(self) -> bool {
        match self {
            Self::Release | Self::AcqRel | Self::SeqCst => true,
            Self::Relaxed | Self::Acquire => false,
        }
    }

    /// Whether this ordering enforces sequential consistency.
    #[must_use]
    #[inline]
    pub const fn is_seq_cst(self) -> bool {
        matches!(self, Self::SeqCst)
    }

    /// Join two atomic orderings into the weakest ordering satisfying both.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        use AtomicOrdering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
        match (self, other) {
            (SeqCst, _) | (_, SeqCst) => SeqCst,
            (AcqRel, _) | (_, AcqRel) | (Acquire, Release) | (Release, Acquire) => AcqRel,
            (Acquire, Acquire) => Acquire,
            (Release, Release) => Release,
            (Relaxed, ordering) | (ordering, Relaxed) => ordering,
        }
    }
}

/// Exhaustive compile-time check for [`AtomicOrdering`].
#[must_use]
pub const fn exhaustiveness_check_atomic_ordering(val: AtomicOrdering) -> &'static str {
    match val {
        AtomicOrdering::Relaxed => "Relaxed",
        AtomicOrdering::Acquire => "Acquire",
        AtomicOrdering::Release => "Release",
        AtomicOrdering::AcqRel => "AcqRel",
        AtomicOrdering::SeqCst => "SeqCst",
    }
}
