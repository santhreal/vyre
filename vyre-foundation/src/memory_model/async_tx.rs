//! Closed state machine lifecycle for asynchronous memory transactions / DMA / pipeline stages.

/// Closed state machine lifecycle for asynchronous memory transactions / DMA / pipeline stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AsyncTransactionLifecycle {
    /// Transaction descriptor submitted to hardware / queue.
    Submitted,
    /// Transaction in flight on memory / copy engine.
    InFlight,
    /// Transaction arrived at completion barrier / stage counter incremented.
    Arrived,
    /// Transaction committed and data visible in destination storage domain.
    Committed,
    /// Transaction failed due to bus error / out of bounds / hardware fault.
    Failed,
    /// Transaction cancelled or aborted before commit.
    Aborted,
}

impl AsyncTransactionLifecycle {
    /// Every async transaction lifecycle state in the closed contract.
    pub const ALL: [Self; 6] = [
        Self::Submitted,
        Self::InFlight,
        Self::Arrived,
        Self::Committed,
        Self::Failed,
        Self::Aborted,
    ];

    /// Stable wire tag for async transaction lifecycle.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Submitted => 0,
            Self::InFlight => 1,
            Self::Arrived => 2,
            Self::Committed => 3,
            Self::Failed => 4,
            Self::Aborted => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `AsyncTransactionLifecycle`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Submitted),
            1 => Ok(Self::InFlight),
            2 => Ok(Self::Arrived),
            3 => Ok(Self::Committed),
            4 => Ok(Self::Failed),
            5 => Ok(Self::Aborted),
            other => Err(format!(
                "InvalidDiscriminant: async transaction lifecycle tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for async transaction lifecycle.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::InFlight => "in_flight",
            Self::Arrived => "arrived",
            Self::Committed => "committed",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }

    /// Whether this lifecycle state represents a terminal state.
    #[must_use]
    #[inline]
    pub const fn is_terminal(self) -> bool {
        match self {
            Self::Committed | Self::Failed | Self::Aborted => true,
            Self::Submitted | Self::InFlight | Self::Arrived => false,
        }
    }

    /// Whether this transaction has successfully completed and committed.
    #[must_use]
    #[inline]
    pub const fn is_successful_commit(self) -> bool {
        matches!(self, Self::Committed)
    }

    /// Whether this transaction is currently in flight or pending arrival.
    #[must_use]
    #[inline]
    pub const fn is_in_flight(self) -> bool {
        matches!(self, Self::Submitted | Self::InFlight | Self::Arrived)
    }

    /// Whether this lifecycle state can legally transition to `next`.
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        match (self, next) {
            (Self::Submitted, Self::InFlight | Self::Failed | Self::Aborted) => true,
            (Self::InFlight, Self::Arrived | Self::Committed | Self::Failed | Self::Aborted) => {
                true
            }
            (Self::Arrived, Self::Committed | Self::Failed | Self::Aborted) => true,
            _ => false,
        }
    }
}

/// Exhaustive compile-time check for [`AsyncTransactionLifecycle`].
#[must_use]
pub const fn exhaustiveness_check_async_transaction_lifecycle(
    val: AsyncTransactionLifecycle,
) -> &'static str {
    match val {
        AsyncTransactionLifecycle::Submitted => "Submitted",
        AsyncTransactionLifecycle::InFlight => "InFlight",
        AsyncTransactionLifecycle::Arrived => "Arrived",
        AsyncTransactionLifecycle::Committed => "Committed",
        AsyncTransactionLifecycle::Failed => "Failed",
        AsyncTransactionLifecycle::Aborted => "Aborted",
    }
}
