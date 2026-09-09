//! Linear types and linear resource state declarations.

/// Closed linear resource kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum LinearResourceKind {
    /// Buffer allocation / region handle.
    BufferHandle,
    /// Asynchronous DMA / copy transaction tag.
    AsyncTransaction,
    /// Execution or memory fence token.
    FenceToken,
    /// Monotonic state epoch token.
    EpochToken,
    /// Collective communication group token.
    CollectiveToken,
}

impl LinearResourceKind {
    /// Every linear resource kind variant in the closed contract.
    pub const ALL: [Self; 5] = [
        Self::BufferHandle,
        Self::AsyncTransaction,
        Self::FenceToken,
        Self::EpochToken,
        Self::CollectiveToken,
    ];
}

/// Closed linear obligation consumption state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum LinearState {
    /// Active obligation that has not yet been consumed.
    Unconsumed,
    /// Successfully consumed / discharged obligation.
    Consumed,
    /// Temporarily borrowed under active scope.
    Borrowed,
    /// Poisoned due to an execution or memory fault.
    Poisoned,
}

impl LinearState {
    /// Every linear state variant in the closed contract.
    pub const ALL: [Self; 4] = [
        Self::Unconsumed,
        Self::Consumed,
        Self::Borrowed,
        Self::Poisoned,
    ];

    /// Whether this state represents an active unconsumed obligation.
    #[must_use]
    #[inline]
    pub const fn is_unconsumed(self) -> bool {
        matches!(self, Self::Unconsumed)
    }

    /// Whether this state represents a consumed/discharged obligation.
    #[must_use]
    #[inline]
    pub const fn is_consumed(self) -> bool {
        matches!(self, Self::Consumed)
    }
}
