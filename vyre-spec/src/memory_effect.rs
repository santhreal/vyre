//! Closed memory effect classifications and semantics.

/// Closed memory effect classification for operations and statement regions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryEffect {
    /// Pure computation with no memory reads, writes, or synchronization.
    Pure,
    /// Read-only access to memory locations.
    Read,
    /// Write mutation to memory locations.
    Write,
    /// Atomic read-modify-write on memory locations.
    Atomic,
    /// Execution or memory barrier synchronization.
    Synchronizing,
    /// Asynchronous data transfer initiation or wait.
    AsyncTransfer,
    /// Collective group communication.
    Collective,
    /// Trapping or faulting execution effect.
    Fault,
}

impl MemoryEffect {
    /// Every memory effect variant in the closed contract.
    pub const ALL: [Self; 8] = [
        Self::Pure,
        Self::Read,
        Self::Write,
        Self::Atomic,
        Self::Synchronizing,
        Self::AsyncTransfer,
        Self::Collective,
        Self::Fault,
    ];

    /// Whether this effect mutates memory.
    #[must_use]
    #[inline]
    pub const fn is_mutating(self) -> bool {
        match self {
            Self::Write | Self::Atomic | Self::AsyncTransfer => true,
            Self::Pure | Self::Read | Self::Synchronizing | Self::Collective | Self::Fault => false,
        }
    }

    /// Whether this effect synchronizes participant execution.
    #[must_use]
    #[inline]
    pub const fn is_synchronizing(self) -> bool {
        match self {
            Self::Synchronizing | Self::Collective => true,
            Self::Pure | Self::Read | Self::Write | Self::Atomic | Self::AsyncTransfer | Self::Fault => false,
        }
    }
}
