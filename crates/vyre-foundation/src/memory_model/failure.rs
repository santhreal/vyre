//! Closed fault, trap, and cancellation policies.

/// Closed fault, trap, and cancellation policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum FailureCancellationBehavior {
    /// Hard execution trap halting the kernel / thread.
    Trap,
    /// Mark destination buffer / register as poisoned with propagating error token.
    Poison,
    /// Propagate error status code to caller / runtime return slot.
    Propagate,
    /// Abort the entire dispatch / launch immediately.
    AbortKernel,
    /// Ignore fault / silently drop invalid transaction.
    Ignore,
}

impl FailureCancellationBehavior {
    /// Every failure cancellation behavior variant in the closed contract.
    pub const ALL: [Self; 5] = [
        Self::Trap,
        Self::Poison,
        Self::Propagate,
        Self::AbortKernel,
        Self::Ignore,
    ];

    /// Stable wire tag for failure cancellation behavior.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Trap => 0,
            Self::Poison => 1,
            Self::Propagate => 2,
            Self::AbortKernel => 3,
            Self::Ignore => 4,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `FailureCancellationBehavior`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Trap),
            1 => Ok(Self::Poison),
            2 => Ok(Self::Propagate),
            3 => Ok(Self::AbortKernel),
            4 => Ok(Self::Ignore),
            other => Err(format!(
                "InvalidDiscriminant: failure cancellation behavior tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for failure cancellation behavior.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Trap => "trap",
            Self::Poison => "poison",
            Self::Propagate => "propagate",
            Self::AbortKernel => "abort_kernel",
            Self::Ignore => "ignore",
        }
    }

    /// Whether this failure policy unconditionally halts kernel execution.
    #[must_use]
    #[inline]
    pub const fn halts_execution(self) -> bool {
        match self {
            Self::Trap | Self::AbortKernel => true,
            Self::Poison | Self::Propagate | Self::Ignore => false,
        }
    }

    /// Whether this failure policy poisons memory or registers.
    #[must_use]
    #[inline]
    pub const fn poisons_memory(self) -> bool {
        matches!(self, Self::Poison)
    }

    /// Whether this failure policy propagates an error code to the host caller.
    #[must_use]
    #[inline]
    pub const fn propagates_to_caller(self) -> bool {
        matches!(self, Self::Propagate)
    }
}

/// Exhaustive compile-time check for [`FailureCancellationBehavior`].
#[must_use]
pub const fn exhaustiveness_check_failure_cancellation_behavior(
    val: FailureCancellationBehavior,
) -> &'static str {
    match val {
        FailureCancellationBehavior::Trap => "Trap",
        FailureCancellationBehavior::Poison => "Poison",
        FailureCancellationBehavior::Propagate => "Propagate",
        FailureCancellationBehavior::AbortKernel => "AbortKernel",
        FailureCancellationBehavior::Ignore => "Ignore",
    }
}
