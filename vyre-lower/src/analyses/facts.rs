//! Device facts a neutral analysis needs from its caller.
//!
//! A neutral analysis computes nothing from a vendor name and assumes no
//! capacity. Every fact here is absent until a target reports it, and an
//! analysis whose fact is absent does not run: its section of the report is
//! absent too, rather than computed from a guessed limit. A default capacity
//! stored in a shared crate is a device fact recorded in the wrong crate,
//! because it is right for the device it was copied from and silently wrong
//! for every other one.

use core::num::NonZeroU32;

/// Device capacities a caller states before a neutral analysis runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalysisFacts {
    shared_memory_banks: Option<NonZeroU32>,
    shared_memory_bytes: Option<NonZeroU32>,
    constant_buffer_bytes: Option<NonZeroU32>,
}

impl AnalysisFacts {
    /// Facts for a caller that has reported nothing. Every analysis that
    /// needs a capacity is skipped.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            shared_memory_banks: None,
            shared_memory_bytes: None,
            constant_buffer_bytes: None,
        }
    }

    /// State the number of shared-memory banks the target reported.
    #[must_use]
    pub const fn with_shared_memory_banks(mut self, banks: NonZeroU32) -> Self {
        self.shared_memory_banks = Some(banks);
        self
    }

    /// State the per-workgroup shared-memory capacity the target reported.
    #[must_use]
    pub const fn with_shared_memory_bytes(mut self, bytes: NonZeroU32) -> Self {
        self.shared_memory_bytes = Some(bytes);
        self
    }

    /// State the constant-buffer capacity the target reported.
    #[must_use]
    pub const fn with_constant_buffer_bytes(mut self, bytes: NonZeroU32) -> Self {
        self.constant_buffer_bytes = Some(bytes);
        self
    }

    /// Shared-memory bank count, absent when the caller stated none.
    #[must_use]
    pub const fn shared_memory_banks(&self) -> Option<NonZeroU32> {
        self.shared_memory_banks
    }

    /// Per-workgroup shared-memory capacity, absent when the caller stated
    /// none.
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> Option<NonZeroU32> {
        self.shared_memory_bytes
    }

    /// Constant-buffer capacity, absent when the caller stated none.
    #[must_use]
    pub const fn constant_buffer_bytes(&self) -> Option<NonZeroU32> {
        self.constant_buffer_bytes
    }
}
