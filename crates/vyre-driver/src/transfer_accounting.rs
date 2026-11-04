//! Backend-neutral transfer accounting policy.
//!
//! Backends repeatedly account bytes, operations, copy counts, and copy slots
//! while staging host/device transfers. This module centralizes the checked
//! arithmetic and leaves each caller to supply only domain wording.

use crate::BackendError;

/// Error wording and split guidance for a transfer-accounting domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferAccountingPolicy {
    domain: &'static str,
    fix_action: &'static str,
}

impl TransferAccountingPolicy {
    /// Create a transfer-accounting policy.
    #[must_use]
    pub const fn new(domain: &'static str, fix_action: &'static str) -> Self {
        Self { domain, fix_action }
    }

    /// Convert a host-sized byte count to `u64` without truncation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when `bytes` cannot fit in `u64`.
    pub fn bytes_to_u64(self, bytes: usize, label: &str) -> Result<u64, BackendError> {
        u64::try_from(bytes).map_err(|_| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} {label} exceeds u64; {}.",
                self.domain, self.fix_action
            ),
        })
    }

    /// Add a byte count to a `u64` accumulator without wraparound.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when conversion or addition would overflow.
    pub fn add_bytes(self, total: &mut u64, bytes: usize, label: &str) -> Result<(), BackendError> {
        let bytes = u64::try_from(bytes).map_err(|_| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} {label} byte count exceeds u64; {}.",
                self.domain, self.fix_action
            ),
        })?;
        *total = total
            .checked_add(bytes)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} {label} byte accounting overflowed u64; {}.",
                    self.domain, self.fix_action
                ),
            })?;
        Ok(())
    }

    /// Add a `u64` counter value without wraparound.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_u64_counter(
        self,
        total: &mut u64,
        value: u64,
        label: &str,
        counter: &str,
    ) -> Result<(), BackendError> {
        *total = total
            .checked_add(value)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} {label} {counter} overflowed u64; {}.",
                    self.domain, self.fix_action
                ),
            })?;
        Ok(())
    }

    /// Add a `usize` counter value without wraparound.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_usize_counter(
        self,
        total: &mut usize,
        value: usize,
        label: &str,
        counter: &str,
    ) -> Result<(), BackendError> {
        *total = total
            .checked_add(value)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} {label} {counter} overflowed usize; {}.",
                    self.domain, self.fix_action
                ),
            })?;
        Ok(())
    }

    /// Add one transfer operation to a `u64` accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_operation(self, total: &mut u64, label: &str) -> Result<(), BackendError> {
        self.add_u64_counter(total, 1, label, "transfer operation accounting")
    }

    /// Add one copy to a `usize` accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_copy_count(self, total: &mut usize, label: &str) -> Result<(), BackendError> {
        self.add_usize_counter(total, 1, label, "copy counting")
    }

    /// Add copy slots to a `usize` accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_copy_slots(
        self,
        total: &mut usize,
        slots: usize,
        label: &str,
    ) -> Result<(), BackendError> {
        self.add_usize_counter(total, slots, label, "copy-slot accounting")
    }

    /// Multiply two capacity counts without wraparound.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when multiplication would overflow.
    pub fn mul_usize_capacity(
        self,
        lhs: usize,
        rhs: usize,
        label: &str,
    ) -> Result<usize, BackendError> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} {label} capacity overflowed usize for {lhs} x {rhs}; {}.",
                    self.domain, self.fix_action
                ),
            })
    }

    /// Add two capacity counts without wraparound.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when addition would overflow.
    pub fn add_usize_capacity(
        self,
        lhs: usize,
        rhs: usize,
        label: &str,
    ) -> Result<usize, BackendError> {
        lhs.checked_add(rhs)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {} {label} capacity overflowed usize for {lhs} + {rhs}; {}.",
                    self.domain, self.fix_action
                ),
            })
    }
}
