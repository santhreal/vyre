//! Fixed-width canonical wire types and checked host size conversions (Row 118).

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error during checked numeric conversion between host usize/isize and fixed wire types.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ConversionError {
    /// Host value exceeded 32-bit integer capacity.
    #[error("host value {val} overflows 32-bit integer range")]
    Overflow32 {
        /// Value that overflowed.
        val: u128,
    },
    /// Host value was negative when unsigned was required.
    #[error("negative host value {val} cannot be converted to unsigned integer")]
    NegativeToUnsigned {
        /// Negative value.
        val: i128,
    },
    /// 64-bit integer overflows host usize on a 32-bit platform.
    #[error("64-bit value {val} overflows 32-bit host pointer width")]
    OverflowHostPointer {
        /// Value that overflowed host pointer capacity.
        val: u64,
    },
}

/// Checked conversion from host `usize` to fixed canonical `u32`.
#[inline]
pub fn checked_usize_to_u32(val: usize) -> Result<u32, ConversionError> {
    u32::try_from(val).map_err(|_| ConversionError::Overflow32 { val: val as u128 })
}

/// Checked conversion from host `usize` to fixed canonical `u64`.
#[inline]
pub const fn checked_usize_to_u64(val: usize) -> u64 {
    val as u64
}

/// Checked conversion from fixed `u64` to host `usize`.
#[inline]
pub fn checked_u64_to_usize(val: u64) -> Result<usize, ConversionError> {
    usize::try_from(val).map_err(|_| ConversionError::OverflowHostPointer { val })
}

/// Checked conversion from host `isize` to fixed canonical `i32`.
#[inline]
pub fn checked_isize_to_i32(val: isize) -> Result<i32, ConversionError> {
    i32::try_from(val).map_err(|_| ConversionError::Overflow32 {
        val: val.unsigned_abs() as u128,
    })
}

/// Checked conversion from host `isize` to fixed canonical `i64`.
#[inline]
pub const fn checked_isize_to_i64(val: isize) -> i64 {
    val as i64
}

/// Checked conversion from fixed `i64` to host `isize`.
#[inline]
pub fn checked_i64_to_isize(val: i64) -> Result<isize, ConversionError> {
    isize::try_from(val).map_err(|_| ConversionError::OverflowHostPointer {
        val: val.unsigned_abs(),
    })
}

/// Fixed canonical 32-bit unsigned integer with explicit little-endian byte layout.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[repr(transparent)]
pub struct CanonicalU32(pub u32);

impl CanonicalU32 {
    /// Create new canonical u32.
    pub const fn new(val: u32) -> Self {
        Self(val)
    }

    /// Return native u32.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Encode into canonical Little-Endian 4-byte buffer.
    pub const fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Decode from canonical Little-Endian 4-byte buffer.
    pub const fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }
}

impl fmt::Display for CanonicalU32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Fixed canonical 64-bit unsigned integer with explicit little-endian byte layout.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[repr(transparent)]
pub struct CanonicalU64(pub u64);

impl CanonicalU64 {
    /// Create new canonical u64.
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Return native u64.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Encode into canonical Little-Endian 8-byte buffer.
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Decode from canonical Little-Endian 8-byte buffer.
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }
}

impl fmt::Display for CanonicalU64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
