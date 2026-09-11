//! Fixed-width canonical wire types and checked host size conversions.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Rejection of a conversion between a host size and a fixed wire field.
///
/// Every variant is reachable from a conversion below. A diagnostic nothing
/// constructs cannot fail on the thing it names, so none is kept for
/// symmetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ConversionError {
    /// The value does not fit the 32-bit wire field it was destined for.
    #[error(
        "host value {val} does not fit a 32-bit wire field. Fix: bound the count before encoding, or move the field to a 64-bit canonical integer and revise the wire schema version."
    )]
    Overflow32 {
        /// The value that did not fit.
        val: i128,
    },
    /// The value does not fit the 64-bit wire field it was destined for.
    #[error(
        "host value {val} does not fit a 64-bit wire field. Fix: bound the count before encoding; a host with a pointer wider than 64 bits cannot write this record without a wire schema revision."
    )]
    Overflow64 {
        /// The value that did not fit.
        val: i128,
    },
    /// The wire value exceeds what this host can address.
    #[error(
        "wire value {val} exceeds the addressable range of a {host_pointer_bits}-bit host. Fix: read this record on a 64-bit host, or reject the payload; truncating the value would index a different element."
    )]
    OverflowHostPointer {
        /// The value the wire carried.
        val: u64,
        /// Pointer width of the host that refused it.
        host_pointer_bits: u32,
    },
}

/// Checked conversion from host `usize` to fixed canonical `u32`.
///
/// # Errors
///
/// Returns [`ConversionError::Overflow32`] when the value does not fit.
#[inline]
pub fn checked_usize_to_u32(val: usize) -> Result<u32, ConversionError> {
    u32::try_from(val).map_err(|_| ConversionError::Overflow32 {
        val: i128::try_from(val).unwrap_or(i128::MAX),
    })
}

/// Checked conversion from host `usize` to fixed canonical `u64`.
///
/// Widening on every host this crate builds for, and still checked: a host
/// whose pointer is wider than 64 bits would truncate the value into the wire
/// field, and a truncated length is a different record under the same name.
///
/// # Errors
///
/// Returns [`ConversionError::Overflow64`] when the value does not fit.
#[inline]
pub fn checked_usize_to_u64(val: usize) -> Result<u64, ConversionError> {
    u64::try_from(val).map_err(|_| ConversionError::Overflow64 {
        val: i128::try_from(val).unwrap_or(i128::MAX),
    })
}

/// Checked conversion from fixed `u64` to host `usize`.
///
/// # Errors
///
/// Returns [`ConversionError::OverflowHostPointer`] when the wire value
/// exceeds what this host can address.
#[inline]
pub fn checked_u64_to_usize(val: u64) -> Result<usize, ConversionError> {
    usize::try_from(val).map_err(|_| ConversionError::OverflowHostPointer {
        val,
        host_pointer_bits: usize::BITS,
    })
}

/// Checked conversion from host `isize` to fixed canonical `i32`.
///
/// # Errors
///
/// Returns [`ConversionError::Overflow32`] when the value does not fit.
#[inline]
pub fn checked_isize_to_i32(val: isize) -> Result<i32, ConversionError> {
    i32::try_from(val).map_err(|_| ConversionError::Overflow32 {
        val: i128::from(val as i64),
    })
}

/// Checked conversion from host `isize` to fixed canonical `i64`.
///
/// Checked for the same reason as [`checked_usize_to_u64`].
///
/// # Errors
///
/// Returns [`ConversionError::Overflow64`] when the value does not fit.
#[inline]
pub fn checked_isize_to_i64(val: isize) -> Result<i64, ConversionError> {
    i64::try_from(val).map_err(|_| ConversionError::Overflow64 {
        val: i128::try_from(val).unwrap_or(i128::MAX),
    })
}

/// Checked conversion from fixed `i64` to host `isize`.
///
/// # Errors
///
/// Returns [`ConversionError::OverflowHostPointer`] when the wire value
/// exceeds what this host can address.
#[inline]
pub fn checked_i64_to_isize(val: i64) -> Result<isize, ConversionError> {
    isize::try_from(val).map_err(|_| ConversionError::OverflowHostPointer {
        val: val.unsigned_abs(),
        host_pointer_bits: usize::BITS,
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
