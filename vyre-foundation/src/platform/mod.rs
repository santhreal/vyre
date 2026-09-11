//! Platform Support Matrix, Fixed-Width Canonical Wire Types, and Platform Adapters.
//!
//! Provides source-derived host OS, architecture, endianness, pointer-width
//! and support-tier records; checked numeric conversions between host sizes
//! and fixed wire fields; and small typed platform adapters.

pub(crate) mod adapters;
pub(crate) mod matrix;
pub(crate) mod wire;

pub use adapters::{
    ClockAdapter, FileSystemAdapter, PlatformAdapterError, ScratchDir, ThreadAdapter,
};
pub use matrix::{
    Endianness, HostArch, HostCell, HostOs, HostSupportTier, PlatformSupportMatrix, PointerWidth,
    UnsupportedPlatformError, CANONICAL_RUST_VERSION, PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
};
pub use wire::{
    checked_i64_to_isize, checked_isize_to_i32, checked_isize_to_i64, checked_u64_to_usize,
    checked_usize_to_u32, checked_usize_to_u64, CanonicalU32, CanonicalU64, ConversionError,
};
