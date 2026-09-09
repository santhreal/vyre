//! Platform Support Matrix, Fixed-Width Canonical Wire Types, and Platform Adapters (Row 118).
//!
//! Provides source-derived host OS, architecture, endianness, pointer-width,
//! and neutral device capability records; checked numeric conversions; and
//! small typed platform adapters.

pub mod adapters;
pub mod matrix;
pub mod wire;

pub use adapters::{ClockAdapter, FileSystemAdapter, PlatformAdapterError, ScratchDir, ThreadAdapter};
pub use matrix::{
    DeviceCapabilityProfile, Endianness, HostArch, HostCell, HostOs, PlatformSupportMatrix,
    PointerWidth, UnsupportedPlatformError, PLATFORM_SUPPORT_MATRIX_SCHEMA_VERSION,
};
pub use wire::{
    checked_i64_to_isize, checked_isize_to_i32, checked_isize_to_i64, checked_u64_to_usize,
    checked_usize_to_u32, checked_usize_to_u64, CanonicalU32, CanonicalU64, ConversionError,
};
