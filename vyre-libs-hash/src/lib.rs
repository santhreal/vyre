//! Hash and checksum compositions including FNV-1a, CRC-32, Adler-32, and BLAKE3.

pub mod hash;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
