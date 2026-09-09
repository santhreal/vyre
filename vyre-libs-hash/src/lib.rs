//! Hash and checksum compositions including FNV-1a, CRC-32, Adler-32, and BLAKE3.

#[cfg(feature = "hash")]
pub mod hash;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
