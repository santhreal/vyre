//! Compiler-internal bitset, provenance, matroid, and fingerprint encoding compositions.

pub mod encoding;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
