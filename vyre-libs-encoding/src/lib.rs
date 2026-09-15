//! Compiler-internal bitset, provenance, matroid, and fingerprint encoding compositions.

pub mod encoding;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
