//! Text processing, byte classification, UTF-8 validation, and line indexing.

#[cfg(feature = "text")]
pub mod text;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
