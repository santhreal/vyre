//! Text processing, byte classification, UTF-8 validation, and line indexing.

#[cfg(feature = "text")]
pub mod text;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
