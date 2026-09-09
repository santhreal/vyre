//! Deterministic fixpoint iteration kernels and grid synchronization barriers.

#[cfg(feature = "fixpoint")]
pub mod fixpoint;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
