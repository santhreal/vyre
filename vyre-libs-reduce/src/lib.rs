//! Workgroup reduction trees, atomic scalar reductions, and prefix scans.

#[cfg(feature = "reduce")]
pub mod reduce;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
