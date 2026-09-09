//! Compiler-internal static analysis, cost models, dataflow fixpoint, and diagnostics.

pub mod analysis;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
