//! Compiler-internal static analysis, cost models, dataflow fixpoint, and diagnostics.

pub mod analysis;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
