//! Compiler-internal numerical solvers, autotuning, and spectral schedule analysis.

pub mod solvers;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
