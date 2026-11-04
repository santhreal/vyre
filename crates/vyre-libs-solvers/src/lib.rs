//! Compiler-internal numerical solvers, autotuning, and spectral schedule analysis.

pub mod solvers;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
