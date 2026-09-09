//! Compiler-internal logic, causal reasoning, categorical rewrites, and knowledge compilation.

pub mod reasoning;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
