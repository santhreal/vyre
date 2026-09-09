//! Compiler-internal scheduling, fusion, batching, and dispatch strategy compositions.

pub mod scheduling;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
