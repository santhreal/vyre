//! Detection rule engine conditions, formulas, and program evaluation builders.

pub mod rule;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
