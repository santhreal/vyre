//! Compiler-internal device boundary contracts, memory ownership, and resident graph layout.

pub mod device;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
