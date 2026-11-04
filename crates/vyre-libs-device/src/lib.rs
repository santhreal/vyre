//! Compiler-internal device boundary contracts, memory ownership, resident graph
//! layout, and the virtual filesystem block loads that fill device arenas.

pub mod device;
#[cfg(feature = "vfs")]
pub mod vfs;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
