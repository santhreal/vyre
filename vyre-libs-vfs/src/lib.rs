//! Virtual filesystem DMA asynchronous block load and asset resolution compositions.

#[cfg(feature = "vfs")]
pub mod vfs;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
