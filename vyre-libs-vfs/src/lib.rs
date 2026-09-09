//! Virtual filesystem DMA asynchronous block load and asset resolution compositions.

#[cfg(feature = "vfs")]
pub mod vfs;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
