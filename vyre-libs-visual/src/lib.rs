//! Visual rendering and compositing effects: blur, shadow, filters, gradients, glass.

#[cfg(feature = "visual")]
pub mod visual;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
