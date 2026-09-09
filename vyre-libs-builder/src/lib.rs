//! Shared IR composition infrastructure, child region skeletons, operands, and link anchors.

pub mod builder;
pub mod plumbing;
pub mod prelude;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    plumbing::registration::operation_catalog::link_anchor()
}
