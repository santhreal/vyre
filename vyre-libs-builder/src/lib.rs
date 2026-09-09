//! Shared IR composition infrastructure, child region skeletons, operands, and link anchors.

pub mod builder;
pub mod plumbing;
pub mod prelude;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    let _ = plumbing::registration::operation_catalog::link_anchor();
}
