//! Security taint analysis compositions, predicate evaluators, and label resolvers.

pub mod security;
pub mod predicate;
pub mod label;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
