//! Security taint analysis compositions, predicate evaluators, and label resolvers.

#[cfg(feature = "label")]
pub mod label;
#[cfg(feature = "predicate")]
pub mod predicate;
#[cfg(feature = "security")]
pub mod security;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
