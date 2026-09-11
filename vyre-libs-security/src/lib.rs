//! Security taint analysis compositions, predicate evaluators, and label resolvers.

#[cfg(feature = "label")]
pub mod label;
#[cfg(feature = "predicate")]
pub mod predicate;
#[cfg(feature = "security")]
pub mod security;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
