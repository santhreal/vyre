//! Linear algebra, matrix operations, scans, broadcasting, algebra, and succinct data structures.

pub mod math;
pub mod geom;
pub mod opt;
pub mod representation;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
