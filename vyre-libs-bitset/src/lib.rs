//! Packed u32 bitset operations, word utilities, and logical bitwise compositions.

pub mod bitset;
pub mod logical;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
