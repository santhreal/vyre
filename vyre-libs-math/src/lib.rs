//! Linear algebra, matrix operations, scans, broadcasting, algebra, and succinct data structures.

#[cfg(feature = "geom")]
pub mod geom;
pub mod math;
#[cfg(feature = "opt")]
pub mod opt;
#[cfg(feature = "representation")]
pub mod representation;

/// Expected output words for the canonical 2x2 u32 matmul fixture shared by
/// the `matmul_tiled` and `semiring_gemm` registrations.
pub(crate) const MATMUL_2X2_EXPECTED_WORDS: [u32; 4] = [19, 22, 43, 50];

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
