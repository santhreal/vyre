//! Linear algebra, matrix operations, scans, broadcasting, algebra, and succinct data structures.

#[cfg(feature = "geom")]
pub mod geom;
pub mod math;
#[cfg(feature = "opt")]
pub mod opt;
#[cfg(feature = "representation")]
pub mod representation;

/// Expected output bytes for the canonical 2x2 u32 matmul fixture shared by
/// the `matmul_tiled` and `semiring_gemm` registrations.
///
/// A registration states the bytes a device must produce. Computing them from
/// words through the host packing routine makes the fixture agree with the
/// host encoder rather than with the operation, so a packing defect passes
/// conformance on both sides at once.
#[cfg(feature = "math-kernels")]
pub(crate) const MATMUL_2X2_EXPECTED_BYTES: [u8; 16] = [
    0x13, 0x00, 0x00, 0x00, // 19
    0x16, 0x00, 0x00, 0x00, // 22
    0x2b, 0x00, 0x00, 0x00, // 43
    0x32, 0x00, 0x00, 0x00, // 50
];

#[cfg(all(test, feature = "math-kernels"))]
mod expected_bytes_tests {
    use super::MATMUL_2X2_EXPECTED_BYTES;

    /// WHY: the constant above replaces a host packing call, so nothing else
    /// states what those bytes mean. This states it once, in test scope, where
    /// a wrong literal fails instead of being shipped as the answer a device
    /// is graded against.
    #[test]
    fn the_pinned_matmul_bytes_are_the_little_endian_product_words() {
        let constructed = vyre_test_support::test_parity_oracles::u32_bytes(&[19, 22, 43, 50]);
        assert_eq!(constructed, MATMUL_2X2_EXPECTED_BYTES);
    }
}

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
