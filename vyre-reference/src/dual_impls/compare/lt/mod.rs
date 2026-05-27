/// Less-than comparison dual implementation reference.
pub mod reference;

/// Operation ID for less-than-comparison dual references.
pub const OP_ID: &str = "primitive.compare.lt";

/// Direct word-oriented unsigned less-than reference.
pub mod reference_a {
    /// Evaluate `left < right` over two little-endian u32 inputs.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::binary_direct_predicate(input, |left, right| left < right)
    }
}

/// Independent byte-walk unsigned less-than reference.
pub mod reference_b {
    /// Evaluate unsigned little-endian less-than by walking most-significant bytes first.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::lt_bytes(input)
    }
}

/// Dual-reference marker for unsigned less-than comparison.
pub struct LtDualReference;

impl crate::dual::DualReference for LtDualReference {
    fn reference_a(input: &[u8]) -> Vec<u8> {
        reference_a::reference(input)
    }

    fn reference_b(input: &[u8]) -> Vec<u8> {
        reference_b::reference(input)
    }
}
