/// Equality-comparison dual implementation reference.
pub mod reference;

/// Operation ID for equality-comparison dual references.
pub const OP_ID: &str = "primitive.compare.eq";

/// Direct word-oriented equality reference.
pub mod reference_a {
    /// Evaluate `left == right` over two little-endian u32 inputs.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::binary_direct_predicate(input, |left, right| left == right)
    }
}

/// Independent byte-oriented equality reference.
pub mod reference_b {
    /// Evaluate equality by comparing the raw little-endian u32 byte payloads.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::eq_bytes(input)
    }
}

/// Dual-reference marker for equality comparison.
pub struct EqDualReference;

impl crate::dual::DualReference for EqDualReference {
    fn reference_a(input: &[u8]) -> Vec<u8> {
        reference_a::reference(input)
    }

    fn reference_b(input: &[u8]) -> Vec<u8> {
        reference_b::reference(input)
    }
}
