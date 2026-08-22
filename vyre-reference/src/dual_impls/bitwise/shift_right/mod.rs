/// Right-shift dual implementation reference.
mod reference;

/// Operation ID for right-shift dual references.
pub const OP_ID: &str = "primitive.bitwise.shift_right";

/// Direct word-oriented right-shift reference.
pub mod reference_a {
    /// Evaluate `left >> (right & 31)` over two little-endian u32 inputs.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        crate::dual_impls::evaluator::binary_direct(input, |left, right| left >> (right & 31))
    }
}

/// Independent bit-walk right-shift reference.
pub mod reference_b {
    /// Evaluate right shift without using the native shift operator on the full word.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::bit_walk_reference::shift_right_bits(input)
    }
}

inventory::submit! {
    crate::DualReferenceFacet::new(OP_ID, reference_a::reference, reference_b::reference)
}
