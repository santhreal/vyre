/// Population-count dual implementation reference.
pub mod reference {}

/// Operation ID for population-count dual references.
pub const OP_ID: &str = "primitive.bitwise.popcount";

/// Direct word-oriented population-count reference.
pub mod reference_a {
    /// Evaluate `count_ones` over one little-endian u32 input.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        crate::dual_impls::evaluator::unary_direct(input, u32::count_ones)
    }
}

/// Independent bit-walk population-count reference.
pub mod reference_b {
    /// Count bits by walking every lane explicitly.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::bit_walk_reference::popcount_bits(input)
    }
}

inventory::submit! {
    crate::DualReferenceFacet::new(OP_ID, reference_a::reference, reference_b::reference)
}
