//! Dual CPU references for `primitive.arith.add`.

/// Operation ID for wrapping-add dual references.
pub const OP_ID: &str = "primitive.arith.add";

/// Direct word-oriented wrapping-add reference.
pub mod reference_a {
    /// Evaluate `left.wrapping_add(right)` over two little-endian u32 inputs.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::binary_direct(input, u32::wrapping_add)
    }
}

/// Independent bit-carry wrapping-add reference.
pub mod reference_b {
    /// Evaluate wrapping addition without using native integer addition.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::wrapping_add_bits_reference(input)
    }
}

/// Dual-reference marker for wrapping addition.
pub struct AddDualReference;

impl crate::dual::DualReference for AddDualReference {
    fn reference_a(input: &[u8]) -> Vec<u8> {
        reference_a::reference(input)
    }

    fn reference_b(input: &[u8]) -> Vec<u8> {
        reference_b::reference(input)
    }
}
