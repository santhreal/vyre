//! Dual CPU references for `primitive.arith.mul`.

/// Operation ID for wrapping-multiply dual references.
pub const OP_ID: &str = "primitive.arith.mul";

/// Direct word-oriented wrapping-multiply reference.
pub mod reference_a {
    /// Evaluate `left.wrapping_mul(right)` over two little-endian u32 inputs.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::binary_direct(input, u32::wrapping_mul)
    }
}

/// Independent shift-and-add wrapping-multiply reference.
pub mod reference_b {
    /// Evaluate wrapping multiplication without using native integer multiplication.
    #[must_use]
    pub fn reference(input: &[u8]) -> Vec<u8> {
        super::super::common::wrapping_mul_shift_add_reference(input)
    }
}

/// Dual-reference marker for wrapping multiplication.
pub struct MulDualReference;

impl crate::dual::DualReference for MulDualReference {
    fn reference_a(input: &[u8]) -> Vec<u8> {
        reference_a::reference(input)
    }

    fn reference_b(input: &[u8]) -> Vec<u8> {
        reference_b::reference(input)
    }
}
