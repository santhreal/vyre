macro_rules! define_binary_bitwise_dual {
    ($marker:ident, $op_id:literal, $word_op:expr, $bit_op:expr) => {
        /// Operation ID for this bitwise primitive.
        pub const OP_ID: &str = $op_id;

        /// Direct word-oriented reference.
        pub mod reference_a {
            /// Evaluate the direct word-oriented bitwise reference.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                $crate::dual_impls::evaluator::binary_direct(input, $word_op)
            }
        }

        /// Independent bit-by-bit reference.
        pub mod reference_b {
            /// Evaluate the bit-by-bit bitwise reference.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                super::super::bit_walk_reference::binary_bits(input, $bit_op)
            }
        }

        /// Dual-reference marker for this bitwise primitive.
        pub struct $marker;

        impl $crate::dual::DualReference for $marker {
            fn reference_a(input: &[u8]) -> Vec<u8> {
                reference_a::reference(input)
            }

            fn reference_b(input: &[u8]) -> Vec<u8> {
                reference_b::reference(input)
            }
        }

        inventory::submit! {
            $crate::DualReferenceFacet::new(OP_ID, reference_a::reference, reference_b::reference)
        }
    };
}

macro_rules! define_unary_bitwise_dual {
    ($marker:ident, $op_id:literal, $word_op:expr, $bit_op:expr) => {
        /// Operation ID for this bitwise primitive.
        pub const OP_ID: &str = $op_id;

        /// Direct word-oriented reference.
        pub mod reference_a {
            /// Evaluate the direct word-oriented bitwise reference.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                $crate::dual_impls::evaluator::unary_direct(input, $word_op)
            }
        }

        /// Independent bit-by-bit reference.
        pub mod reference_b {
            /// Evaluate the bit-by-bit bitwise reference.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                super::super::bit_walk_reference::unary_bits(input, $bit_op)
            }
        }

        /// Dual-reference marker for this bitwise primitive.
        pub struct $marker;

        impl $crate::dual::DualReference for $marker {
            fn reference_a(input: &[u8]) -> Vec<u8> {
                reference_a::reference(input)
            }

            fn reference_b(input: &[u8]) -> Vec<u8> {
                reference_b::reference(input)
            }
        }

        inventory::submit! {
            $crate::DualReferenceFacet::new(OP_ID, reference_a::reference, reference_b::reference)
        }
    };
}

#[path = "and/mod.rs"]
/// docs
pub mod and;
pub(crate) mod bit_walk_reference;
/// docs
pub mod clz;
#[path = "not/mod.rs"]
/// docs
pub mod not;
#[path = "or/mod.rs"]
/// docs
pub mod or;
/// docs
pub mod popcount;
/// docs
pub mod shift_left;
/// docs
pub mod shift_right;
#[path = "xor/mod.rs"]
/// docs
pub mod xor;
