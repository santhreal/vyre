/*! Standalone primitive-operation CPU references. */
#![allow(missing_docs)]

macro_rules! define_dual_reference_impl {
    ($marker:ident, $eval_call:expr, $independent:path) => {
        /// Direct word-oriented reference over two little-endian u32 inputs.
        pub mod reference_a {
            /// Evaluate the operation using the direct word-oriented oracle.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                $eval_call(input)
            }
        }

        /// Independent reference over two little-endian u32 inputs.
        pub mod reference_b {
            /// Evaluate the operation using the independent oracle.
            #[must_use]
            pub fn reference(input: &[u8]) -> Vec<u8> {
                $independent(input)
            }
        }

        impl crate::dual::DualReference for $marker {
            fn reference_a(input: &[u8]) -> Vec<u8> {
                reference_a::reference(input)
            }

            fn reference_b(input: &[u8]) -> Vec<u8> {
                reference_b::reference(input)
            }
        }
    };
}
/// docs
pub mod arith;
/// docs
pub mod bitwise;
/// docs
pub mod compare;
/// docs
pub(crate) mod evaluator;
/// docs
pub(crate) mod hash;
mod indexed_reference_impls;
/// docs
pub(crate) mod memory;
mod scalar_reference_impls;
/// docs
pub(crate) mod scan;
/// docs
pub(crate) mod workgroup;
pub use evaluator::{EvalError, ReferenceEvaluator};
