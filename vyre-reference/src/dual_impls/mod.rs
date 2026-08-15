//! Standalone primitive-operation CPU references.
#![allow(missing_docs)]

/// docs
pub mod arith;
/// docs
pub mod bitwise;
/// docs
pub mod compare;
/// docs
pub(crate) mod evaluator;
/// docs
pub mod hash;
mod indexed_reference_impls;
/// docs
pub mod memory;
mod scalar_reference_impls;
/// docs
pub mod scan;
/// docs
pub mod workgroup;
pub use evaluator::{EvalError, ReferenceEvaluator};
