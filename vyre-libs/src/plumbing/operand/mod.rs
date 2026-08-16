//! What a composition's buffer arguments are: name, dtype, shape.
//!
//! Every Cat-A composition takes buffers by name. These three modules are the
//! only place that decides what a name means, what element type it carries and
//! how many cells it addresses, so a dtype mismatch, a colliding generic alias
//! and an overflowing cell count are refused the same way by every op.

pub mod buffer_names;
pub(crate) mod tensor_ref;

#[cfg(any(feature = "graph", feature = "math-kernels"))]
pub(crate) mod shape;
