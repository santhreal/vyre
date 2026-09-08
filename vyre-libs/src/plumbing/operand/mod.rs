//! What a composition's buffer arguments are: name, dtype, shape.
//!
//! Every Cat-A composition takes buffers by name. These modules are the only
//! place that decides what a name means, what element type it carries, how
//! many cells it addresses and what its zero is, so a dtype mismatch, a
//! colliding generic alias, an overflowing cell count and a wrong-width
//! accumulator seed are refused the same way by every op.

pub mod buffer_names;
pub(crate) mod element_zero;
pub(crate) mod tensor_ref;

#[cfg(any(feature = "graph", feature = "math-kernels"))]
pub(crate) mod shape;
