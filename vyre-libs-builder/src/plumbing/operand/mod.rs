pub mod buffer_names;
/// The one mapping from an element type to its additive zero. Private because
/// the module holds exactly that function and is named after it, so
/// `plumbing::operand::element_zero::element_zero` is a second path to the
/// crate root's `element_zero` and to nothing else.
pub(crate) mod element_zero;
pub mod shape;
pub mod tensor_ref;
