use crate::rule::condition_op;

condition_op::impl_literal_program!(LiteralFalse, OP_ID, 0);

/// Literal false condition operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiteralFalse;

/// Stable operation id for constant false leaves.
pub const OP_ID: &str = "vyre-libs::rule::literal_false";
