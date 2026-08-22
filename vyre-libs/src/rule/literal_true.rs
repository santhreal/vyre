//! The constant-true rule leaf.

use crate::rule::condition_op;
use vyre_spec::OperationContract;

condition_op::impl_literal_program!(LiteralTrue, OP_ID, 1);

/// Literal true condition operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiteralTrue;

/// Stable operation id for constant true leaves.
pub const OP_ID: &str = "vyre-libs::rule::literal_true";

/// Execution contract annotation for the standard catalog.
pub const CONTRACT: OperationContract = crate::contracts::RULE_PREDICATE_CHEAP;
