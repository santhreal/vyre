//! Result-id use queries over a body tree.
//!
//! Owns the question "is this result read anywhere below here", which the
//! scheduler and the vector fuser ask before moving or eliding a producer.
//! Only operand positions classified as SSA references count, so a binding
//! slot or literal-pool index that happens to equal the id is not a use.

use vyre_lower::operand_class::{classify_operand, OperandClass};
use vyre_lower::KernelBody;

pub(super) fn body_descendants_read_operand(body: &KernelBody, result_id: u32) -> bool {
    body.child_bodies
        .iter()
        .any(|child| body_reads_operand_recursive(child, result_id))
}

fn body_reads_operand_recursive(body: &KernelBody, result_id: u32) -> bool {
    body.ops.iter().any(|op| {
        op.operands.iter().enumerate().any(|(pos, &operand)| {
            operand == result_id && classify_operand(&op.kind, pos) == OperandClass::ResultRef
        })
    }) || body
        .child_bodies
        .iter()
        .any(|child| body_reads_operand_recursive(child, result_id))
}
