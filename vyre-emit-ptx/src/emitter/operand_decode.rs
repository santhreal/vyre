//! Positional operand decoding for lowered kernel ops.
//!
//! Owns the reads that turn a `KernelOp`'s flat `Vec<u32>` operand list into
//! the named tuple an emission path expects, and the structured error when a
//! position is missing. It owns no operand classification: which positions
//! are SSA references is `vyre_lower::operand_class`.

use crate::EmitError;
use vyre_lower::KernelOp;

pub(super) fn read_two_operands(op: &KernelOp, op_name: &str) -> Result<(u32, u32), EmitError> {
    let a = *op
        .operands
        .first()
        .ok_or_else(|| EmitError::InvalidDescriptor(format!("{op_name} missing operand 0")))?;
    let b = *op
        .operands
        .get(1)
        .ok_or_else(|| EmitError::InvalidDescriptor(format!("{op_name} missing operand 1")))?;
    Ok((a, b))
}

pub(super) fn read_store_operands(op: &KernelOp) -> Result<(u32, u32, u32), EmitError> {
    let binding_slot = *op
        .operands
        .first()
        .ok_or_else(|| EmitError::InvalidDescriptor("StoreGlobal missing slot".into()))?;
    let index_op_id = *op
        .operands
        .get(1)
        .ok_or_else(|| EmitError::InvalidDescriptor("StoreGlobal missing index".into()))?;
    let value_op_id = *op
        .operands
        .get(2)
        .ok_or_else(|| EmitError::InvalidDescriptor("StoreGlobal missing value".into()))?;
    Ok((binding_slot, index_op_id, value_op_id))
}
