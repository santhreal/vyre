//! SSA result-id renaming across a body tree.
//!
//! Owns the one sound way to rename result ids in a descriptor: rewrite
//! `op.result` and every operand position the operand namespace classifies as
//! an SSA reference, and leave every index and metadata operand alone so a
//! literal-pool index that happens to share a numeric value with a renamed id
//! is never corrupted.

use std::collections::BTreeMap;

use crate::operand_class::operand_is_result_reference;
use crate::{KernelBody, KernelOp};

/// Remap kernel SSA result ids throughout a body tree.
///
/// Every operand that [`operand_is_result_reference`] classifies as an SSA
/// reference, and every `op.result`, is looked up in `id_map` and replaced when
/// present (identity otherwise). Index/metadata operands (literal-pool, binding
/// slot, child-body index, axis) are left untouched, so this never corrupts a
/// non-SSA operand that happens to share a numeric value with a remapped id.
/// Recurses into all child bodies.
///
/// This is the one sound way to rename result ids across the descriptor. A
/// rewrite that replaces existing producers with a single op carrying fresh
/// result ids (e.g. collapsing an Fma chain into a `MatrixMma`) uses it to
/// re-point the old producers' consumers at the new ids.
#[must_use]
pub(crate) fn remap_body_result_ids(body: &KernelBody, id_map: &BTreeMap<u32, u32>) -> KernelBody {
    let new_ops: Vec<KernelOp> = body
        .ops
        .iter()
        .map(|op| {
            let new_operands: Vec<u32> = op
                .operands
                .iter()
                .enumerate()
                .map(|(pos, val)| {
                    if operand_is_result_reference(&op.kind, pos) {
                        *id_map.get(val).unwrap_or(val)
                    } else {
                        *val
                    }
                })
                .collect();
            KernelOp {
                kind: op.kind.clone(),
                operands: new_operands,
                result: op.result.map(|r| *id_map.get(&r).unwrap_or(&r)),
            }
        })
        .collect();
    let new_children: Vec<KernelBody> = body
        .child_bodies
        .iter()
        .map(|c| remap_body_result_ids(c, id_map))
        .collect();
    KernelBody {
        ops: new_ops,
        child_bodies: new_children,
        literals: body.literals.clone(),
    }
}

