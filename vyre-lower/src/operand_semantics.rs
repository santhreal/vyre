//! Operand namespace semantics for lowered kernel ops.
//!
//! `KernelOp` operands are a compact `u32` vector, but not every entry is
//! a result-id. Some positions are binding slots, literal-pool indices,
//! child-body indices, axes, or metadata. Optimizer analyses and rewrites
//! must agree on this classifier or they will silently miscompile by
//! treating metadata as SSA values, or by missing real data dependencies.

use std::collections::BTreeMap;

use crate::{KernelBody, KernelOp, KernelOpKind};

/// True when `kind.operands[pos]` is a result-id reference in the lowered
/// kernel SSA namespace.
#[must_use]
pub(crate) fn operand_is_result_reference(kind: &KernelOpKind, pos: usize) -> bool {
    use KernelOpKind::*;
    match kind {
        Literal => false,
        LocalInvocationId | GlobalInvocationId | WorkgroupId => false,
        SubgroupLocalId | SubgroupSize | LoopIndex { .. } => false,
        LoopCarrierInit { .. } | LoopCarrier { .. } | LoopCarrierEnd { .. } => pos == 0,
        LoadGlobal | LoadShared | LoadConstant => pos != 0,
        BufferLength => false,
        StoreGlobal | StoreShared => pos != 0,
        Copy | BinOpKind(_) | UnOpKind(_) | Fma | MatrixMma { .. } | Select | Cast { .. } => true,
        Atomic { .. } => pos != 0,
        SubgroupBallot | SubgroupShuffle | SubgroupBroadcast | SubgroupReduce { .. } => true,
        StructuredIfThen | StructuredIfThenElse => pos == 0,
        StructuredForLoop { .. } => pos != 2,
        StructuredBlock | Region { .. } => false,
        Return | Barrier { .. } => false,
        AsyncLoad { .. } | AsyncStore { .. } => pos >= 2,
        AsyncWait { .. } => false,
        Trap { .. } => pos == 0,
        Resume { .. } => false,
        IndirectDispatch { .. } => false,
        Call { .. } => true,
        OpaqueExpr(..) | OpaqueNode(..) => true,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelOpKind;

    #[test]
    fn operand_classifier_separates_indices_from_result_ids() {
        assert!(!operand_is_result_reference(&KernelOpKind::Literal, 0));
        assert!(!operand_is_result_reference(&KernelOpKind::LoadGlobal, 0));
        assert!(operand_is_result_reference(&KernelOpKind::LoadGlobal, 1));
        assert!(!operand_is_result_reference(
            &KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            },
            2,
        ));
        assert!(operand_is_result_reference(
            &KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            },
            1,
        ));
        assert!(operand_is_result_reference(
            &KernelOpKind::AsyncStore { tag: "copy".into() },
            2,
        ));
        assert!(!operand_is_result_reference(
            &KernelOpKind::IndirectDispatch { count_offset: 0 },
            0,
        ));
    }
}
