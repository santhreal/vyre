//! Bounded representation repair for verified descriptor emission.
//!
//! Semantic rewrites run on `Program` in `vyre-foundation`. This module only
//! orders pure descriptor producers before same-body consumers because the
//! linear emitters cannot consume forward SSA references. It performs one
//! structural walk and never iterates to an optimization fixed point.

use rustc_hash::FxHashMap;

use crate::operand_class::{classify_operand, OperandClass};
use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

/// Canonicalize representation constraints required by every emitter.
#[must_use]
pub fn canonicalize_for_emit(descriptor: &KernelDescriptor) -> KernelDescriptor {
    let mut output = descriptor.clone();
    schedule_body(&mut output.body);
    output
}

fn schedule_body(body: &mut KernelBody) {
    for child in &mut body.child_bodies {
        schedule_body(child);
    }
    let producer_indices = body
        .ops
        .iter()
        .enumerate()
        .filter_map(|(index, op)| op.result.map(|result| (result, index)))
        .collect::<FxHashMap<_, _>>();
    let old_ops = std::mem::take(&mut body.ops);
    let mut emitted = vec![false; old_ops.len()];
    let mut visiting = vec![false; old_ops.len()];
    let mut new_ops = Vec::with_capacity(old_ops.len());
    for op_index in 0..old_ops.len() {
        emit_with_dependencies(
            op_index,
            &old_ops,
            &producer_indices,
            &mut emitted,
            &mut visiting,
            &mut new_ops,
        );
    }
    body.ops = new_ops;
}

fn emit_with_dependencies(
    index: usize,
    old_ops: &[KernelOp],
    producer_indices: &FxHashMap<u32, usize>,
    emitted: &mut [bool],
    visiting: &mut [bool],
    new_ops: &mut Vec<KernelOp>,
) {
    if emitted[index] || visiting[index] {
        return;
    }
    visiting[index] = true;
    let op = &old_ops[index];
    for (operand_pos, operand) in op.operands.iter().copied().enumerate() {
        if classify_operand(&op.kind, operand_pos) != OperandClass::ResultRef {
            continue;
        }
        let Some(&producer_index) = producer_indices.get(&operand) else {
            continue;
        };
        if producer_index == index || emitted[producer_index] {
            continue;
        }
        if is_pure_movable(&old_ops[producer_index].kind) {
            emit_with_dependencies(
                producer_index,
                old_ops,
                producer_indices,
                emitted,
                visiting,
                new_ops,
            );
        }
    }
    visiting[index] = false;
    if !emitted[index] {
        emitted[index] = true;
        new_ops.push(op.clone());
    }
}

fn is_pure_movable(kind: &KernelOpKind) -> bool {
    matches!(
        kind,
        KernelOpKind::Literal
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::LoopIndex { .. }
            | KernelOpKind::BufferLength
            | KernelOpKind::Copy
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Fma
            | KernelOpKind::Select
            | KernelOpKind::Cast { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::lit;
    use crate::{BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOpKind};
    use vyre_foundation::ir::BinOp;

    #[test]
    fn canonicalize_orders_pure_producers_before_consumers() {
        let descriptor = KernelDescriptor {
            id: "forward_dep".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch {
                workgroup_size: [1, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 1),
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![2, 3],
                        result: Some(1),
                    },
                    lit(0, 2),
                    lit(0, 3),
                ],
                literals: vec![crate::LiteralValue::U32(42)],
                child_bodies: vec![],
            },
        };

        let output = canonicalize_for_emit(&descriptor);

        assert_eq!(output.body.ops[1].result, Some(2));
        assert_eq!(output.body.ops[2].result, Some(3));
        assert_eq!(canonicalize_for_emit(&output), output);
    }
}
