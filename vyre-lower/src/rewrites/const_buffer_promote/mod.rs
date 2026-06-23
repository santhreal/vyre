//! Promote eligible read-only global buffers to constant bindings.
//!
//! The analysis in `analyses::const_buffer_promote` identifies small
//! fixed-size read-only global buffers with repeated loads. This rewrite makes
//! that decision material: it changes the binding memory class to `Constant`
//! and rewrites all matching `LoadGlobal` ops to `LoadConstant` across nested
//! bodies. Stores/atomics/async stores against a candidate slot veto the
//! promotion even if a malformed descriptor claims the binding is read-only.
//! Opaque escape hatches (`Call` / `OpaqueExpr` / `OpaqueNode`) carry
//! backend-defined effects with no addressable slot operand and could write any
//! global buffer, so their presence anywhere in the tree also vetoes promotion:
//! we fail closed rather than risk serving stale constant-cached data.

use crate::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};
use rustc_hash::FxHashSet;

/// Promote constant-buffer candidates using the default 64 KiB budget.
#[must_use]
pub fn const_buffer_promote(desc: &KernelDescriptor) -> KernelDescriptor {
    const_buffer_promote_with_budget(
        desc,
        crate::analyses::const_buffer_promote::DEFAULT_CONST_BUFFER_BUDGET_BYTES,
    )
}

/// Promote constant-buffer candidates using a caller-provided byte budget.
#[must_use]
pub fn const_buffer_promote_with_budget(
    desc: &KernelDescriptor,
    budget_bytes: u32,
) -> KernelDescriptor {
    let plan = crate::analyses::const_buffer_promote::analyze_with_budget(desc, budget_bytes);
    if plan.candidates.is_empty() {
        return desc.clone();
    }

    let candidates = plan
        .candidates
        .iter()
        .map(|candidate| candidate.binding_slot)
        .filter(|slot| !slot_may_be_written(&desc.body, *slot))
        .collect::<FxHashSet<_>>();
    if candidates.is_empty() {
        return desc.clone();
    }

    let mut out = desc.clone();
    let mut promoted_slots = FxHashSet::default();
    for binding in &mut out.bindings.slots {
        if candidates.contains(&binding.slot) {
            binding.memory_class = MemoryClass::Constant;
            promoted_slots.insert(binding.slot);
        }
    }
    if promoted_slots.is_empty() {
        return desc.clone();
    }
    rewrite_body_loads(&mut out.body, &promoted_slots);
    out
}

fn rewrite_body_loads(body: &mut KernelBody, slots: &FxHashSet<u32>) {
    for op in &mut body.ops {
        if matches!(op.kind, KernelOpKind::LoadGlobal)
            && op.operands.first().is_some_and(|slot| slots.contains(slot))
        {
            op.kind = KernelOpKind::LoadConstant;
        }
    }
    for child in &mut body.child_bodies {
        rewrite_body_loads(child, slots);
    }
}

/// True when the pass cannot prove `slot` stays read-only for the kernel's
/// lifetime, so promoting it to a constant buffer would be unsound.
///
/// An addressed store (`StoreGlobal` / `StoreShared` / `Atomic` / `AsyncStore`)
/// names its target slot in `operands[0]`, so we check it directly. Opaque
/// escape hatches (`Call` / `OpaqueExpr` / `OpaqueNode`) carry backend-defined
/// effects with NO addressable slot operand: they may write any global buffer,
/// including this candidate. Their effect is unknowable from the descriptor, so
/// we fail closed and treat the slot as possibly-written rather than promote it
/// and risk a backend serving stale constant-cached data after the opaque write.
fn slot_may_be_written(body: &KernelBody, slot: u32) -> bool {
    body.ops.iter().any(|op| match &op.kind {
        KernelOpKind::StoreGlobal
        | KernelOpKind::StoreShared
        | KernelOpKind::Atomic { .. }
        | KernelOpKind::AsyncStore { .. } => op.operands.first().copied() == Some(slot),
        KernelOpKind::Call { .. }
        | KernelOpKind::OpaqueExpr(..)
        | KernelOpKind::OpaqueNode(..) => true,
        _ => false,
    }) || body
        .child_bodies
        .iter()
        .any(|child| slot_may_be_written(child, slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelOp, LiteralValue,
        OpaqueNodeData,
    };
    use vyre_foundation::ir::DataType;

    fn op(kind: KernelOpKind, operands: Vec<u32>, result: Option<u32>) -> KernelOp {
        KernelOp {
            kind,
            operands,
            result,
        }
    }

    fn ro_global(slot: u32, count: u32) -> BindingSlot {
        BindingSlot {
            slot,
            element_type: DataType::F32,
            element_count: Some(count),
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadOnly,
            name: format!("ro{slot}"),
        }
    }

    fn kernel(ops: Vec<KernelOp>, child_bodies: Vec<KernelBody>) -> KernelDescriptor {
        KernelDescriptor {
            id: "const".into(),
            bindings: BindingLayout {
                slots: vec![ro_global(0, 16)],
            },
            dispatch: Dispatch::new(32, 1, 1),
            body: KernelBody {
                ops,
                child_bodies,
                literals: vec![LiteralValue::U32(0), LiteralValue::F32(1.0)],
            },
        }
    }

    #[test]
    fn promotes_repeated_read_only_global_loads() {
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
            ],
            vec![],
        );
        let output = const_buffer_promote(&input);

        assert_eq!(output.bindings.slots[0].memory_class, MemoryClass::Constant);
        assert!(matches!(
            output.body.ops[1].kind,
            KernelOpKind::LoadConstant
        ));
        assert!(matches!(
            output.body.ops[2].kind,
            KernelOpKind::LoadConstant
        ));
    }

    #[test]
    fn rewrites_nested_body_loads() {
        let child = KernelBody {
            ops: vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0)],
        };
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::StructuredBlock, vec![0], None),
            ],
            vec![child],
        );
        let output = const_buffer_promote(&input);

        assert_eq!(output.bindings.slots[0].memory_class, MemoryClass::Constant);
        assert!(matches!(
            output.body.child_bodies[0].ops[1].kind,
            KernelOpKind::LoadConstant
        ));
        assert!(matches!(
            output.body.child_bodies[0].ops[2].kind,
            KernelOpKind::LoadConstant
        ));
    }

    #[test]
    fn write_veto_keeps_descriptor_unchanged() {
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::Literal, vec![1], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(3)),
                op(KernelOpKind::StoreGlobal, vec![0, 0, 1], None),
            ],
            vec![],
        );
        let output = const_buffer_promote(&input);

        assert_eq!(output, input);
    }

    #[test]
    fn opaque_node_writer_vetoes_promotion() {
        // An OpaqueNode is a backend-defined escape hatch with no addressable
        // slot operand: it may write ANY global buffer, including this
        // read-only-DECLARED candidate. The pass cannot prove the slot stays
        // read-only, so it must fail closed and NOT promote it to Constant —
        // promoting would let a backend serve stale constant-cached data after
        // the opaque op writes the buffer.
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
                op(
                    KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
                        extension_kind: "backend-write".into(),
                        payload: Vec::new(),
                    })),
                    vec![],
                    None,
                ),
            ],
            vec![],
        );
        let output = const_buffer_promote(&input);

        assert_eq!(output.bindings.slots[0].memory_class, MemoryClass::Global);
        assert!(matches!(output.body.ops[1].kind, KernelOpKind::LoadGlobal));
        assert!(matches!(output.body.ops[2].kind, KernelOpKind::LoadGlobal));
        assert_eq!(output, input);
    }

    #[test]
    fn opaque_node_writer_in_child_body_vetoes_promotion() {
        // The unprovable-effect veto must be tree-wide: an opaque writer hiding
        // in a child body is just as dangerous as one in the entry body.
        let child = KernelBody {
            ops: vec![op(
                KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
                    extension_kind: "backend-write".into(),
                    payload: Vec::new(),
                })),
                vec![],
                None,
            )],
            child_bodies: vec![],
            literals: vec![],
        };
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
                op(KernelOpKind::StructuredBlock, vec![0], None),
            ],
            vec![child],
        );
        let output = const_buffer_promote(&input);

        assert_eq!(output.bindings.slots[0].memory_class, MemoryClass::Global);
        assert_eq!(output, input);
    }

    #[test]
    fn budget_veto_keeps_descriptor_unchanged() {
        let input = kernel(
            vec![
                op(KernelOpKind::Literal, vec![0], Some(0)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(1)),
                op(KernelOpKind::LoadGlobal, vec![0, 0], Some(2)),
            ],
            vec![],
        );
        let output = const_buffer_promote_with_budget(&input, 32);

        assert_eq!(output, input);
    }
}
