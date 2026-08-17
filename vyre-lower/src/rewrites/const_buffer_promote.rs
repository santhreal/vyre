//! Constant buffer promotion lowering rewrite.
//!
//! Promotes eligible read-only global bindings with multiple workgroup loads
//! to `MemoryClass::Constant` and rewrites `KernelOpKind::LoadGlobal` ops
//! on those slots to `KernelOpKind::LoadConstant`.
//!
//! Preserves Program semantics, values, and descriptor verification invariants.

use rustc_hash::FxHashSet;

use crate::analyses::const_buffer_promote::{
    analyze_with_budget, DEFAULT_CONST_BUFFER_BUDGET_BYTES,
};
use crate::{KernelBody, KernelDescriptor, KernelOpKind, MemoryClass};

/// Apply constant buffer promotion using the default budget (64 KiB).
#[must_use]
pub fn rewrite_const_buffer_promote(descriptor: &KernelDescriptor) -> KernelDescriptor {
    rewrite_const_buffer_promote_with_budget(descriptor, DEFAULT_CONST_BUFFER_BUDGET_BYTES)
}

/// Apply constant buffer promotion using an explicit byte budget.
#[must_use]
pub fn rewrite_const_buffer_promote_with_budget(
    descriptor: &KernelDescriptor,
    budget_bytes: u32,
) -> KernelDescriptor {
    let plan = analyze_with_budget(descriptor, budget_bytes);
    if plan.candidates.is_empty() {
        return descriptor.clone();
    }

    let candidate_slots: FxHashSet<u32> = plan
        .candidates
        .iter()
        .map(|candidate| candidate.binding_slot)
        .collect();

    let mut output = descriptor.clone();
    for slot in &mut output.bindings.slots {
        if candidate_slots.contains(&slot.slot) {
            slot.memory_class = MemoryClass::Constant;
        }
    }

    rewrite_body_loads(&mut output.body, &candidate_slots);
    output
}

fn rewrite_body_loads(body: &mut KernelBody, candidate_slots: &FxHashSet<u32>) {
    for op in &mut body.ops {
        if matches!(op.kind, KernelOpKind::LoadGlobal) {
            if let Some(&slot) = op.operands.first() {
                if candidate_slots.contains(&slot) {
                    op.kind = KernelOpKind::LoadConstant;
                }
            }
        }
    }

    for child in &mut body.child_bodies {
        rewrite_body_loads(child, candidate_slots);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::lit;
    use crate::{
        BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelOp, LiteralValue,
    };
    use vyre_foundation::ir::DataType;

    fn test_descriptor_with_loads(
        slots: Vec<BindingSlot>,
        load_slots: Vec<u32>,
    ) -> KernelDescriptor {
        let mut ops = vec![lit(0, 0)]; // Literal 0 at result 0
        let mut result_id = 1;
        for slot in load_slots {
            ops.push(KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![slot, 0],
                result: Some(result_id),
            });
            result_id += 1;
        }

        KernelDescriptor {
            id: "const_promote_test".into(),
            bindings: BindingLayout { slots },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops,
                literals: vec![LiteralValue::U32(0)],
                child_bodies: vec![],
            },
        }
    }

    #[test]
    fn eligible_readonly_buffer_with_multiple_loads_is_promoted() {
        let slot = BindingSlot {
            slot: 0,
            name: "readonly_lut".into(),
            element_type: DataType::F32,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadOnly,
            element_count: Some(256),
        };

        let desc = test_descriptor_with_loads(vec![slot], vec![0, 0, 0]);
        let rewritten = rewrite_const_buffer_promote(&desc);

        assert_eq!(
            rewritten.bindings.slots[0].memory_class,
            MemoryClass::Constant
        );
        for op in &rewritten.body.ops[1..] {
            assert_eq!(op.kind, KernelOpKind::LoadConstant);
        }
        assert!(crate::verify(&rewritten).is_ok());
    }

    #[test]
    fn single_load_buffer_is_not_promoted() {
        let slot = BindingSlot {
            slot: 0,
            name: "single_load".into(),
            element_type: DataType::F32,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadOnly,
            element_count: Some(256),
        };

        let desc = test_descriptor_with_loads(vec![slot], vec![0]);
        let rewritten = rewrite_const_buffer_promote(&desc);

        assert_eq!(rewritten.bindings.slots[0].memory_class, MemoryClass::Global);
        assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::LoadGlobal);
    }

    #[test]
    fn readwrite_buffer_is_not_promoted() {
        let slot = BindingSlot {
            slot: 0,
            name: "rw_buffer".into(),
            element_type: DataType::F32,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadWrite,
            element_count: Some(256),
        };

        let desc = test_descriptor_with_loads(vec![slot], vec![0, 0]);
        let rewritten = rewrite_const_buffer_promote(&desc);

        assert_eq!(rewritten.bindings.slots[0].memory_class, MemoryClass::Global);
        assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::LoadGlobal);
    }

    #[test]
    fn unbounded_buffer_is_not_promoted() {
        let slot = BindingSlot {
            slot: 0,
            name: "unbounded_buffer".into(),
            element_type: DataType::F32,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadOnly,
            element_count: None,
        };

        let desc = test_descriptor_with_loads(vec![slot], vec![0, 0]);
        let rewritten = rewrite_const_buffer_promote(&desc);

        assert_eq!(rewritten.bindings.slots[0].memory_class, MemoryClass::Global);
        assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::LoadGlobal);
    }
}
