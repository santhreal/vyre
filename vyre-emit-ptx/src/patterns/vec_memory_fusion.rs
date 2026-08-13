//! Shared PTX vector memory fusion chain detector.

use crate::emitter::schedule::{is_schedulable_pure_op, is_scheduling_fence};
use crate::index_facts::IndexFacts;
use rustc_hash::FxHashMap;
use vyre_foundation::ir::DataType;
use vyre_lower::{BindingSlot, KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MemoryFusionKind {
    Load,
    Store,
}

impl MemoryFusionKind {
    fn matches(self, kind: &KernelOpKind) -> bool {
        match self {
            Self::Load => matches!(kind, KernelOpKind::LoadGlobal | KernelOpKind::LoadConstant),
            Self::Store => matches!(kind, KernelOpKind::StoreGlobal),
        }
    }

    fn slot_and_index(self, op: &KernelOp) -> Option<(u32, u32)> {
        let min_operands = match self {
            Self::Load => 2,
            Self::Store => 3,
        };
        if op.operands.len() < min_operands {
            return None;
        }
        Some((op.operands[0], op.operands[1]))
    }

    fn value_operand(self, op: &KernelOp) -> Option<u32> {
        match self {
            Self::Load => None,
            Self::Store => op.operands.get(2).copied(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MemoryFusionCandidate {
    pub(super) first_op_idx: usize,
    pub(super) group_size: u8,
    pub(super) binding_slot: u32,
    pub(super) element_type: DataType,
    pub(super) alignment_bytes: u32,
}

#[must_use]
pub(super) fn analyze_memory_fusion(
    desc: &KernelDescriptor,
    kind: MemoryFusionKind,
) -> Vec<MemoryFusionCandidate> {
    let binding_by_slot: FxHashMap<u32, &BindingSlot> = desc
        .bindings
        .slots
        .iter()
        .map(|binding| (binding.slot, binding))
        .collect();
    let mut candidates = Vec::new();
    walk(&desc.body, &binding_by_slot, kind, &mut candidates);
    candidates
}

fn walk(
    body: &KernelBody,
    binding_by_slot: &FxHashMap<u32, &BindingSlot>,
    kind: MemoryFusionKind,
    candidates: &mut Vec<MemoryFusionCandidate>,
) {
    let facts = IndexFacts::new(body);
    let mut i = 0;
    while i < body.ops.len() {
        let op = &body.ops[i];
        if !kind.matches(&op.kind) {
            i += 1;
            continue;
        }
        let Some((slot, base_idx_id)) = kind.slot_and_index(op) else {
            i += 1;
            continue;
        };
        let Some(binding) = binding_by_slot.get(&slot).copied() else {
            i += 1;
            continue;
        };

        let mut chain_len: u8 = 1;
        let mut prev_idx_id = base_idx_id;
        let mut j = i + 1;
        while j < body.ops.len() && chain_len < 4 {
            let gap_start = j;
            while j < body.ops.len() {
                let next = &body.ops[j];
                if kind.matches(&next.kind) {
                    break;
                }
                if is_scheduling_fence(next) || !is_schedulable_pure_op(next) {
                    break;
                }
                j += 1;
            }
            if j >= body.ops.len() {
                break;
            }
            let next = &body.ops[j];
            if !kind.matches(&next.kind) {
                break;
            }
            let Some((next_slot, next_idx_id)) = kind.slot_and_index(next) else {
                break;
            };
            if let Some(next_value_id) = kind.value_operand(next) {
                if body.ops[gap_start..j]
                    .iter()
                    .any(|gap_op| gap_op.result == Some(next_value_id))
                {
                    break;
                }
            }
            if next_slot != slot || !facts.is_index_plus_one(body, next_idx_id, prev_idx_id) {
                break;
            }
            chain_len += 1;
            prev_idx_id = next_idx_id;
            j += 1;
        }

        if chain_len >= 2 {
            let group_size = if chain_len >= 4 { 4 } else { 2 };
            let elem_size = binding.element_type.size_bytes().unwrap_or(0) as u32;
            candidates.push(MemoryFusionCandidate {
                first_op_idx: i,
                group_size,
                binding_slot: slot,
                element_type: binding.element_type.clone(),
                alignment_bytes: group_size as u32 * elem_size,
            });
            i += (group_size as usize) * 2 - 1;
        } else {
            i += 1;
        }
    }

    for child in &body.child_bodies {
        walk(child, binding_by_slot, kind, candidates);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use vyre_foundation::ir::BinOp;
    use vyre_lower::{BindingLayout, BindingVisibility, Dispatch, LiteralValue, MemoryClass};

    const KINDS: [MemoryFusionKind; 2] = [MemoryFusionKind::Load, MemoryFusionKind::Store];

    impl MemoryFusionKind {
        fn opposite(self) -> Self {
            match self {
                Self::Load => Self::Store,
                Self::Store => Self::Load,
            }
        }

        fn visibility(self) -> BindingVisibility {
            match self {
                Self::Load => BindingVisibility::ReadOnly,
                Self::Store => BindingVisibility::WriteOnly,
            }
        }
    }

    fn binding(slot: u32, name: &str, visibility: BindingVisibility) -> BindingSlot {
        BindingSlot {
            slot,
            element_type: DataType::U32,
            element_count: None,
            memory_class: MemoryClass::Global,
            visibility,
            name: name.into(),
        }
    }

    fn lit(literal_index: u32, result: u32) -> KernelOp {
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![literal_index],
            result: Some(result),
        }
    }

    fn add(lhs: u32, rhs: u32, result: u32) -> KernelOp {
        KernelOp {
            kind: KernelOpKind::BinOpKind(BinOp::Add),
            operands: vec![lhs, rhs],
            result: Some(result),
        }
    }

    /// One access of `kind`: a load reads `slot[index_id]`, a store
    /// writes `value_id` into it.
    fn access(
        kind: MemoryFusionKind,
        slot: u32,
        index_id: u32,
        value_id: u32,
        result: u32,
    ) -> KernelOp {
        match kind {
            MemoryFusionKind::Load => KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![slot, index_id],
                result: Some(result),
            },
            MemoryFusionKind::Store => KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![slot, index_id, value_id],
                result: None,
            },
        }
    }

    fn descriptor(
        slots: Vec<BindingSlot>,
        ops: Vec<KernelOp>,
        literals: Vec<LiteralValue>,
    ) -> KernelDescriptor {
        KernelDescriptor {
            id: "k".into(),
            bindings: BindingLayout { slots },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops,
                child_bodies: vec![],
                literals,
            },
        }
    }

    /// `count` accesses on slot 0, each index the previous plus `stride`.
    /// Op 0 is the base index literal, op 1 the stride literal, so the
    /// first access is always op 2.
    fn chain(kind: MemoryFusionKind, count: usize, stride: u32) -> KernelDescriptor {
        let mut ops = vec![lit(0, 0), lit(1, 1)];
        let mut next_id = 2;
        let mut index_id = 0;
        for position in 0..count {
            if position > 0 {
                ops.push(add(index_id, 1, next_id));
                index_id = next_id;
                next_id += 1;
            }
            ops.push(access(kind, 0, index_id, 1, next_id));
            next_id += 1;
        }
        descriptor(
            vec![binding(0, "buf", kind.visibility())],
            ops,
            vec![LiteralValue::U32(0), LiteralValue::U32(stride)],
        )
    }

    /// A v2 load chain starting at op 2 and a v2 store chain starting at
    /// op 5, so a facade that asks for the wrong kind reports the wrong
    /// first-op index instead of an empty plan.
    pub(in crate::patterns) fn mixed_load_and_store_chains() -> KernelDescriptor {
        descriptor(
            vec![
                binding(0, "in", BindingVisibility::ReadOnly),
                binding(1, "out", BindingVisibility::WriteOnly),
            ],
            vec![
                lit(0, 0),
                lit(1, 1),
                access(MemoryFusionKind::Load, 0, 0, 1, 2),
                add(0, 1, 3),
                access(MemoryFusionKind::Load, 0, 3, 1, 4),
                access(MemoryFusionKind::Store, 1, 0, 2, 0),
                access(MemoryFusionKind::Store, 1, 3, 4, 0),
            ],
            vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        )
    }

    fn only_candidate(
        desc: &KernelDescriptor,
        kind: MemoryFusionKind,
    ) -> Option<MemoryFusionCandidate> {
        let mut found = analyze_memory_fusion(desc, kind);
        assert!(found.len() <= 1, "{kind:?}: expected at most one candidate");
        found.pop()
    }

    #[test]
    fn empty_body_has_no_candidates() {
        for kind in KINDS {
            let desc = chain(kind, 0, 1);
            assert!(analyze_memory_fusion(&desc, kind).is_empty(), "{kind:?}");
        }
    }

    #[test]
    fn single_access_has_no_candidate() {
        for kind in KINDS {
            let desc = chain(kind, 1, 1);
            assert!(analyze_memory_fusion(&desc, kind).is_empty(), "{kind:?}");
        }
    }

    #[test]
    fn two_unit_stride_accesses_form_a_v2_candidate() {
        for kind in KINDS {
            let desc = chain(kind, 2, 1);
            let candidate = only_candidate(&desc, kind).unwrap_or_else(|| panic!("{kind:?}"));
            assert_eq!(candidate.first_op_idx, 2, "{kind:?}");
            assert_eq!(candidate.group_size, 2, "{kind:?}");
            assert_eq!(candidate.binding_slot, 0, "{kind:?}");
            assert_eq!(candidate.element_type, DataType::U32, "{kind:?}");
            assert_eq!(candidate.alignment_bytes, 8, "{kind:?}");
        }
    }

    #[test]
    fn four_unit_stride_accesses_form_a_v4_candidate() {
        for kind in KINDS {
            let desc = chain(kind, 4, 1);
            let candidate = only_candidate(&desc, kind).unwrap_or_else(|| panic!("{kind:?}"));
            assert_eq!(candidate.first_op_idx, 2, "{kind:?}");
            assert_eq!(candidate.group_size, 4, "{kind:?}");
            assert_eq!(candidate.alignment_bytes, 16, "{kind:?}");
        }
    }

    #[test]
    fn three_accesses_yield_only_a_v2_candidate() {
        // PTX has no v3, so the third access stays scalar.
        for kind in KINDS {
            let desc = chain(kind, 3, 1);
            let candidate = only_candidate(&desc, kind).unwrap_or_else(|| panic!("{kind:?}"));
            assert_eq!(candidate.group_size, 2, "{kind:?}");
        }
    }

    #[test]
    fn non_unit_stride_does_not_chain() {
        for kind in KINDS {
            let desc = chain(kind, 2, 2);
            assert!(analyze_memory_fusion(&desc, kind).is_empty(), "{kind:?}");
        }
    }

    #[test]
    fn accesses_to_different_slots_do_not_chain() {
        for kind in KINDS {
            let desc = descriptor(
                vec![
                    binding(0, "a", kind.visibility()),
                    binding(1, "b", kind.visibility()),
                ],
                vec![
                    lit(0, 0),
                    lit(1, 1),
                    access(kind, 0, 0, 1, 2),
                    add(0, 1, 3),
                    access(kind, 1, 3, 1, 4),
                ],
                vec![LiteralValue::U32(0), LiteralValue::U32(1)],
            );
            assert!(analyze_memory_fusion(&desc, kind).is_empty(), "{kind:?}");
        }
    }

    #[test]
    fn intervening_memory_effect_breaks_the_chain() {
        // Pure arithmetic may be scheduled into the gap; another memory
        // access may not be crossed.
        for kind in KINDS {
            let desc = descriptor(
                vec![
                    binding(0, "buf", kind.visibility()),
                    binding(1, "other", kind.opposite().visibility()),
                ],
                vec![
                    lit(0, 0),
                    lit(1, 1),
                    access(kind, 0, 0, 1, 2),
                    access(kind.opposite(), 1, 0, 1, 3),
                    add(0, 1, 4),
                    access(kind, 0, 4, 1, 5),
                ],
                vec![LiteralValue::U32(0), LiteralValue::U32(1)],
            );
            assert!(analyze_memory_fusion(&desc, kind).is_empty(), "{kind:?}");
        }
    }

    #[test]
    fn folded_literal_indices_form_a_v4_candidate() {
        // Indices 0,1,2,3 arrive as separate literals rather than adds.
        for kind in KINDS {
            let desc = descriptor(
                vec![binding(0, "buf", kind.visibility())],
                vec![
                    lit(0, 0),
                    lit(1, 1),
                    access(kind, 0, 0, 1, 2),
                    lit(2, 3),
                    access(kind, 0, 3, 1, 4),
                    lit(3, 5),
                    access(kind, 0, 5, 1, 6),
                    lit(4, 7),
                    access(kind, 0, 7, 1, 8),
                ],
                vec![
                    LiteralValue::U32(0),
                    LiteralValue::U32(100),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                ],
            );
            let candidate = only_candidate(&desc, kind).unwrap_or_else(|| panic!("{kind:?}"));
            assert_eq!(candidate.first_op_idx, 2, "{kind:?}");
            assert_eq!(candidate.group_size, 4, "{kind:?}");
            assert_eq!(candidate.alignment_bytes, 16, "{kind:?}");
        }
    }

    #[test]
    fn store_value_produced_in_the_gap_breaks_the_chain() {
        // Store-only: the fused value registers must already be live at
        // the first store. A load has no value operand to constrain.
        let kind = MemoryFusionKind::Store;
        let desc = descriptor(
            vec![binding(0, "out", BindingVisibility::WriteOnly)],
            vec![
                lit(0, 0),
                lit(1, 1),
                access(kind, 0, 0, 1, 0),
                lit(2, 2),
                lit(3, 3),
                access(kind, 0, 2, 3, 0),
            ],
            vec![
                LiteralValue::U32(0),
                LiteralValue::U32(10),
                LiteralValue::U32(1),
                LiteralValue::U32(11),
            ],
        );
        assert!(analyze_memory_fusion(&desc, kind).is_empty());
    }
}
