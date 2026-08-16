//! `vec_memory_fusion` pattern analysis contracts.

use vyre_foundation::ir::DataType;
use vyre_lower::BindingSlot;
use vyre_lower::KernelDescriptor;
use vyre_lower::KernelOp;
use vyre_lower::KernelOpKind;
use vyre_emit_ptx::patterns::vec_memory_fusion::*;
use vyre_foundation::ir::BinOp;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_ro, global_wo, lit, op};
use vyre_lower::{BindingVisibility, LiteralValue};

const KINDS: [MemoryFusionKind; 2] = [MemoryFusionKind::Load, MemoryFusionKind::Store];

fn opposite(kind: MemoryFusionKind) -> MemoryFusionKind {
    match kind {
        MemoryFusionKind::Load => MemoryFusionKind::Store,
        MemoryFusionKind::Store => MemoryFusionKind::Load,
    }
}

fn visibility(kind: MemoryFusionKind) -> BindingVisibility {
    match kind {
        MemoryFusionKind::Load => BindingVisibility::ReadOnly,
        MemoryFusionKind::Store => BindingVisibility::WriteOnly,
    }
}

fn binding(slot: u32, name: &str, visibility: BindingVisibility) -> BindingSlot {
    match visibility {
        BindingVisibility::ReadOnly => global_ro(slot, DataType::U32, name),
        _ => global_wo(slot, DataType::U32, name),
    }
}

fn add(lhs: u32, rhs: u32, result: u32) -> KernelOp {
    op(KernelOpKind::BinOpKind(BinOp::Add), [lhs, rhs], result)
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
        MemoryFusionKind::Load => op(KernelOpKind::LoadGlobal, [slot, index_id], result),
        MemoryFusionKind::Store => {
            effect(KernelOpKind::StoreGlobal, [slot, index_id, value_id])
        }
    }
}

fn kernel(
    slots: Vec<BindingSlot>,
    ops: Vec<KernelOp>,
    literals: Vec<LiteralValue>,
) -> KernelDescriptor {
    descriptor("k")
        .slots(slots)
        .body(body().ops(ops).literals(literals))
        .build()
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
    kernel(
        vec![binding(0, "buf", visibility(kind))],
        ops,
        vec![LiteralValue::U32(0), LiteralValue::U32(stride)],
    )
}

/// A v2 load chain starting at op 2 and a v2 store chain starting at
/// op 5, so asking for the wrong kind reports the wrong first-op index
/// instead of an empty plan.
fn mixed_load_and_store_chains() -> KernelDescriptor {
    kernel(
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

/// One kind must not report the other kind's chain, or `vec_load` and
/// `vec_store` on the audit report would carry the same findings.
#[test]
fn the_load_kind_reports_only_the_load_chain() {
    let plan = analyze(&mixed_load_and_store_chains(), MemoryFusionKind::Load);
    assert_eq!(plan.candidates.len(), 1);
    assert_eq!(plan.candidates[0].first_op_idx, 2);
    assert_eq!(plan.candidates[0].group_size, 2);
    assert_eq!(plan.candidates[0].binding_slot, 0);
    assert_eq!(plan.candidates[0].alignment_bytes, 8);
}

#[test]
fn the_store_kind_reports_only_the_store_chain() {
    let plan = analyze(&mixed_load_and_store_chains(), MemoryFusionKind::Store);
    assert_eq!(plan.candidates.len(), 1);
    assert_eq!(plan.candidates[0].first_op_idx, 5);
    assert_eq!(plan.candidates[0].group_size, 2);
    assert_eq!(plan.candidates[0].binding_slot, 1);
    assert_eq!(plan.candidates[0].alignment_bytes, 8);
}

fn only_candidate(
    desc: &KernelDescriptor,
    kind: MemoryFusionKind,
) -> Option<MemoryFusionCandidate> {
    let mut found = analyze(desc, kind).candidates;
    assert!(found.len() <= 1, "{kind:?}: expected at most one candidate");
    found.pop()
}

#[test]
fn empty_body_has_no_candidates() {
    for kind in KINDS {
        let desc = chain(kind, 0, 1);
        assert!(analyze(&desc, kind).candidates.is_empty(), "{kind:?}");
    }
}

#[test]
fn single_access_has_no_candidate() {
    for kind in KINDS {
        let desc = chain(kind, 1, 1);
        assert!(analyze(&desc, kind).candidates.is_empty(), "{kind:?}");
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
        assert!(analyze(&desc, kind).candidates.is_empty(), "{kind:?}");
    }
}

#[test]
fn accesses_to_different_slots_do_not_chain() {
    for kind in KINDS {
        let desc = kernel(
            vec![
                binding(0, "a", visibility(kind)),
                binding(1, "b", visibility(kind)),
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
        assert!(analyze(&desc, kind).candidates.is_empty(), "{kind:?}");
    }
}

#[test]
fn intervening_memory_effect_breaks_the_chain() {
    // Pure arithmetic may be scheduled into the gap; another memory
    // access may not be crossed.
    for kind in KINDS {
        let desc = kernel(
            vec![
                binding(0, "buf", visibility(kind)),
                binding(1, "other", visibility(opposite(kind))),
            ],
            vec![
                lit(0, 0),
                lit(1, 1),
                access(kind, 0, 0, 1, 2),
                access(opposite(kind), 1, 0, 1, 3),
                add(0, 1, 4),
                access(kind, 0, 4, 1, 5),
            ],
            vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        );
        assert!(analyze(&desc, kind).candidates.is_empty(), "{kind:?}");
    }
}

#[test]
fn folded_literal_indices_form_a_v4_candidate() {
    // Indices 0,1,2,3 arrive as separate literals rather than adds.
    for kind in KINDS {
        let desc = kernel(
            vec![binding(0, "buf", visibility(kind))],
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
    let desc = kernel(
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
    assert!(analyze(&desc, kind).candidates.is_empty());
}
