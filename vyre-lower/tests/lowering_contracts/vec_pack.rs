//! Vector-pack analysis contracts.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::analyses::vec_pack::*;
use vyre_lower::descriptor_builder::{effect, lit, op};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOpKind, LiteralValue, MemoryClass,
};

fn input_slot(id: u32, name: &str) -> BindingSlot {
    BindingSlot {
        slot: id,
        element_type: DataType::U32,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadOnly,
        name: name.into(),
    }
}

/// Build a body with N consecutive LoadGlobal(slot, Literal(i)) ops.
fn linear_load_body(slot: u32, n: u32) -> KernelBody {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for i in 0..n {
        literals.push(LiteralValue::U32(i));
        ops.push(op(KernelOpKind::Literal, [i], i));
    }
    for i in 0..n {
        ops.push(op(KernelOpKind::LoadGlobal, [slot, i], n + i));
    }
    KernelBody {
        ops,
        child_bodies: vec![],
        literals,
    }
}

fn desc_with_body(body: KernelBody) -> KernelDescriptor {
    KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout {
            slots: vec![input_slot(0, "in")],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body,
    }
}

#[test]
fn empty_body_has_no_chains() {
    let desc = desc_with_body(KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![],
    });
    let report = analyze(&desc);
    assert!(!report.has_chains());
    assert_eq!(report.total_ops_eliminated, 0);
}

#[test]
fn single_load_is_not_a_chain() {
    let desc = desc_with_body(linear_load_body(0, 1));
    assert!(!analyze(&desc).has_chains());
}

#[test]
fn two_adjacent_loads_form_a_chain() {
    let desc = desc_with_body(linear_load_body(0, 2));
    let report = analyze(&desc);
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].op_indices.len(), 2);
    assert_eq!(report.chains[0].slot, 0);
    assert_eq!(report.chains[0].start_index, 0);
    assert_eq!(report.chains[0].pack_width(), 2);
    assert_eq!(report.total_ops_eliminated, 1);
}

#[test]
fn four_adjacent_loads_form_one_chain_at_pack_width_4() {
    let desc = desc_with_body(linear_load_body(0, 4));
    let report = analyze(&desc);
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].pack_width(), 4);
    assert_eq!(report.total_ops_eliminated, 3);
}

#[test]
fn five_adjacent_loads_pack_width_capped_at_4() {
    // 5 consecutive loads still form one chain of length 5;
    // pack_width caps at 4 (vec4 is the widest WGSL/PTX
    // primitive load). Total ops eliminated = 4 (5 → 1 wide
    // load gives 4 saved transactions).
    let desc = desc_with_body(linear_load_body(0, 5));
    let report = analyze(&desc);
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].op_indices.len(), 5);
    assert_eq!(report.chains[0].pack_width(), 4);
    assert_eq!(report.total_ops_eliminated, 4);
}

#[test]
fn loads_on_different_slots_form_separate_chains() {
    let mut body = linear_load_body(0, 3);
    // Append two more loads on slot 1 with consecutive indices.
    body.literals.push(LiteralValue::U32(0));
    body.literals.push(LiteralValue::U32(1));
    let lit_a = body.literals.len() as u32 - 2;
    let lit_b = body.literals.len() as u32 - 1;
    let result_a = body.ops.len() as u32 + 100;
    let result_b = result_a + 1;
    body.ops.push(op(KernelOpKind::Literal, [lit_a], result_a));
    body.ops.push(op(KernelOpKind::Literal, [lit_b], result_b));
    body.ops
        .push(op(KernelOpKind::LoadGlobal, [1, result_a], 200));
    body.ops
        .push(op(KernelOpKind::LoadGlobal, [1, result_b], 201));
    let mut desc = desc_with_body(body);
    desc.bindings.slots.push(input_slot(1, "in2"));
    let report = analyze(&desc);
    // Chain on slot 0 (length 3) + chain on slot 1 (length 2).
    assert_eq!(report.chains.len(), 2);
    assert_eq!(report.chains[0].slot, 0);
    assert_eq!(report.chains[0].op_indices.len(), 3);
    assert_eq!(report.chains[1].slot, 1);
    assert_eq!(report.chains[1].op_indices.len(), 2);
}

#[test]
fn non_consecutive_indices_break_the_chain() {
    // Loads at indices 0, 1, 3 (skip 2) → only 0,1 chain;
    // index 3 is a singleton.
    let mut body = KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(3),
        ],
    };
    for (i, _) in [0, 1, 3].iter().enumerate() {
        body.ops
            .push(op(KernelOpKind::Literal, [i as u32], i as u32));
    }
    for (offset, lit_id) in [0, 1, 2].iter().enumerate() {
        body.ops.push(op(
            KernelOpKind::LoadGlobal,
            [0, *lit_id as u32],
            10 + offset as u32,
        ));
    }
    let report = analyze(&desc_with_body(body));
    // Only 0, 1 form a chain (length 2). Index 3 is a singleton.
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].op_indices.len(), 2);
    assert_eq!(report.total_ops_eliminated, 1);
}

#[test]
fn dynamic_base_plus_adjacent_offsets_forms_chain() {
    let body = KernelBody {
        ops: vec![
            op(KernelOpKind::LocalInvocationId, [0], 0),
            lit(0, 1),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 2),
            lit(1, 3),
            lit(2, 4),
            lit(3, 5),
            lit(4, 6),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 7),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 4], 8),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 5], 9),
            op(KernelOpKind::BinOpKind(BinOp::Add), [2, 6], 10),
            op(KernelOpKind::LoadGlobal, [0, 7], 11),
            op(KernelOpKind::LoadGlobal, [0, 8], 12),
            op(KernelOpKind::LoadGlobal, [0, 9], 13),
            op(KernelOpKind::LoadGlobal, [0, 10], 14),
        ],
        child_bodies: vec![],
        literals: vec![
            LiteralValue::U32(4),
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
        ],
    };
    let report = analyze(&desc_with_body(body));
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].op_indices, vec![11, 12, 13, 14]);
    assert_eq!(report.chains[0].start_index, 0);
    assert_eq!(report.chains[0].pack_width(), 4);
    assert_eq!(report.total_ops_eliminated, 3);
}

#[test]
fn adjacent_offsets_from_different_dynamic_bases_do_not_chain() {
    let body = KernelBody {
        ops: vec![
            op(KernelOpKind::LocalInvocationId, [0], 0),
            lit(0, 1),
            lit(1, 2),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 3),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 2], 4),
            op(KernelOpKind::BinOpKind(BinOp::Add), [3, 1], 5),
            op(KernelOpKind::BinOpKind(BinOp::Add), [4, 2], 6),
            op(KernelOpKind::LoadGlobal, [0, 5], 7),
            op(KernelOpKind::LoadGlobal, [0, 6], 8),
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    };
    let report = analyze(&desc_with_body(body));
    assert!(!report.has_chains());
    assert_eq!(report.total_ops_eliminated, 0);
}

#[test]
fn singleton_computed_index_is_not_chainable() {
    let body = KernelBody {
        ops: vec![
            lit(0, 0),
            op(KernelOpKind::LocalInvocationId, [0], 1),
            op(
                KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                [0, 1],
                2,
            ),
            op(KernelOpKind::LoadGlobal, [0, 2], 3),
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(0)],
    };
    let report = analyze(&desc_with_body(body));
    assert!(!report.has_chains());
}

#[test]
fn chains_in_child_bodies_are_detected_too() {
    let child = linear_load_body(0, 3);
    let parent = KernelBody {
        ops: vec![effect(KernelOpKind::StructuredBlock, [0])],
        child_bodies: vec![child],
        literals: vec![],
    };
    let report = analyze(&desc_with_body(parent));
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].op_indices.len(), 3);
}
