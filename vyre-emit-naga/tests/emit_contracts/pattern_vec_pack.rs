//! `vec_pack` pattern analysis contracts.

use vyre_foundation::ir::BinOp;
use vyre_lower::analyses::AccessKind;
use vyre_emit_naga::patterns::vec_pack::*;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, op};
use vyre_lower::{BindingSlot, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};

fn binding(slot: u32) -> BindingSlot {
    global_rw(slot, DataType::F32, &format!("buf{slot}"))
}

fn k(
    slots: Vec<BindingSlot>,
    ops: Vec<KernelOp>,
    literals: Vec<LiteralValue>,
) -> KernelDescriptor {
    descriptor("k")
        .slots(slots)
        .dispatch(64, 1, 1)
        .body(body().ops(ops).literals(literals))
        .build()
}

// ============== Positive truth (group detected) ==============

#[test]
fn positive_four_consecutive_loads_form_vec4_group() {
    // Setup: base = LocalInvocationId. Then load(buf, base+0..3).
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0), // base, id 0
            op(KernelOpKind::Literal, [0], 10),         // 0
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11), // base+0
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12), // 1
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13), // base+1
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
            op(KernelOpKind::Literal, [2], 14), // 2
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 14], 15), // base+2
            op(KernelOpKind::LoadGlobal, [0, 15], 22),
            op(KernelOpKind::Literal, [3], 16), // 3
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 16], 17), // base+3
            op(KernelOpKind::LoadGlobal, [0, 17], 23),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
        ],
    );
    let p = analyze(&kk);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec4);
    assert_eq!(p.groups[0].kind, AccessKind::Load);
    assert_eq!(p.groups[0].binding_slot, 0);
    assert_eq!(p.groups[0].op_count(), 4);
    assert_eq!(p.ops_eliminated(), 3);
}

#[test]
fn positive_two_consecutive_loads_form_vec2_group() {
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    );
    let p = analyze(&kk);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec2);
}

#[test]
fn positive_three_consecutive_stores_form_vec3_group() {
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::Literal, [3], 99),
            effect(KernelOpKind::StoreGlobal, [0, 11, 99]),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            effect(KernelOpKind::StoreGlobal, [0, 13, 99]),
            op(KernelOpKind::Literal, [2], 14),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 14], 15),
            effect(KernelOpKind::StoreGlobal, [0, 15, 99]),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(7),
        ],
    );
    let p = analyze(&kk);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec3);
    assert_eq!(p.groups[0].kind, AccessKind::Store);
}

// ============== Negative precision ==============

#[test]
fn negative_loads_with_non_consecutive_offsets_not_packed() {
    // Indices base+0 and base+2  -  gap of 1  -  not packable.
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(2)],
    );
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
}

#[test]
fn negative_loads_on_different_buffers_not_packed() {
    let kk = k(
        vec![binding(0), binding(1)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [1, 13], 21),
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    );
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
}

#[test]
fn negative_load_then_store_not_packed() {
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            effect(KernelOpKind::StoreGlobal, [0, 13, 20]),
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    );
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
}

#[test]
fn negative_no_global_accesses_yields_empty_plan() {
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 1),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 2),
        ],
        vec![LiteralValue::U32(7)],
    );
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
}

// ============== Adversarial ==============

#[test]
fn adversarial_five_consecutive_loads_pack_first_four_only() {
    // Vec4 is the max group size in phase 1. Five consecutive
    // loads should pack the first four and leave the fifth alone.
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
            op(KernelOpKind::Literal, [2], 14),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 14], 15),
            op(KernelOpKind::LoadGlobal, [0, 15], 22),
            op(KernelOpKind::Literal, [3], 16),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 16], 17),
            op(KernelOpKind::LoadGlobal, [0, 17], 23),
            op(KernelOpKind::Literal, [4], 18),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 18], 19),
            op(KernelOpKind::LoadGlobal, [0, 19], 24),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
            LiteralValue::U32(4),
        ],
    );
    let p = analyze(&kk);
    // First 4 pack as Vec4. The 5th load is a singleton.
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec4);
}

#[test]

fn adversarial_loads_with_compute_op_between_still_pack() {
    // load(buf, base+0); add(...); load(buf, base+1)
    // The intervening compute op is pure (consumes the loaded
    // value, doesn't touch the buffer)  -  this is exactly the
    // pattern a real lowered op produces. Two-phase analysis
    // (collect-then-group) treats them as adjacent accesses.
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::BinOpKind(BinOp::Add), [20, 20], 99), // pure compute, no buffer touch
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    );
    let p = analyze(&kk);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec2);
}

#[test]
fn adversarial_load_then_store_then_load_breaks_group_via_hazard() {
    // load(buf, base+0); store(buf, base+5, ...); load(buf, base+1)
    // The intervening Store to the same buffer creates a RAW hazard.
    // The two loads must NOT pack  -  phase-1 hazard barrier.
    let kk = k(
        vec![binding(0)],
        vec![
            op(KernelOpKind::LocalInvocationId, [], 0),
            op(KernelOpKind::Literal, [0], 10),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
            op(KernelOpKind::LoadGlobal, [0, 11], 20),
            op(KernelOpKind::Literal, [3], 98), // 5
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 98], 50), // base+5
            op(KernelOpKind::Literal, [0], 99), // value to store
            effect(KernelOpKind::StoreGlobal, [0, 50, 99]),
            op(KernelOpKind::Literal, [1], 12),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
            op(KernelOpKind::LoadGlobal, [0, 13], 21),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(7),
            LiteralValue::U32(5),
        ],
    );
    let p = analyze(&kk);
    // RAW hazard barrier prevents packing the two loads.
    let load_groups: Vec<_> = p
        .groups
        .iter()
        .filter(|g| g.kind == AccessKind::Load)
        .collect();
    assert!(load_groups.is_empty(), "RAW hazard must prevent grouping");
}

#[test]
fn adversarial_load_with_no_operands_skipped_safely() {
    let kk = k(
        vec![binding(0)],
        vec![effect(KernelOpKind::LoadGlobal, [])],
        vec![],
    );
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
}

#[test]
fn adversarial_load_inside_loop_body_packs_inner_group() {
    // Phase 1 walks structured-body children, so a 4-load group
    // inside a for-loop should pack.
    let kk = descriptor("k")
        .slot(binding(0))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    op(KernelOpKind::Literal, [0], 0),
                    op(KernelOpKind::Literal, [1], 1),
                    effect(
                        KernelOpKind::StructuredForLoop {
                            loop_var: "".into(),
                        },
                        [0, 1, 0],
                    ),
                ])
                .children([body()
                    .ops([
                        op(KernelOpKind::LocalInvocationId, [], 0),
                        op(KernelOpKind::Literal, [0], 10),
                        op(KernelOpKind::BinOpKind(BinOp::Add), [0, 10], 11),
                        op(KernelOpKind::LoadGlobal, [0, 11], 20),
                        op(KernelOpKind::Literal, [1], 12),
                        op(KernelOpKind::BinOpKind(BinOp::Add), [0, 12], 13),
                        op(KernelOpKind::LoadGlobal, [0, 13], 21),
                    ])
                    .literals([LiteralValue::U32(0), LiteralValue::U32(1)])])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let p = analyze(&kk);
    assert_eq!(p.groups.len(), 1);
    assert_eq!(p.groups[0].pack, PackKind::Vec2);
}

// ============== Aggregation ==============

#[test]
fn empty_kernel_yields_empty_plan() {
    let kk = k(vec![], vec![], vec![]);
    let p = analyze(&kk);
    assert!(p.groups.is_empty());
    assert_eq!(p.ops_eliminated(), 0);
    assert_eq!(p.kernel_id, "k");
}
