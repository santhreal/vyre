//! What the vec-packing analysis refuses to call fusable.
//!
//! A reported group promises that replacing its members with one packed access
//! at a single point in the op stream computes the same thing. Every case here
//! is a body where that promise is false and the reported group count is the
//! observable the audit publishes.
//!
//! The hazard classifier is an exhaustive match on `KernelOpKind` with no
//! catch-all arm, so a new op kind stops `vyre-emit-naga` from compiling until
//! someone states which side of the rule it is on. That closure is the
//! compiler's, not this file's; these cases pin the classification itself.

use vyre_emit_naga::patterns::vec_pack::{analyze, PackKind};
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, MemoryOrdering};
use vyre_lower::analyses::AccessKind;
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, effect, global_rw, lit, load_global, op, store_global,
    vector_load_global, vector_store_global,
};
use vyre_lower::{KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};

/// Result id of the base index every case packs around.
const BASE: u32 = 10;
/// Result id of `BASE + 1`.
const BASE_PLUS_ONE: u32 = 11;
/// Result id of the literal `1` the offset is built from.
const ONE: u32 = 12;
/// Result id of the value the store cases write.
const VALUE: u32 = 13;

/// A body whose first three ops define `BASE`, `BASE + 1`, and a store value,
/// followed by `tail`.
///
/// Every case shares this prologue so the only difference between a fusable
/// body and an unfusable one is what sits between the two accesses.
fn kernel(id: &str, tail: impl IntoIterator<Item = KernelOp>) -> KernelDescriptor {
    let mut ops = vec![
        op(KernelOpKind::GlobalInvocationId, Vec::<u32>::new(), BASE),
        lit(0, ONE),
        lit(1, VALUE),
        binop(BinOp::Add, BASE, ONE, BASE_PLUS_ONE),
    ];
    ops.extend(tail);
    descriptor(id)
        .slots([
            global_rw(0, DataType::U32, "subject"),
            global_rw(1, DataType::U32, "other"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(1), LiteralValue::U32(7)])
                .ops(ops),
        )
        .build()
}

/// A relaxed workgroup barrier, which carries no operands.
fn barrier() -> KernelOp {
    effect(
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::Relaxed,
        },
        Vec::<u32>::new(),
    )
}

/// A relaxed atomic add against `slot`.
fn atomic_add(slot: u32) -> KernelOp {
    effect(
        KernelOpKind::Atomic {
            op: AtomicOp::Add,
            ordering: MemoryOrdering::Relaxed,
        },
        [slot, BASE, VALUE],
    )
}

/// Two consecutive loads of one binding fuse into one `vec2` load.
///
/// The baseline every hazard case is measured against: without it a refusal
/// proves nothing, because a classifier that reports no group ever would pass
/// every other case in this file.
#[test]
fn adjacent_loads_of_one_binding_pack() {
    let plan = analyze(&kernel(
        "clean-load",
        [load_global(0, BASE, 20), load_global(0, BASE_PLUS_ONE, 21)],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
    assert_eq!(plan.groups[0].kind, AccessKind::Load);
    assert_eq!(plan.groups[0].pack, PackKind::Vec2);
    assert_eq!(plan.groups[0].binding_slot, 0);
}

/// Two consecutive stores to one binding fuse into one `vec2` store.
#[test]
fn adjacent_stores_to_one_binding_pack() {
    let plan = analyze(&kernel(
        "clean-store",
        [
            store_global(0, BASE, VALUE),
            store_global(0, BASE_PLUS_ONE, VALUE),
        ],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
    assert_eq!(plan.groups[0].kind, AccessKind::Store);
    assert_eq!(plan.groups[0].pack, PackKind::Vec2);
}

/// A barrier between two loads refuses the group.
///
/// Fusing would move the second load before the barrier, so it would read
/// memory the barrier exists to order.
#[test]
fn a_barrier_between_loads_refuses_the_group() {
    let plan = analyze(&kernel(
        "barrier-load",
        [
            load_global(0, BASE, 20),
            barrier(),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert!(
        plan.groups.is_empty(),
        "barrier crossed, got {:?}",
        plan.groups
    );
}

/// A barrier between two stores refuses the group.
#[test]
fn a_barrier_between_stores_refuses_the_group() {
    let plan = analyze(&kernel(
        "barrier-store",
        [
            store_global(0, BASE, VALUE),
            barrier(),
            store_global(0, BASE_PLUS_ONE, VALUE),
        ],
    ));
    assert!(
        plan.groups.is_empty(),
        "barrier crossed, got {:?}",
        plan.groups
    );
}

/// An atomic against the packed binding refuses the group.
#[test]
fn an_atomic_on_the_packed_binding_refuses_the_group() {
    let plan = analyze(&kernel(
        "atomic-same-slot",
        [
            load_global(0, BASE, 20),
            atomic_add(0),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert!(
        plan.groups.is_empty(),
        "atomic on slot 0 crossed, got {:?}",
        plan.groups
    );
}

/// An atomic against a different binding leaves the group alone.
///
/// The rule is per binding, not per memory op. A classifier that refused every
/// atomic would report no group on a body where fusion is legal, which costs
/// throughput on every kernel that touches two buffers.
#[test]
fn an_atomic_on_another_binding_keeps_the_group() {
    let plan = analyze(&kernel(
        "atomic-other-slot",
        [
            load_global(0, BASE, 20),
            atomic_add(1),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
    assert_eq!(plan.groups[0].binding_slot, 0);
}

/// A vector load of the packed binding leaves a load group alone.
///
/// Two reads of one buffer commute, so a load group crosses a read freely.
#[test]
fn a_vector_load_keeps_a_load_group() {
    let plan = analyze(&kernel(
        "vecload-between-loads",
        [
            load_global(0, BASE, 20),
            vector_load_global(0, BASE, 2, 30),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
    assert_eq!(plan.groups[0].kind, AccessKind::Load);
}

/// A vector load of the packed binding refuses a store group.
///
/// Fusing would move the second store before the read, so the read would
/// observe a value the original program writes after it.
#[test]
fn a_vector_load_refuses_a_store_group() {
    let plan = analyze(&kernel(
        "vecload-between-stores",
        [
            store_global(0, BASE, VALUE),
            vector_load_global(0, BASE, 2, 30),
            store_global(0, BASE_PLUS_ONE, VALUE),
        ],
    ));
    assert!(
        plan.groups.is_empty(),
        "read crossed by a store group, got {:?}",
        plan.groups
    );
}

/// A vector store of the packed binding refuses a load group.
#[test]
fn a_vector_store_refuses_a_load_group() {
    let plan = analyze(&kernel(
        "vecstore-between-loads",
        [
            load_global(0, BASE, 20),
            vector_store_global(0, BASE, 2, &[VALUE, VALUE]),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert!(
        plan.groups.is_empty(),
        "write crossed by a load group, got {:?}",
        plan.groups
    );
}

/// A vector store of another binding leaves a load group alone.
#[test]
fn a_vector_store_on_another_binding_keeps_a_load_group() {
    let plan = analyze(&kernel(
        "vecstore-other-slot",
        [
            load_global(0, BASE, 20),
            vector_store_global(1, BASE, 2, &[VALUE, VALUE]),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
}

/// Pure value production between two accesses leaves the group alone.
///
/// An arithmetic op has no memory effect and no ordering meaning, so a group
/// that refused to cross one would find nothing to pack in any real kernel:
/// the index arithmetic for the second access sits between the two accesses in
/// every lowered body.
#[test]
fn arithmetic_between_accesses_keeps_the_group() {
    let plan = analyze(&kernel(
        "arith-between",
        [
            load_global(0, BASE, 20),
            binop(BinOp::Add, 20, ONE, 40),
            load_global(0, BASE_PLUS_ONE, 21),
        ],
    ));
    assert_eq!(plan.groups.len(), 1, "one group, got {:?}", plan.groups);
    assert_eq!(plan.groups[0].start_op_index, 4);
    assert_eq!(plan.groups[0].end_op_index, 6);
}
