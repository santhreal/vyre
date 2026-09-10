//! WHY: a `BufferAccess::ReadOnly` declaration is what a backend compiles into
//! a non-coherent read-only load path, and a resident handle is `Copy`, so one
//! allocation can be bound to a read-only slot and to a writable slot in the
//! same dispatch. That combination makes the declaration false and the
//! compiled read stale. This suite pins the class: the refusal is symmetric in
//! slot order, it admits every pairing that falsifies no declaration, and it
//! keeps two owners' identical local ids apart.
//!
//! It does not catch aliasing produced below the handle, such as two distinct
//! resident allocations a backend later maps onto one device address range.

use vyre_driver::{ReadOnlyAliasCheck, ResidentHandle, ResidentOwner};
use vyre_foundation::ir::BufferAccess;

fn owner() -> ResidentOwner {
    ResidentOwner::new().expect("Fix: resident owner minting must succeed.")
}

fn two_allocations() -> (ResidentHandle, ResidentHandle) {
    let owner = owner();
    (owner.handle(1), owner.handle(2))
}

#[test]
fn distinct_allocations_at_opposite_directions_are_admitted() {
    let (table, out) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(table))
        .expect("Fix: a read-only slot on its own allocation must be admitted.");
    check
        .observe("out", BufferAccess::WriteOnly, Some(out))
        .expect("Fix: a writable slot on its own allocation must be admitted.");
}

#[test]
fn one_allocation_at_two_read_only_slots_is_admitted() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("left", BufferAccess::ReadOnly, Some(table))
        .expect("Fix: the first read-only slot must be admitted.");
    check
        .observe("right", BufferAccess::Uniform, Some(table))
        .expect("Fix: two read-only slots on one allocation write nothing and must be admitted.");
}

#[test]
fn one_allocation_at_two_writable_slots_is_admitted() {
    let (ring, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("ring", BufferAccess::ReadWrite, Some(ring))
        .expect("Fix: the first writable slot must be admitted.");
    check
        .observe("ring_tail", BufferAccess::WriteOnly, Some(ring))
        .expect("Fix: no read-only declaration is falsified by two writable slots.");
}

#[test]
fn a_read_only_slot_aliasing_a_later_writable_slot_is_refused() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(table))
        .expect("Fix: the read-only slot alone must be admitted.");
    let error = check
        .observe("scratch", BufferAccess::ReadWrite, Some(table))
        .expect_err("Fix: a writable slot on a read-only allocation must be refused.");
    let text = error.to_string();
    assert!(
        text.contains("read-only binding `table`") && text.contains("writable binding `scratch`"),
        "Fix: the refusal must name both bindings so the caller knows which pair to split. Got: {text}"
    );
}

#[test]
fn a_writable_slot_aliasing_a_later_read_only_slot_is_refused() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("scratch", BufferAccess::WriteOnly, Some(table))
        .expect("Fix: the writable slot alone must be admitted.");
    let error = check
        .observe("table", BufferAccess::ReadOnly, Some(table))
        .expect_err("Fix: slot order must not decide whether the pair is refused.");
    let text = error.to_string();
    assert!(
        text.contains("read-only binding `table`") && text.contains("writable binding `scratch`"),
        "Fix: the refusal must name the read-only side as read-only whichever slot came first. Got: {text}"
    );
}

#[test]
fn a_uniform_slot_aliasing_a_writable_slot_is_refused() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("params", BufferAccess::Uniform, Some(table))
        .expect("Fix: the uniform slot alone must be admitted.");
    check
        .observe("scratch", BufferAccess::WriteOnly, Some(table))
        .expect_err("Fix: Uniform lowers to the same read-only load path as ReadOnly.");
}

#[test]
fn a_workgroup_slot_never_conflicts() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(table))
        .expect("Fix: the read-only slot must be admitted.");
    check
        .observe("tile", BufferAccess::Workgroup, Some(table))
        .expect("Fix: workgroup memory is allocated per launch and names no caller allocation.");
}

#[test]
fn a_staged_slot_carries_no_allocation_and_never_conflicts() {
    let (table, _) = two_allocations();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(table))
        .expect("Fix: the read-only slot must be admitted.");
    check
        .observe("staged", BufferAccess::ReadWrite, None)
        .expect("Fix: a slot staged for this dispatch alone is reachable from one slot only.");
}

#[test]
fn two_owners_minting_the_same_local_id_are_distinct_allocations() {
    let first = owner();
    let second = owner();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(first.handle(1)))
        .expect("Fix: the read-only slot must be admitted.");
    check
        .observe("scratch", BufferAccess::ReadWrite, Some(second.handle(1)))
        .expect("Fix: local ids restart per owner, so id 1 on two owners is two allocations.");
}

#[test]
fn the_conflict_is_found_past_the_inline_capacity_of_the_scan() {
    let owner = owner();
    let mut check = ReadOnlyAliasCheck::new();
    check
        .observe("table", BufferAccess::ReadOnly, Some(owner.handle(1)))
        .expect("Fix: the read-only slot must be admitted.");
    for id in 2..=32u64 {
        check
            .observe("filler", BufferAccess::ReadWrite, Some(owner.handle(id)))
            .expect("Fix: unrelated writable slots must be admitted.");
    }
    check
        .observe("scratch", BufferAccess::WriteOnly, Some(owner.handle(1)))
        .expect_err("Fix: the scan must reach past its inline capacity to find the conflict.");
}
