//! What the physical storage layout promises a target.
//!
//! The layout is the only statement of how many bytes one workgroup allocates
//! and where each region sits inside that allocation. Two properties carry the
//! weight: a region's lifetime covers every op that can still read it, and two
//! regions share bytes only when their lifetimes are disjoint. A narrower
//! lifetime is not a smaller pool, it is a miscompile, so the cases below pin
//! the loop and nesting spans as hard as they pin the offsets.

use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, for_loop, global_rw, if_then, lit, op, shared_rw, slot, SlotCount,
};
use vyre_lower::{
    verify, BindingVisibility, FragmentValue, KernelOp, KernelOpKind, LiteralValue,
    MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec, MatrixTileShape, MemoryClass, StorageLayout,
    StorageLayoutError, StorageLifetime, VerifyErrorKind, WORKGROUP_SLOT_BASE,
};

/// First workgroup-class slot a hand-built descriptor may use.
const A: u32 = WORKGROUP_SLOT_BASE;
/// Second workgroup-class slot.
const B: u32 = WORKGROUP_SLOT_BASE + 1;
/// Third workgroup-class slot.
const C: u32 = WORKGROUP_SLOT_BASE + 2;

/// Sixteen u32 elements: 64 bytes, 4-byte alignment.
fn tile(index: u32, name: &str) -> vyre_lower::BindingSlot {
    shared_rw(index, DataType::U32, 16, name)
}

fn scratch(index: u32, name: &str) -> vyre_lower::BindingSlot {
    slot(
        index,
        DataType::U32,
        MemoryClass::Scratch,
        BindingVisibility::ReadWrite,
        name,
    )
    .with_count(16)
}

fn load_shared(binding: u32, index: u32, result: u32) -> vyre_lower::KernelOp {
    op(KernelOpKind::LoadShared, [binding, index], result)
}

#[test]
fn two_regions_that_are_never_live_together_share_bytes() {
    let descriptor = descriptor("disjoint")
        .slots([tile(A, "first"), tile(B, "second")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .op(load_shared(B, 0, 2))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    assert_eq!(layout.region(A).expect("region A").offset, 0);
    assert_eq!(layout.region(B).expect("region B").offset, 0);
    assert_eq!(layout.shared_pool_bytes, 64);
    assert_eq!(layout.distinct_bytes, 128);
    assert_eq!(layout.overlaid_bytes(), 64);
    assert_eq!(
        layout.region(A).expect("region A").lifetime,
        StorageLifetime::at(1)
    );
    assert_eq!(
        layout.region(B).expect("region B").lifetime,
        StorageLifetime::at(2)
    );
}

#[test]
fn a_region_read_again_later_keeps_its_bytes_to_itself() {
    let descriptor = descriptor("overlapping")
        .slots([tile(A, "first"), tile(B, "second")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .op(load_shared(B, 0, 2))
                .op(load_shared(A, 0, 3))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    assert_eq!(
        layout.region(A).expect("region A").lifetime,
        StorageLifetime {
            first_op: 1,
            last_op: 3
        }
    );
    assert_eq!(layout.region(A).expect("region A").offset, 0);
    assert_eq!(layout.region(B).expect("region B").offset, 64);
    assert_eq!(layout.shared_pool_bytes, 128);
    assert_eq!(layout.overlaid_bytes(), 0);
}

#[test]
fn regions_touched_inside_one_loop_body_stay_live_across_the_whole_loop() {
    // `for i in 0..n { read B; read C }` then `read A`. B and C are read on
    // every iteration, so overlaying them would let the second read of B land
    // on bytes C still holds. A is read after the loop retires and may reuse
    // either.
    let descriptor = descriptor("loop_carried")
        .slots([tile(A, "after"), tile(B, "left"), tile(C, "right")])
        .body(
            body()
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(for_loop("i", 0, 1, 0))
                .op(load_shared(A, 0, 2))
                .child(
                    body()
                        .op(load_shared(B, 0, 10))
                        .op(load_shared(C, 0, 11)),
                )
                .literal(LiteralValue::U32(0))
                .literal(LiteralValue::U32(8)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    let b = layout.region(B).expect("region B");
    let c = layout.region(C).expect("region C");
    assert_eq!(
        b.lifetime,
        StorageLifetime {
            first_op: 2,
            last_op: 4
        }
    );
    assert_eq!(c.lifetime, b.lifetime);
    assert!(
        !b.shares_bytes(c),
        "two regions read on every iteration must not share bytes: {b:?} against {c:?}"
    );
    let a = layout.region(A).expect("region A");
    assert_eq!(a.lifetime, StorageLifetime::at(5));
    assert_eq!(a.offset, 0, "a region live after the loop reuses its bytes");
    assert_eq!(layout.shared_pool_bytes, 128);
}

#[test]
fn a_region_touched_in_a_nested_arm_is_live_across_the_branch() {
    let descriptor = descriptor("nested_arm")
        .slots([tile(A, "guarded"), tile(B, "after")])
        .body(
            body()
                .op(lit(0, 0))
                .op(if_then(0, 0))
                .op(load_shared(B, 0, 2))
                .child(body().op(load_shared(A, 0, 10)))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    assert_eq!(
        layout.region(A).expect("region A").lifetime,
        StorageLifetime {
            first_op: 1,
            last_op: 2
        }
    );
    assert_eq!(
        layout.region(B).expect("region B").lifetime,
        StorageLifetime::at(3)
    );
    assert_eq!(layout.shared_pool_bytes, 64);
}

#[test]
fn a_declared_region_nothing_reads_holds_its_bytes_for_the_whole_kernel() {
    let descriptor = descriptor("unread")
        .slots([tile(A, "unread"), tile(B, "read")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(B, 0, 1))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    assert_eq!(
        layout.region(A).expect("region A").lifetime,
        StorageLifetime {
            first_op: 0,
            last_op: 1
        }
    );
    assert_eq!(layout.shared_pool_bytes, 128);
    assert_eq!(layout.overlaid_bytes(), 0);
}

#[test]
fn workgroup_and_invocation_private_storage_are_separate_pools() {
    let descriptor = descriptor("two_classes")
        .slots([tile(A, "shared"), scratch(B, "private")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .op(load_shared(B, 0, 2))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the overlay");

    assert_eq!(layout.region(A).expect("region A").offset, 0);
    assert_eq!(layout.region(B).expect("region B").offset, 0);
    assert_eq!(layout.shared_pool_bytes, 64);
    assert_eq!(layout.private_pool_bytes, 64);
    assert_eq!(layout.distinct_bytes, 128);
}

#[test]
fn a_workgroup_binding_with_no_element_count_is_rejected() {
    let descriptor = descriptor("unsized")
        .slot(slot(
            A,
            DataType::U32,
            MemoryClass::Shared,
            BindingVisibility::ReadWrite,
            "runtime_sized",
        ))
        .build();

    assert_eq!(
        StorageLayout::plan(&descriptor),
        Err(StorageLayoutError::UnsizedRegion { slot: A })
    );
}

#[test]
fn a_global_binding_is_not_workgroup_storage() {
    let descriptor = descriptor("global_only")
        .slot(global_rw(0, DataType::U32, "out"))
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("a global binding needs no pool");

    assert!(layout.regions.is_empty());
    assert_eq!(layout.shared_pool_bytes, 0);
    assert_eq!(layout.private_pool_bytes, 0);
    assert_eq!(layout.distinct_bytes, 0);
}

#[test]
fn one_invocation_holds_the_values_its_enclosing_scope_still_needs() {
    // Two values live across a branch whose body holds two more at once. The
    // nested pair cannot reuse the enclosing pair's registers, so the peak is
    // the sum and not the deepest single body.
    let descriptor = descriptor("live_values")
        .slot(global_rw(0, DataType::U32, "out"))
        .body(
            body()
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(if_then(0, 0))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .child(
                    body()
                        .op(lit(0, 10))
                        .op(lit(0, 11))
                        .op(effect(KernelOpKind::StoreGlobal, [0, 10, 11])),
                )
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the registers");

    assert_eq!(layout.registers_per_invocation, 4);
    assert_eq!(layout.fragment_words_per_invocation, 0);
}

#[test]
fn a_value_nothing_reads_holds_a_register_only_where_it_is_written() {
    // Two definitions nothing consumes are not two registers: the second
    // reuses the first. Counting them as concurrent would inflate the register
    // ceiling a launch is checked against and cost occupancy for values no op
    // reads.
    let descriptor = descriptor("dead_values")
        .body(
            body()
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(lit(0, 2))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the registers");

    assert_eq!(layout.registers_per_invocation, 1);
}

#[test]
fn a_matrix_fragment_states_the_register_words_one_invocation_holds() {
    let fragment = |element| FragmentValue::in_registers(element, MatrixMmaLayout::RowMajor, 32);
    let spec = MatrixMmaSpec {
        tile: MatrixTileShape {
            m: 16,
            n: 8,
            k: 16,
        },
        left: fragment(MatrixMmaElement::F16),
        right: fragment(MatrixMmaElement::F16),
        accumulator: fragment(MatrixMmaElement::F32),
    };
    let arity = spec.operand_count().expect("spec must state its arity");
    let ops = vec![
        lit(0, 0),
        op(
            KernelOpKind::MatrixMma(Box::new(spec)),
            vec![0; arity as usize],
            100,
        ),
    ];
    let descriptor = descriptor("mma")
        .body(body().ops(ops).literal(LiteralValue::U32(0)))
        .build();

    let layout = StorageLayout::plan(&descriptor).expect("planning must state the fragment words");

    // 16x8x16 over 32 lanes: 4 + 2 + 4 words of 32 bits.
    assert_eq!(layout.fragment_words_per_invocation, 10);
}

#[test]
fn planning_the_same_descriptor_twice_states_the_same_layout() {
    let descriptor = descriptor("deterministic")
        .slots([tile(A, "first"), tile(B, "second"), tile(C, "third")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(C, 0, 1))
                .op(load_shared(A, 0, 2))
                .op(load_shared(B, 0, 3))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let first = StorageLayout::plan(&descriptor).expect("planning must succeed");
    let second = StorageLayout::plan(&descriptor).expect("planning must succeed");

    assert_eq!(first, second);
}

#[test]
fn a_stated_ceiling_the_pool_exceeds_is_rejected_and_an_unstated_one_is_not() {
    let descriptor = descriptor("ceilings")
        .slots([tile(A, "first"), tile(B, "second")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .op(load_shared(B, 0, 2))
                .op(load_shared(A, 0, 3))
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let layout = StorageLayout::plan(&descriptor).expect("planning must succeed");

    assert_eq!(layout.validate(0, 0), Ok(()));
    assert_eq!(layout.validate(128, 4), Ok(()));
    assert_eq!(
        layout.validate(127, 0),
        Err(StorageLayoutError::SharedCeilingExceeded {
            pool_bytes: 128,
            ceiling: 127
        })
    );
    assert_eq!(
        layout.validate(0, 1),
        Err(StorageLayoutError::RegisterCeilingExceeded {
            registers: layout.registers_per_invocation,
            ceiling: 1
        })
    );
}

#[test]
fn a_layout_whose_live_regions_share_bytes_does_not_check() {
    let descriptor = descriptor("hand_edited")
        .slots([tile(A, "first"), tile(B, "second")])
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .op(load_shared(B, 0, 2))
                .op(load_shared(A, 0, 3))
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let planned = StorageLayout::plan(&descriptor).expect("planning must succeed");

    let mut overlaid = planned.clone();
    let region = overlaid
        .regions
        .iter_mut()
        .find(|region| region.slot == B)
        .expect("region B");
    region.offset = 0;
    overlaid.shared_pool_bytes = 64;
    assert_eq!(
        overlaid.validate(0, 0),
        Err(StorageLayoutError::OverlaidWhileLive {
            first: A,
            second: B
        })
    );

    let mut misstated = planned.clone();
    misstated.shared_pool_bytes = 64;
    assert_eq!(
        misstated.validate(0, 0),
        Err(StorageLayoutError::PoolBytesMismatch {
            class: "workgroup",
            stated: 64,
            planned: 128
        })
    );

    let mut summed = planned.clone();
    summed.distinct_bytes = 1;
    assert_eq!(
        summed.validate(0, 0),
        Err(StorageLayoutError::DistinctBytesMismatch {
            stated: 1,
            summed: 128
        })
    );

    let mut unaligned = planned.clone();
    unaligned
        .regions
        .iter_mut()
        .find(|region| region.slot == B)
        .expect("region B")
        .offset = 65;
    assert_eq!(
        unaligned.validate(0, 0),
        Err(StorageLayoutError::UnalignedOffset {
            slot: B,
            offset: 65,
            alignment: 4
        })
    );

    let mut versioned = planned.clone();
    versioned.version = 0;
    assert_eq!(
        versioned.validate(0, 0),
        Err(StorageLayoutError::Version { version: 0 })
    );

    let mut duplicated = planned.clone();
    let copy = duplicated
        .regions
        .iter()
        .find(|region| region.slot == A)
        .expect("region A")
        .clone();
    duplicated.regions.push(copy);
    assert_eq!(
        duplicated.validate(0, 0),
        Err(StorageLayoutError::DuplicateSlot { slot: A })
    );
}

/// WHY: a region is planned from the binding that declares it, so an op naming
/// a slot no binding declares addresses storage with no size, no class and no
/// lifetime. Every operand position that resolves against the binding layout
/// runs through one classification table, so the check sits at that choke
/// point and covers every op kind that reaches it rather than the memory ops
/// somebody remembered.
#[test]
fn an_operation_that_addresses_an_undeclared_slot_is_rejected() {
    let addressing = [
        (KernelOpKind::LoadShared, vec![A, 0], Some(1_u32)),
        (KernelOpKind::LoadGlobal, vec![7, 0], Some(1)),
        (KernelOpKind::BufferLength, vec![7], Some(1)),
    ];
    for (kind, operands, result) in addressing {
        let slot_operand = operands[0];
        let built = descriptor("undeclared")
            .body(
                body()
                    .op(lit(0, 0))
                    .op(KernelOp {
                        kind: kind.clone(),
                        operands,
                        result,
                    })
                    .literal(LiteralValue::U32(0)),
            )
            .build();
        let kinds: Vec<VerifyErrorKind> = match verify(&built) {
            Ok(()) => Vec::new(),
            Err(errors) => errors.into_iter().map(|error| error.kind).collect(),
        };
        assert!(
            kinds.contains(&VerifyErrorKind::UndeclaredBindingSlot {
                operand_pos: 0,
                slot: slot_operand,
            }),
            "Fix: {kind:?} addressing undeclared slot {slot_operand} must be rejected; got {kinds:?}"
        );
    }

    let declared = descriptor("declared")
        .slot(tile(A, "first"))
        .body(
            body()
                .op(lit(0, 0))
                .op(load_shared(A, 0, 1))
                .literal(LiteralValue::U32(0)),
        )
        .build();
    assert_eq!(
        verify(&declared),
        Ok(()),
        "Fix: an op addressing a declared slot is well formed"
    );
}
