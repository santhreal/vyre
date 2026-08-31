//! Every capacity a neutral analysis needs is a fact the caller states.
//!
//! Bank count, per-workgroup shared capacity, and constant capacity all differ
//! between targets. A default stored in this crate would report a finding for a
//! device that has none and miss one on a device that does, while looking
//! exactly like a real finding. The cases below pin, for each fact, that the
//! stated value decides the verdict and that an unstated fact yields no section
//! and no transform rather than one computed from an assumed limit.
//!
//! The closure is structural: `AnalysisFacts` is the single record every
//! neutral entry point reads, so a fourth capacity cannot reach an analysis
//! without appearing here as a field with no default.

use std::num::NonZeroU32;

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::analyses::{AnalysisFacts, BankConflictKind};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, effect, fixed_global_ro, global_rw, lit, op, shared_rw,
};
use vyre_lower::rewrites::apply_lowering_rewrites;
use vyre_lower::{
    audit, KernelDescriptor, KernelOpKind, LiteralValue, MemoryClass, WORKGROUP_SLOT_BASE,
};

fn nonzero(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("a stated capacity is nonzero")
}

/// `shared[tid * 32]`: every lane of a 32-lane wave hits one bank when the
/// device has 32 of them, and no two lanes collide when it has a prime count.
fn strided_shared_load() -> KernelDescriptor {
    descriptor("strided")
        .slots([shared_rw(WORKGROUP_SLOT_BASE, DataType::U32, 1024, "tile")])
        .dispatch(32, 1, 1)
        .body(
            body()
                .op(op(KernelOpKind::LocalInvocationId, [0], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(op(KernelOpKind::LoadShared, [WORKGROUP_SLOT_BASE, 2], 3))
                .literal(LiteralValue::U32(32)),
        )
        .build()
}

/// A fixed 64-element read-only global read twice, the shape both the
/// shared-memory and the constant-buffer rules admit.
fn twice_read_lookup_table() -> KernelDescriptor {
    descriptor("lookup")
        .slot(fixed_global_ro(0, DataType::U32, 64, "lut"))
        .slot(global_rw(1, DataType::U32, "dest"))
        .dispatch(64, 1, 1)
        .body(body().literals([LiteralValue::U32(0)]).ops([
            lit(0, 0),
            op(KernelOpKind::LoadGlobal, [0, 0], 1),
            op(KernelOpKind::LoadGlobal, [0, 0], 2),
            op(KernelOpKind::BinOpKind(BinOp::Add), [1, 2], 3),
            effect(KernelOpKind::StoreGlobal, [1, 0, 3]),
        ]))
        .build()
}

#[test]
fn an_audit_with_no_stated_facts_reports_no_capacity_section() {
    let report = audit(&strided_shared_load(), &AnalysisFacts::none());

    assert!(
        report.bank_conflict.is_none(),
        "no reported geometry means no bank verdict"
    );
    assert!(
        report.shared_mem.is_none(),
        "no reported capacity means no promotion verdict"
    );
    for category in ["BankConflict", "SharedMemPromote"] {
        assert!(
            !report
                .recommendations
                .iter()
                .any(|item| format!("{:?}", item.category).contains(category)),
            "a {category} recommendation without its device fact is a guess"
        );
    }
}

#[test]
fn the_stated_bank_count_decides_the_verdict() {
    let kernel = strided_shared_load();

    let on_thirty_two = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_banks(nonzero(32)),
    )
    .bank_conflict
    .expect("a stated count produces a bank section");
    let on_seventeen = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_banks(nonzero(17)),
    )
    .bank_conflict
    .expect("a stated count produces a bank section");

    assert_eq!(on_thirty_two.bank_count, 32);
    assert_eq!(on_seventeen.bank_count, 17);
    assert_eq!(
        on_thirty_two.sites[0].conflict,
        BankConflictKind::Conflict { way_count: 32 }
    );
    assert_eq!(on_seventeen.sites[0].conflict, BankConflictKind::NoConflict);
}

#[test]
fn a_stated_bank_count_carries_its_conflict_weight_into_the_waste_score() {
    let kernel = strided_shared_load();

    let unstated = audit(&kernel, &AnalysisFacts::none()).waste_score;
    let conflicting = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_banks(nonzero(32)),
    )
    .waste_score;
    let clean = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_banks(nonzero(17)),
    )
    .waste_score;

    assert!(
        conflicting > unstated,
        "a reported 32-way conflict is worth more waste than no bank fact"
    );
    assert_eq!(
        clean, unstated,
        "a device whose layout has no conflict scores like one with no bank findings"
    );
}

#[test]
fn the_stated_shared_capacity_decides_whether_a_tile_fits() {
    let kernel = twice_read_lookup_table();

    let roomy = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_bytes(nonzero(48 * 1024)),
    )
    .shared_mem
    .expect("a stated capacity produces a promotion section");
    let cramped = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_bytes(nonzero(64)),
    )
    .shared_mem
    .expect("a stated capacity produces a promotion section");

    // 64 threads * 4 bytes = 256 bytes of tile for the one candidate binding.
    assert_eq!(roomy.candidates.len(), 1);
    assert_eq!(roomy.total_tile_bytes, 256);
    assert!(roomy.fits_in_budget());
    assert_eq!(cramped.candidates.len(), 1);
    assert!(
        !cramped.fits_in_budget(),
        "256 bytes of tile does not fit a stated 64-byte capacity"
    );
}

#[test]
fn a_stated_shared_capacity_carries_its_candidate_weight_into_the_waste_score() {
    let kernel = twice_read_lookup_table();

    let unstated = audit(&kernel, &AnalysisFacts::none()).waste_score;
    let stated = audit(
        &kernel,
        &AnalysisFacts::none().with_shared_memory_bytes(nonzero(48 * 1024)),
    )
    .waste_score;

    assert!(
        stated > unstated,
        "a promotion candidate is only waste once a capacity is stated"
    );
}

#[test]
fn constant_promotion_waits_for_a_stated_constant_capacity() {
    let kernel = twice_read_lookup_table();

    let unstated = apply_lowering_rewrites(&kernel, &AnalysisFacts::none());
    let stated = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(64 * 1024)),
    );

    assert_eq!(
        unstated.bindings.slots[0].memory_class,
        MemoryClass::Global,
        "no reported constant capacity means no promotion"
    );
    assert_eq!(stated.bindings.slots[0].memory_class, MemoryClass::Constant);
}

#[test]
fn the_stated_constant_capacity_decides_eligibility() {
    let kernel = twice_read_lookup_table();

    // The binding is 64 elements of u32: 256 bytes.
    let too_small = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(128)),
    );
    let large_enough = apply_lowering_rewrites(
        &kernel,
        &AnalysisFacts::none().with_constant_buffer_bytes(nonzero(256)),
    );

    assert_eq!(
        too_small.bindings.slots[0].memory_class,
        MemoryClass::Global,
        "a 256-byte binding does not fit a stated 128-byte capacity"
    );
    assert_eq!(
        large_enough.bindings.slots[0].memory_class,
        MemoryClass::Constant
    );
}
