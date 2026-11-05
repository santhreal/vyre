//! Test: analysis fixture corpuses.
use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::analyses::{
    analyze_bank_conflict, analyze_coalesce, analyze_shared_mem_promote, vec_pack,
};
use vyre_lower::descriptor_builder::{
    binop, body, descriptor, fixed_global_ro, lit, load_global, op, shared_rw,
};
use vyre_lower::{KernelOpKind, LiteralValue};
/// Bank count a case states; the analysis holds no default for it.
fn banks(count: u32) -> std::num::NonZeroU32 {
    std::num::NonZeroU32::new(count).expect("a stated bank count is nonzero")
}
/// Per-workgroup shared capacity a case states; the analysis holds no default.
const SHARED_BYTES: std::num::NonZeroU32 = match std::num::NonZeroU32::new(48 * 1024) {
    Some(bytes) => bytes,
    None => unreachable!(),
};
#[test]
fn a13_coalesce_corpus_classifies_unit_stride_strided_and_broadcast() {
    let desc = descriptor("a13_coalesce_fixture")
        .slot(fixed_global_ro(0, DataType::U32, 4096, "input"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(4), LiteralValue::U32(7)])
                .op(op(KernelOpKind::LocalInvocationId, [0], 1))
                .op(lit(0, 2))
                .op(lit(1, 3))
                .op(binop(BinOp::Mul, 1, 2, 4))
                .op(load_global(0, 1, 10))
                .op(load_global(0, 4, 11))
                .op(load_global(0, 3, 12)),
        )
        .build();

    let report = analyze_coalesce(&desc);
    assert_eq!(report.sites.len(), 3);
    assert_eq!(
        report.sites[0].pattern,
        vyre_lower::analyses::AccessPattern::CoalescedUnitStride
    );
    assert_eq!(
        report.sites[1].pattern,
        vyre_lower::analyses::AccessPattern::Strided { stride: 4 }
    );
    assert_eq!(
        report.sites[2].pattern,
        vyre_lower::analyses::AccessPattern::Broadcast
    );
    assert_eq!(report.problematic_count(), 1);
}

#[test]
fn a14_shared_mem_promote_corpus_finds_reused_global_tile() {
    let desc = descriptor("a14_shared_mem_promote_fixture")
        .slot(fixed_global_ro(0, DataType::U32, 4096, "hot"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .op(op(KernelOpKind::LocalInvocationId, [0], 1))
                .op(load_global(0, 1, 10))
                .op(load_global(0, 1, 11))
                .op(load_global(0, 1, 12)),
        )
        .build();

    let plan = analyze_shared_mem_promote(&desc, SHARED_BYTES);
    assert_eq!(plan.candidates.len(), 1);
    let candidate = &plan.candidates[0];
    assert_eq!(candidate.binding_slot, 0);
    assert_eq!(candidate.access_count, 3);
    assert_eq!(candidate.tile_bytes, 1024);
    assert!(plan.fits_in_budget());
}

#[test]
fn a15_bank_conflict_corpus_detects_full_warp_serialization() {
    let desc = descriptor("a15_bank_conflict_fixture")
        .slot(shared_rw(2, DataType::U32, 4096, "tile"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(32)])
                .op(op(KernelOpKind::LocalInvocationId, [0], 1))
                .op(lit(0, 2))
                .op(binop(BinOp::Mul, 1, 2, 3))
                .op(op(KernelOpKind::LoadShared, [2, 3], 10)),
        )
        .build();

    let report = analyze_bank_conflict(&desc, banks(32));
    assert_eq!(report.sites.len(), 1);
    assert_eq!(
        report.sites[0].conflict,
        vyre_lower::analyses::BankConflictKind::Conflict { way_count: 32 }
    );
    assert_eq!(report.critical_count(), 1);
}

#[test]
fn a16_vec_pack_corpus_detects_adjacent_load_chain() {
    let desc = descriptor("a16_vec_pack_fixture")
        .slot(fixed_global_ro(0, DataType::U32, 4096, "input"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(64),
                    LiteralValue::U32(65),
                    LiteralValue::U32(66),
                    LiteralValue::U32(67),
                ])
                .op(lit(0, 1))
                .op(lit(1, 2))
                .op(lit(2, 3))
                .op(lit(3, 4))
                .op(load_global(0, 1, 10))
                .op(load_global(0, 2, 11))
                .op(load_global(0, 3, 12))
                .op(load_global(0, 4, 13)),
        )
        .build();

    let report = vec_pack::analyze(&desc);
    assert!(report.has_chains());
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].slot, 0);
    assert_eq!(report.chains[0].start_index, 64);
    assert_eq!(report.chains[0].pack_width(), 4);
    assert_eq!(report.total_ops_eliminated, 3);
}
