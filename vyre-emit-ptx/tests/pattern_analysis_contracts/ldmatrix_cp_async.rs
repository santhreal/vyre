//! `ldmatrix_cp_async` pattern analysis contracts.

use vyre_emit_ptx::ComputeCapability;
use vyre_lower::KernelOpKind;
use vyre_emit_ptx::patterns::ldmatrix_cp_async::*;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_rw, lit, op, shared_rw,
};
use vyre_lower::{KernelDescriptor, LiteralValue};

fn cp_async_kernel() -> KernelDescriptor {
    // load(global, 0) → r0; store(shared, 0, r0)
    descriptor("cp_async")
        .slots([
            global_ro(0, DataType::F32, "g"),
            shared_rw(1, DataType::F32, 64, "s"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    effect(KernelOpKind::StoreShared, [1, 0, 1]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build()
}

#[test]
fn cp_async_unsupported_on_volta() {
    let p = analyze(&cp_async_kernel(), ComputeCapability::SM_70);
    assert!(!p.target_supports_cp_async);
    assert!(p.candidates.is_empty());
}

#[test]
fn cp_async_supported_on_ampere() {
    let p = analyze(&cp_async_kernel(), ComputeCapability::SM_80);
    assert!(p.target_supports_cp_async);
    assert_eq!(p.candidates.len(), 1);
    assert_eq!(p.candidates[0].load_op_index, 1);
    assert_eq!(p.candidates[0].store_op_index, 2);
    assert_eq!(p.candidates[0].global_binding_slot, 0);
    assert_eq!(p.candidates[0].shared_binding_slot, 1);
}

#[test]
fn empty_kernel_yields_no_candidates() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(p.candidates.is_empty());
}

#[test]
fn load_without_immediate_store_no_candidate() {
    let desc = descriptor("load_only")
        .slot(global_ro(0, DataType::F32, "g"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([lit(0, 0), op(KernelOpKind::LoadGlobal, [0, 0], 1)])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(p.candidates.is_empty());
}

#[test]
fn store_to_global_not_shared_no_candidate() {
    let desc = descriptor("store_global")
        .slot(global_rw(0, DataType::F32, "g"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(
        p.candidates.is_empty(),
        "global→global not a cp.async candidate"
    );
}

#[test]
fn mismatched_load_store_index_no_candidate() {
    let mut desc = cp_async_kernel();
    desc.id = "cp_async_mismatched_index".into();
    desc.body.ops[2].operands[1] = 99;
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(
        p.candidates.is_empty(),
        "cp.async requires the global load and shared store to use the same logical index"
    );
}
