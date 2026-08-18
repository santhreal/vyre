//! Test: types registers.
use super::*;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_wo, lit, op, SlotCount,
};

#[test]
fn capability_constants_present() {
    assert_eq!(ComputeCapability::SM_70.major, 7);
    assert_eq!(ComputeCapability::SM_89.major, 8);
    assert_eq!(ComputeCapability::SM_89.minor, 9);
    assert_eq!(ComputeCapability::SM_90.major, 9);
    assert_eq!(ComputeCapability::SM_100.major, 10);
    assert_eq!(ComputeCapability::SM_120.major, 12);
    assert!(ComputeCapability::SM_80.supports_async_copy());
    assert!(ComputeCapability::SM_89.supports_async_copy());
    assert!(!ComputeCapability::SM_70.supports_async_copy());
    assert!(ComputeCapability::SM_75.supports_ldmatrix());
    assert!(ComputeCapability::SM_89.supports_ldmatrix());
    assert!(!ComputeCapability::SM_70.supports_ldmatrix());
}

#[test]
fn register_declaration_sized_to_used_count() {
    // A kernel with 3 u32 ops declares those registers on top of
    // the reserved launch-ABI registers.
    let kernel = descriptor("regs")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(1), LiteralValue::U32(2)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains(".reg .u32   %r<30>;"));
}

fn narrow_global_copy_kernel(element_type: DataType) -> KernelDescriptor {
    descriptor("narrow_copy")
        .slots([
            global_ro(0, element_type.clone(), "input").with_count(8),
            global_wo(1, element_type, "output").with_count(8),
        ])
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    effect(KernelOpKind::StoreGlobal, [1, 0, 1]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build()
}

#[test]
fn narrow_integer_global_memory_uses_narrow_ptx_ops() {
    for (data_type, load_op, store_op) in [
        (DataType::U8, "ld.global.u8", "st.global.u8"),
        (DataType::I8, "ld.global.s8", "st.global.u8"),
        (DataType::U16, "ld.global.u16", "st.global.u16"),
        (DataType::I16, "ld.global.s16", "st.global.u16"),
    ] {
        let ptx = emit(&narrow_global_copy_kernel(data_type.clone())).unwrap();
        assert!(
            ptx.contains(load_op),
            "Fix: {data_type:?} loads must use byte/halfword PTX instead of widening the memory transaction:\n{ptx}"
        );
        assert!(
            ptx.contains(store_op),
            "Fix: {data_type:?} stores must use byte/halfword PTX instead of widening the memory transaction:\n{ptx}"
        );
    }
}
