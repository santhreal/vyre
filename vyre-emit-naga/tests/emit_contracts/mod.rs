//! Naga emitter contracts over the public `vyre_emit_naga` surface.
//!
//! The descriptor fixtures in this file are the ones every submodule below
//! reaches through `use super::*`.
use naga::{Binding, BuiltIn, Statement, TypeInner};
use vyre_emit_naga::*;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, SlotCount};
use vyre_lower::{
    BindingSlot, BindingVisibility, KernelDescriptor, KernelOpKind, LiteralValue, MemoryClass,
};

fn empty_desc() -> KernelDescriptor {
    descriptor("empty").build()
}

fn empty_desc_with_workgroup(id: &str, x: u32) -> KernelDescriptor {
    descriptor(id).dispatch(x, 1, 1).build()
}

fn u32_output_slot(slot: u32) -> BindingSlot {
    global_rw(slot, DataType::U32, &format!("out{slot}")).with_count(8)
}

fn trap_sidecar_slot(slot: u32) -> BindingSlot {
    BindingSlot {
        slot,
        element_type: DataType::U32,
        element_count: Some(vyre_lower::TRAP_SIDECAR_WORDS),
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: vyre_lower::TRAP_SIDECAR_NAME.to_owned(),
    }
}

fn async_copy_desc(kind: KernelOpKind) -> KernelDescriptor {
    descriptor("async-copy")
        .slots([u32_output_slot(0), u32_output_slot(1)])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(kind, [0, 1, 0, 1]),
                    effect(KernelOpKind::AsyncWait { tag: "copy".into() }, []),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build()
}

/// One binding, two literals, one `StoreGlobal`: the smallest descriptor that
/// still emits a global variable and a statement.
pub(crate) fn single_store_desc(id: &str) -> KernelDescriptor {
    descriptor(id)
        .slots([global_rw(0, DataType::U32, "out")])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
        )
        .build()
}

#[path = "../support/naga_probe.rs"]
mod naga_probe;
pub(crate) use naga_probe::{block_has_atomic, block_has_loop, entry_has_binary, entry_has_unary};

mod atomics;
mod binop;
mod byte_element_load;
mod cache_entry;
mod descriptor_control;
mod pattern_audit;
mod pattern_pipeline_prewarm;
mod pattern_vec_pack;
mod subgroup;
