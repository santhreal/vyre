use super::*;
use naga::{Binding, BuiltIn, Statement, TypeInner};
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_foundation::MemoryOrdering;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, SlotCount};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelDescriptor, KernelOpKind,
    LiteralValue, MemoryClass,
};

fn empty_desc() -> KernelDescriptor {
    descriptor("empty").build()
}

fn empty_desc_with_workgroup(id: &str, x: u32) -> KernelDescriptor {
    KernelDescriptor {
        id: id.into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(x, 1, 1),
        body: body().build(),
    }
}

#[test]
fn op_dispatch_route_cache_hits_preserve_uncached_classification() {
    let kinds = [
        KernelOpKind::Literal,
        KernelOpKind::Literal,
        KernelOpKind::BinOpKind(BinOp::Add),
        KernelOpKind::BinOpKind(BinOp::Mul),
        KernelOpKind::UnOpKind(UnOp::BitNot),
        KernelOpKind::UnOpKind(UnOp::Abs),
        KernelOpKind::Cast {
            target: DataType::U32,
        },
        KernelOpKind::Cast {
            target: DataType::I32,
        },
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::SeqCst,
        },
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::Acquire,
        },
        KernelOpKind::LoadGlobal,
        KernelOpKind::LoadGlobal,
    ];
    let (parity, hits) = emitter::op_dispatch_route_cache_probe(&kinds);
    assert!(
        parity,
        "Fix: cached Naga op-dispatch route classification must match uncached classification."
    );
    assert!(
        hits >= 6,
        "Fix: repeated Naga op kinds must hit the dispatch-route cache; observed {hits}."
    );
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

mod naga_probe;
pub(crate) use naga_probe::{block_has_atomic, block_has_loop, entry_has_binary, entry_has_unary};

mod atomics;
mod binop;
mod byte_element_load;
mod cache_entry;
mod descriptor_control;
mod subgroup;
