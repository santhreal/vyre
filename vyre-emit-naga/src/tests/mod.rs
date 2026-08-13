use super::*;
use naga::{Binding, Block, BuiltIn, Statement, TypeInner};
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_foundation::memory_model::MemoryOrdering;
use vyre_lower::descriptor_builder::{SlotCount, body, descriptor, effect, global_rw, lit};
use vyre_lower::{
    BindingLayout,
    BindingSlot,
    BindingVisibility,
    Dispatch,
    KernelDescriptor,
    KernelOpKind,
    LiteralValue,
    MemoryClass,
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

fn block_has_loop(block: &Block) -> bool {
    block.iter().any(|statement| match statement {
        Statement::Loop { .. } => true,
        Statement::Block(child) => block_has_loop(child),
        Statement::If { accept, reject, .. } => block_has_loop(accept) || block_has_loop(reject),
        _ => false,
    })
}

fn block_has_atomic(block: &Block) -> bool {
    block.iter().any(|statement| match statement {
        Statement::Atomic { .. } => true,
        Statement::Block(child) => block_has_atomic(child),
        Statement::If { accept, reject, .. } => {
            block_has_atomic(accept) || block_has_atomic(reject)
        }
        Statement::Loop {
            body, continuing, ..
        } => block_has_atomic(body) || block_has_atomic(continuing),
        _ => false,
    })
}

mod atomics;
mod binop;
mod byte_element_load;
mod cache_entry;
mod descriptor_control;
mod subgroup;
