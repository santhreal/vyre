use super::*;
use crate::reg::{PtxType, Reg};
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, UnOp};
use vyre_foundation::memory_model::MemoryOrdering;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_wo, lit, op, SlotCount,
};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MatrixMmaElement, MatrixMmaLayout, MatrixMmaShape,
    MemoryClass,
};

fn one_store_kernel() -> KernelDescriptor {
    descriptor("store_one")
        .slot(global_wo(0, DataType::U32, "out").with_count(1))
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

fn two_slot_u32_kernel(
    id: &str,
    ops: Vec<KernelOp>,
    literals: Vec<LiteralValue>,
) -> KernelDescriptor {
    KernelDescriptor {
        id: id.into(),
        bindings: BindingLayout {
            slots: vec![
                global_ro(0, DataType::U32, "input").with_count(16),
                global_wo(1, DataType::U32, "output").with_count(16),
            ],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: body().ops(ops).literals(literals).build(),
    }
}

fn empty_child_body() -> KernelBody {
    body().build()
}

/// One-slot atomic RMW kernel: `slot` is the only binding, addressed by the
/// literal index 0 and combined with the literal `value`.
fn atomic_kernel(
    id: &str,
    slot: BindingSlot,
    threads: u32,
    atomic_op: AtomicOp,
    ordering: MemoryOrdering,
    value: u32,
) -> KernelDescriptor {
    descriptor(id)
        .slot(slot)
        .dispatch(threads, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(
                        KernelOpKind::Atomic {
                            op: atomic_op,
                            ordering,
                        },
                        [0, 0, 1],
                        2,
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(value)]),
        )
        .build()
}

mod async_ops;
mod atomics;
mod barrier;
mod control_flow;
mod data_tensor;
mod memory_vector;
mod preamble;
mod scalar_ops;
mod subgroup;
mod types_registers;
