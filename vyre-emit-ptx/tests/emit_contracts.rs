//! PTX emitter contracts over the public `vyre_emit_ptx` surface.
//!
//! Every descriptor fixture in this file is shared by the submodules below,
//! which reach it through `use super::*`.
use vyre_emit_ptx::*;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, UnOp};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_wo, lit, mma_f16_m16n8k16, op, SlotCount,
};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
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

/// The op chain a four-way vector load fuses from: literal 0 is the base index
/// and literal 1 the stride, then four loads land on result ids 2, 4, 6 and 8,
/// each preceded by the add that steps the index. `load` selects the load kind,
/// so the `LoadConstant` shape `const_buffer_promote` leaves behind is the same
/// chain. Callers append their own tail ops.
pub(crate) fn four_load_chain(load: KernelOpKind) -> Vec<KernelOp> {
    vec![
        lit(0, 0),
        lit(1, 1),
        op(load.clone(), [0, 0], 2),
        op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 3),
        op(load.clone(), [0, 3], 4),
        op(KernelOpKind::BinOpKind(BinOp::Add), [3, 1], 5),
        op(load.clone(), [0, 5], 6),
        op(KernelOpKind::BinOpKind(BinOp::Add), [5, 1], 7),
        op(load, [0, 7], 8),
    ]
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

#[path = "emit_contracts/async_ops.rs"]
mod async_ops;
#[path = "emit_contracts/atomics.rs"]
mod atomics;
#[path = "emit_contracts/barrier.rs"]
mod barrier;
#[path = "emit_contracts/control_flow.rs"]
mod control_flow;
#[path = "emit_contracts/data_tensor/mod.rs"]
mod data_tensor;
#[path = "emit_contracts/memory_vector/mod.rs"]
mod memory_vector;
#[path = "emit_contracts/preamble.rs"]
mod preamble;
#[path = "emit_contracts/scalar_ops.rs"]
mod scalar_ops;
#[path = "emit_contracts/subgroup.rs"]
mod subgroup;
#[path = "emit_contracts/types_registers.rs"]
mod types_registers;
