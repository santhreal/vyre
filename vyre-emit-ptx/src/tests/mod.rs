use super::*;
use crate::reg::{PtxType, Reg};
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_lower::descriptor_builder::{
    SlotCount,
    body,
    descriptor,
    effect,
    global_ro,
    global_wo,
    lit,
};
use vyre_lower::{
    BindingLayout,
    BindingVisibility,
    Dispatch,
    KernelBody,
    KernelDescriptor,
    KernelOp,
    KernelOpKind,
    LiteralValue,
    MatrixMmaElement,
    MatrixMmaLayout,
    MatrixMmaShape,
    MemoryClass,
};

fn one_store_kernel() -> KernelDescriptor {
    descriptor("store_one")
        .slot(global_wo(0, DataType::U32, "out").with_count(1))
        .body(
            body()
                .ops([lit(0, 0), lit(1, 1), effect(KernelOpKind::StoreGlobal, [0, 0, 1])])
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
