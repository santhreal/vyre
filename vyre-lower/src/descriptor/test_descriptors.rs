//! Descriptor fixture shared by the descriptor unit tests.

use vyre_foundation::ir::DataType;

use super::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, LiteralValue, MemoryClass,
};

pub(super) fn build(ops: Vec<KernelOp>, child_bodies: Vec<KernelBody>) -> KernelDescriptor {
    KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "buf".into(),
            }],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops,
            child_bodies,
            literals: vec![LiteralValue::U32(7)],
        },
    }
}
