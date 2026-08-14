//! Descriptor constructors used by fixtures in this crate.

use crate::descriptor::{BindingSlot, BindingVisibility, KernelOp, KernelOpKind, MemoryClass};
use vyre_foundation::ir::DataType;

/// Binding constructor used by descriptor fixtures in this crate.
pub(crate) fn binding_slot(
    slot: u32,
    name: impl Into<String>,
    element_type: DataType,
    element_count: Option<u32>,
    memory_class: MemoryClass,
    visibility: BindingVisibility,
) -> BindingSlot {
    BindingSlot {
        slot,
        element_type,
        element_count,
        memory_class,
        visibility,
        name: name.into(),
    }
}

/// Scalar-store constructor used by descriptor fixtures in this crate.
pub(crate) fn store_global(
    slot_operand_id: u32,
    index_operand_id: u32,
    value_operand_id: u32,
) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::StoreGlobal,
        operands: vec![slot_operand_id, index_operand_id, value_operand_id],
        result: None,
    }
}

/// U32 literal constructor used by descriptor fixtures in this crate.
pub(crate) fn literal_u32(literal_pool_index: u32, result_id: u32) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::Literal,
        operands: vec![literal_pool_index],
        result: Some(result_id),
    }
}

#[cfg(test)]
mod tests {
    use super::{binding_slot, literal_u32, store_global};
    use crate::descriptor::{BindingVisibility, KernelOpKind, MemoryClass};
    use vyre_foundation::ir::DataType;

    #[test]
    fn binding_slot_helper_records_inputs() {
        let s = binding_slot(
            3,
            "scratch",
            DataType::F32,
            Some(64),
            MemoryClass::Shared,
            BindingVisibility::ReadWrite,
        );
        assert_eq!(s.slot, 3);
        assert_eq!(s.name, "scratch");
        assert_eq!(s.element_type, DataType::F32);
        assert_eq!(s.element_count, Some(64));
        assert_eq!(s.memory_class, MemoryClass::Shared);
        assert_eq!(s.visibility, BindingVisibility::ReadWrite);
    }

    #[test]
    fn store_global_helper_packs_three_operands() {
        let op = store_global(0, 1, 2);
        assert_eq!(op.kind, KernelOpKind::StoreGlobal);
        assert_eq!(op.operands, vec![0, 1, 2]);
        assert_eq!(op.result, None);
    }

    #[test]
    fn literal_u32_helper_assigns_result_id() {
        let op = literal_u32(5, 42);
        assert_eq!(op.kind, KernelOpKind::Literal);
        assert_eq!(op.operands, vec![5]);
        assert_eq!(op.result, Some(42));
    }
}
