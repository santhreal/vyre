//! Descriptor op emission: slot lookup, literals, value allocation, and the
//! trap sidecar binding.

use crate::descriptor::{
    BindingSlot, BindingVisibility, KernelBody, KernelOp, KernelOpKind, LiteralValue, MemoryClass,
    TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS,
};
use crate::error::LowerError;
use vyre_foundation::ir::{DataType, Ident};

use super::body_assembly::push_literal;
use super::LowerCtx;

impl LowerCtx {
    pub(super) fn buffer_slot(&self, buffer: &Ident) -> Result<u32, LowerError> {
        self.buffer_slots
            .get(buffer)
            .copied()
            .ok_or_else(|| LowerError::UndeclaredBuffer(buffer.to_string()))
    }

    pub(super) fn load_kind(&self, slot: u32) -> KernelOpKind {
        self.slot_memory_classes
            .get(&slot)
            .copied()
            .map(|memory_class| match memory_class {
                MemoryClass::Shared => KernelOpKind::LoadShared,
                MemoryClass::Constant | MemoryClass::Uniform => KernelOpKind::LoadConstant,
                MemoryClass::Global | MemoryClass::Scratch => KernelOpKind::LoadGlobal,
            })
            .unwrap_or(KernelOpKind::LoadGlobal)
    }

    pub(super) fn store_kind(&self, slot: u32, buffer: &Ident) -> Result<KernelOpKind, LowerError> {
        match self.slot_memory_classes.get(&slot).copied() {
            Some(MemoryClass::Shared) => Ok(KernelOpKind::StoreShared),
            Some(MemoryClass::Constant | MemoryClass::Uniform) => Err(LowerError::InvalidProgram(format!(
                "Store to constant/uniform-class buffer `{buffer}` is invalid  -  read-only at the dispatch boundary. Fix: change the buffer's MemoryKind to Global or its access to ReadWrite."
            ))),
            Some(MemoryClass::Global | MemoryClass::Scratch) => Ok(KernelOpKind::StoreGlobal),
            None => Ok(KernelOpKind::StoreGlobal),
        }
    }

    pub(super) fn add_trap_sidecar_binding(&mut self) -> Result<(), LowerError> {
        if self
            .buffer_slots
            .contains_key(&Ident::from(TRAP_SIDECAR_NAME))
        {
            return Err(LowerError::UnsupportedConstruct(format!(
                "program declares reserved trap sidecar buffer `{TRAP_SIDECAR_NAME}`. Fix: choose a user buffer name outside the `__vyre_*` namespace."
            )));
        }
        // Only consider host-visible slots when picking the next trap sidecar
        // slot. Shared/Scratch slots live in the WORKGROUP_SLOT_BASE (1<<24)
        // range and are not host-bound; mixing them in here would push the
        // trap sidecar  -  which IS host-bound  -  past the max binding
        // index (1000) and the layout validator would reject it.
        let next_slot = self
            .bindings
            .iter()
            .filter(|binding| {
                !matches!(
                    binding.memory_class,
                    MemoryClass::Shared | MemoryClass::Scratch,
                )
            })
            .map(|binding| binding.slot)
            .max()
            .map_or(Ok(0), |slot| {
                slot.checked_add(1).ok_or(LowerError::OperandIdOverflow)
            })?;
        self.buffer_slots
            .insert(Ident::from(TRAP_SIDECAR_NAME), next_slot);
        self.slot_memory_classes
            .insert(next_slot, MemoryClass::Global);
        self.bindings.push(BindingSlot {
            slot: next_slot,
            element_type: DataType::U32,
            element_count: Some(TRAP_SIDECAR_WORDS),
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadWrite,
            name: TRAP_SIDECAR_NAME.to_owned(),
        });
        self.bindings.sort_by_key(|slot| slot.slot);
        Ok(())
    }

    pub(super) fn literal(
        &mut self,
        body: &mut KernelBody,
        literal: LiteralValue,
    ) -> Result<u32, LowerError> {
        let literal_index = push_literal(body, literal)?;
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![literal_index],
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn builtin_axis(
        &mut self,
        body: &mut KernelBody,
        kind: KernelOpKind,
        axis: u8,
    ) -> Result<u32, LowerError> {
        if axis > 2 {
            return Err(LowerError::InvalidProgram(format!(
                "builtin axis {axis} is out of range. Fix: use axis 0, 1, or 2."
            )));
        }
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind,
            operands: vec![u32::from(axis)],
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn simple_result(
        &mut self,
        body: &mut KernelBody,
        kind: KernelOpKind,
    ) -> Result<u32, LowerError> {
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind,
            operands: Vec::new(),
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn copy_value(
        &mut self,
        body: &mut KernelBody,
        operand: u32,
    ) -> Result<u32, LowerError> {
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind: KernelOpKind::Copy,
            operands: vec![operand],
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn unary(
        &mut self,
        body: &mut KernelBody,
        kind: KernelOpKind,
        operand: u32,
    ) -> Result<u32, LowerError> {
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind,
            operands: vec![operand],
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn binary(
        &mut self,
        body: &mut KernelBody,
        kind: KernelOpKind,
        left: u32,
        right: u32,
    ) -> Result<u32, LowerError> {
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind,
            operands: vec![left, right],
            result: Some(result),
        });
        Ok(result)
    }

    pub(super) fn alloc_value(&mut self) -> Result<u32, LowerError> {
        let id = self.next_value;
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or(LowerError::OperandIdOverflow)?;
        Ok(id)
    }

    pub(super) fn emit_loop_carrier_read(
        &mut self,
        body: &mut KernelBody,
        name: &Ident,
    ) -> Result<(), LowerError> {
        let result = self.alloc_value()?;
        body.ops.push(KernelOp {
            kind: KernelOpKind::LoopCarrier {
                name: name.shared_text(),
            },
            operands: Vec::new(),
            result: Some(result),
        });
        self.scope.bind(name.clone(), result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::lower;
    use crate::descriptor::{BindingVisibility, TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS};
    use vyre_foundation::ir::{DataType, Program};

    #[test]
    pub(super) fn lower_trap_inserts_descriptor_sidecar_binding() {
        use vyre_foundation::ir::{Expr, Node};

        let program = Program::wrapped(
            vec![],
            [64, 1, 1],
            vec![Node::trap(Expr::u32(7), "page-fault")],
        );

        let desc = lower(&program).expect("Fix: trap programs must descriptor-lower");
        let sidecar = desc
            .bindings
            .slots
            .iter()
            .find(|slot| slot.name == TRAP_SIDECAR_NAME)
            .expect("Fix: trap sidecar binding must be inserted");
        assert_eq!(sidecar.element_type, DataType::U32);
        assert_eq!(sidecar.element_count, Some(TRAP_SIDECAR_WORDS));
        assert!(matches!(sidecar.visibility, BindingVisibility::ReadWrite));
        assert!(crate::verify::verify(&desc).is_ok());
    }

    #[test]
    pub(super) fn trap_sidecar_slot_stays_in_host_range_when_program_has_workgroup_buffer() {
        // Regression: trap sidecar must skip Shared/Scratch slots when
        // picking its slot id. Workgroup-class slots live in the
        // 1<<24 reserved range and a host-bound binding past wgpu's
        // 1000-binding limit fails layout creation.
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![
                BufferDecl::output("out", 0, DataType::U32).with_count(1),
                BufferDecl::workgroup("scratch", 16, DataType::U32),
            ],
            [64, 1, 1],
            vec![
                Node::store("out", Expr::u32(0), Expr::u32(1)),
                Node::trap(Expr::u32(7), "fault"),
            ],
        );

        let desc = lower(&program).expect("Fix: trap + workgroup programs must lower");
        let sidecar = desc
            .bindings
            .slots
            .iter()
            .find(|slot| slot.name == TRAP_SIDECAR_NAME)
            .expect("Fix: trap sidecar must be present");
        assert!(
            sidecar.slot < 1024,
            "trap sidecar slot must stay in the host-bindable range; got {}",
            sidecar.slot,
        );
    }
}
