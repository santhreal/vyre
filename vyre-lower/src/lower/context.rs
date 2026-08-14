//! Descriptor context construction: binding slot assignment and carrier state.

use crate::descriptor::{BindingSlot, MemoryClass};
use crate::error::LowerError;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use vyre_foundation::ir::{Ident, Program};

use super::descriptor_metadata::{binding_visibility, memory_class};
use super::scope::VarScope;
use super::{LowerCtx, WORKGROUP_SLOT_BASE};

impl LowerCtx {
    pub(super) fn new(program: &Program) -> Result<Self, LowerError> {
        let mut bindings = Vec::with_capacity(program.buffers().len());
        let mut buffer_slots = FxHashMap::default();
        let mut slot_memory_classes = FxHashMap::default();
        // Soundness: split slot allocation by memory class. Host-bound
        // buffers (Global, Constant) keep their declared binding ids
        // because the dispatch path looks them up by the same
        // BufferDecl::binding() value. Workgroup/Scratch buffers are
        // SM-local  -  they don't get bound by the host  -  so they live
        // in a high range starting at WORKGROUP_SLOT_BASE that cannot
        // collide with host-bound slots. Without this split,
        // multiple `BufferDecl::workgroup(...)` calls (which all
        // default to binding=0) collided with the host-bound input
        // and forced the output's slot to be auto-renumbered, which
        // then broke the dispatch path's slot-id-keyed lookup.
        let mut host_used_slots = FxHashSet::default();
        let mut host_next_free_slot = 0u32;
        let mut shared_next_slot = WORKGROUP_SLOT_BASE;
        for buffer in program.buffers() {
            let mc = memory_class(buffer)?;
            let slot = match mc {
                MemoryClass::Shared | MemoryClass::Scratch => {
                    let s = shared_next_slot;
                    shared_next_slot = shared_next_slot
                        .checked_add(1)
                        .ok_or(LowerError::OperandIdOverflow)?;
                    s
                }
                MemoryClass::Global | MemoryClass::Constant | MemoryClass::Uniform => {
                    let requested = buffer.binding();
                    let s = if host_used_slots.insert(requested) {
                        requested
                    } else {
                        while host_used_slots.contains(&host_next_free_slot)
                            || host_next_free_slot >= WORKGROUP_SLOT_BASE
                        {
                            host_next_free_slot = host_next_free_slot
                                .checked_add(1)
                                .ok_or(LowerError::OperandIdOverflow)?;
                        }
                        host_used_slots.insert(host_next_free_slot);
                        host_next_free_slot
                    };
                    while host_used_slots.contains(&host_next_free_slot)
                        || host_next_free_slot >= WORKGROUP_SLOT_BASE
                    {
                        host_next_free_slot = host_next_free_slot
                            .checked_add(1)
                            .ok_or(LowerError::OperandIdOverflow)?;
                    }
                    s
                }
            };
            buffer_slots.insert(Ident::from(Arc::clone(&buffer.name)), slot);
            slot_memory_classes.insert(slot, mc);
            bindings.push(BindingSlot {
                slot,
                element_type: buffer.element.clone(),
                element_count: (buffer.count != 0).then_some(buffer.count),
                memory_class: mc,
                visibility: binding_visibility(&buffer.access),
                name: buffer.name().to_owned(),
            });
        }
        bindings.sort_by_key(|slot| slot.slot);
        Ok(Self {
            bindings,
            buffer_slots,
            slot_memory_classes,
            scope: VarScope::default(),
            next_value: 0,
            active_carriers: Vec::new(),
        })
    }

    pub(super) fn is_active_carrier(&self, name: &Ident) -> bool {
        self.active_carriers
            .iter()
            .any(|frame| frame.contains(name))
    }
}

#[cfg(test)]
mod tests {
    use super::super::lower;
    use vyre_foundation::ir::{DataType, Program};

    #[test]
    fn lower_assigns_unique_descriptor_slots_for_duplicate_program_bindings() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![
                BufferDecl::workgroup("scratch", 16, DataType::U32),
                BufferDecl::output("out", 0, DataType::U32).with_count(1),
            ],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        );

        let desc = lower(&program).expect("Fix: duplicate Program bindings must descriptor-lower");

        assert_eq!(desc.bindings.slots.len(), 2);
        assert_ne!(desc.bindings.slots[0].slot, desc.bindings.slots[1].slot);
        assert!(crate::verify::verify(&desc).is_ok());
    }
}
