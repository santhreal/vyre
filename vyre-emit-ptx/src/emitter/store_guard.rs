//! Predicate composition for guarded stores.
//!
//! Owns the rule that decides which predicate a store is issued under: an
//! incoming branch predicate, the element-count bounds check every global store
//! requires, or the conjunction of both. It writes only the predicate
//! arithmetic; the store instruction itself comes from `memory`.

use std::fmt::Write as _;

use vyre_lower::{KernelOp, KernelOpKind, MemoryClass};

use super::operand_decode::read_store_operands;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

impl BodyCtx<'_> {
    pub(super) fn emit_predicated_store(
        &mut self,
        op: &KernelOp,
        pred: Reg,
        negate: bool,
    ) -> Result<bool, EmitError> {
        if !matches!(
            op.kind,
            KernelOpKind::StoreGlobal | KernelOpKind::StoreShared
        ) {
            return Ok(false);
        }
        let (binding_slot, index_op_id, value_op_id) = read_store_operands(op)?;
        let binding = self.binding_for_slot(binding_slot)?;
        let element_type = binding.element_type.clone();
        let memory_class = binding.memory_class;
        let elem_ty = PtxType::from_dtype(&element_type)?;
        let value_reg = self.coerce_for_store(self.lookup_operand(value_op_id)?, elem_ty);
        let address =
            self.emit_memory_address(binding_slot, index_op_id, &element_type, memory_class)?;
        let guard = self.store_guard_for_index(
            binding_slot,
            index_op_id,
            memory_class,
            Some((if negate { "@!" } else { "@" }, pred)),
        )?;
        self.emit_store_value(guard, address, &element_type, value_reg)?;
        Ok(true)
    }

    /// Compose the predicate a store is issued under.
    ///
    /// A global store always carries its buffer's bounds check. The address it
    /// writes through is clamped (`memory::clamp_index_to_buffer_length`), which
    /// keeps an out-of-range address from faulting but redirects the write onto
    /// element 0, so a store past the end silently overwrites the first element
    /// with whatever value the widest lane carried. The entry-wide exit a kernel
    /// without shared memory emits does not cover it: that exit compares the
    /// global id against the dispatch element count, which is the largest
    /// buffer's length, and says nothing about a shorter buffer in the same
    /// program.
    ///
    /// Shared bindings keep the address the workgroup window resolves and are
    /// bounded by their compile-time length, so they take the incoming predicate
    /// unchanged.
    pub(super) fn store_guard_for_index(
        &mut self,
        binding_slot: u32,
        index_op_id: u32,
        memory_class: MemoryClass,
        existing: Option<(&str, Reg)>,
    ) -> Result<Option<(String, Reg)>, EmitError> {
        let existing = existing.map(|(prefix, pred)| (prefix.to_string(), pred));
        if !matches!(memory_class, MemoryClass::Global) {
            return Ok(existing);
        }

        let in_bounds = self.emit_index_in_bounds_pred(binding_slot, index_op_id)?;
        let Some((prefix, pred)) = existing else {
            return Ok(Some(("@".to_string(), in_bounds)));
        };
        let branch_live = match prefix.as_str() {
            "@" => pred,
            "@!" => {
                let not_pred = self.alloc(PtxType::Bool);
                let _ = writeln!(self.text, "    not.pred    {not_pred}, {pred};");
                not_pred
            }
            other => {
                return Err(EmitError::InvalidDescriptor(format!(
                    "unsupported PTX store guard prefix {other:?}. Fix: use @ or @! predication."
                )));
            }
        };
        let combined = self.alloc(PtxType::Bool);
        let _ = writeln!(
            self.text,
            "    and.pred    {combined}, {branch_live}, {in_bounds};"
        );
        Ok(Some(("@".to_string(), combined)))
    }
}
