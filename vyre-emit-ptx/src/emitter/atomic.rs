use std::fmt::Write as _;

use vyre_foundation::ir::{AtomicOp, DataType};
use vyre_lower::{KernelOp, MemoryClass};

use super::memory::MemOperand;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

/// A resolved atomic target: the PTX state space, the address register holding
/// the element address in that space, and the bounds predicate the atomic is
/// issued under.
///
/// The state space is part of the instruction mnemonic in PTX
/// (`atom.global.add.u32` vs `atom.shared.add.u32`) and an address is only
/// meaningful in its own space, so the two must be produced together. Emitting
/// `.global` against a shared address either faults with
/// `CUDA_ERROR_ILLEGAL_ADDRESS` or silently reads unrelated global memory.
///
/// The predicate is not optional. Every atomic this emitter issues is bounded
/// by its own binding's length, so an unpredicated form has nothing left to
/// represent and is not offered.
struct AtomicAddress {
    space: &'static str,
    addr: Reg,
    in_bounds: Reg,
}

impl BodyCtx<'_> {
    pub(super) fn emit_atomic(
        &mut self,
        op: &KernelOp,
        atomic_op: AtomicOp,
    ) -> Result<(), EmitError> {
        // CompareExchange / CompareExchangeWeak take 4 operands and use a
        // distinct PTX mnemonic  -  split out so the common single-value
        // RMW path stays clean.
        if matches!(
            atomic_op,
            AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak
        ) {
            return self.emit_atomic_cas(op);
        }
        let mnemonic = match atomic_op {
            AtomicOp::Add => "add",
            AtomicOp::And => "and",
            AtomicOp::Or => "or",
            AtomicOp::Xor => "xor",
            AtomicOp::Min => "min",
            AtomicOp::Max | AtomicOp::LruUpdate => "max",
            AtomicOp::Exchange => "exch",
            _ => {
                return Err(EmitError::UnsupportedOp(KernelOp {
                    kind: op.kind.clone(),
                    operands: op.operands.clone(),
                    result: op.result,
                }));
            }
        };
        let binding_slot = *op
            .operands
            .first()
            .ok_or_else(|| EmitError::InvalidDescriptor("Atomic missing slot".into()))?;
        let index_op_id = *op
            .operands
            .get(1)
            .ok_or_else(|| EmitError::InvalidDescriptor("Atomic missing index".into()))?;
        let value_op_id = *op
            .operands
            .get(2)
            .ok_or_else(|| EmitError::InvalidDescriptor("Atomic missing value".into()))?;
        let binding = self.binding_for_slot(binding_slot)?;
        let element_type = binding.element_type.clone();
        let memory_class = binding.memory_class;
        let elem_ty = PtxType::from_dtype(&element_type)?;
        let index_reg = self.lookup_operand(index_op_id)?;
        let value_reg =
            self.atomic_value_reg(atomic_op, self.lookup_operand(value_op_id)?, elem_ty)?;
        let AtomicAddress {
            space,
            addr,
            in_bounds,
        } = self.emit_atomic_address(binding_slot, index_reg, &element_type, memory_class)?;
        let type_suffix = atomic_type_suffix(atomic_op, elem_ty)?;
        let result_reg = self.alloc(elem_ty);
        let zero_lit = match elem_ty {
            PtxType::F32 => "0f00000000",
            _ => "0",
        };
        let _ = writeln!(
            self.text,
            "    mov.{}    {result_reg}, {zero_lit};",
            elem_ty.ptx_type_str()
        );
        let _ = writeln!(
            self.text,
            "    @{in_bounds} atom.{space}.{mnemonic}.{type_suffix}    {result_reg}, [{addr}], {value_reg};"
        );
        self.bind_result(op, result_reg)
    }

    /// Resolve the address and state space for an atomic on `binding_slot`.
    ///
    /// Global, constant, and uniform bindings keep the original unclamped
    /// `mul.wide.u32` plus `add.u64` arithmetic and the params-buffer length
    /// predicate. Changing those to the clamping form used by plain loads would
    /// silently redirect an out-of-range atomic onto element 0 instead of
    /// predicating it off, corrupting that element.
    ///
    /// The predicate is unconditional. A kernel with no shared memory and no
    /// barrier exits every lane whose global id reaches the dispatch element
    /// count, which is the largest buffer's length; an atomic on a shorter
    /// buffer in the same program is still out of range, and here the address is
    /// not clamped either, so the access leaves the allocation entirely.
    ///
    /// Workgroup-shared bindings take the 32-bit shared-window path. Their
    /// length is NOT in the params buffer (`preload_bindings` skips shared
    /// slots), so the bounds predicate must compare against the compile-time
    /// `element_count` from the binding. Reading a length register for a shared
    /// slot would load an unrelated params word and produce a garbage predicate.
    fn emit_atomic_address(
        &mut self,
        binding_slot: u32,
        index_reg: Reg,
        element_type: &DataType,
        memory_class: MemoryClass,
    ) -> Result<AtomicAddress, EmitError> {
        if matches!(memory_class, MemoryClass::Shared) {
            let element_count = self.binding_for_slot(binding_slot)?.element_count;
            let address = self.emit_memory_address_from_index_reg(
                binding_slot,
                index_reg,
                element_type,
                memory_class,
            )?;
            let MemOperand::Reg(addr) = address.operand else {
                return Err(EmitError::InvalidBinding {
                    slot: binding_slot,
                    reason: "shared atomic address must resolve to a register".into(),
                });
            };
            let Some(count) = element_count else {
                return Err(EmitError::InvalidBinding {
                    slot: binding_slot,
                    reason: "shared atomic binding must declare an element count so the bounds \
                             predicate has a bound"
                        .into(),
                });
            };
            let in_bounds = self.emit_index_lt_immediate_pred(index_reg, count);
            return Ok(AtomicAddress {
                space: address.space,
                addr,
                in_bounds,
            });
        }

        let global_ptr =
            *self
                .slot_to_ptr
                .get(&binding_slot)
                .ok_or_else(|| EmitError::InvalidBinding {
                    slot: binding_slot,
                    reason: format!(
                        "no device pointer is preloaded for this {memory_class:?} binding, so an \
                         atomic cannot address it. preload_bindings populates pointers for global, \
                         constant, and uniform bindings only. Workgroup-shared atomics take a \
                         separate 32-bit shared-window path (handled above); Scratch bindings must \
                         be resolved to real storage before PTX emission."
                    ),
                })?;
        let in_bounds = self.emit_index_reg_in_bounds_pred(binding_slot, index_reg);
        let stride = element_type
            .size_bytes()
            .ok_or_else(|| EmitError::UnsupportedDataType(format!("{element_type:?}")))?;
        let offset_reg = self.alloc(PtxType::U64);
        let _ = writeln!(
            self.text,
            "    mul.wide.u32    {offset_reg}, {index_reg}, {stride};"
        );
        let addr = self.alloc(PtxType::U64);
        let _ = writeln!(
            self.text,
            "    add.u64    {addr}, {global_ptr}, {offset_reg};"
        );
        Ok(AtomicAddress {
            space: "global",
            addr,
            in_bounds,
        })
    }

    /// `pred = index < count` against a compile-time bound.
    ///
    /// Used for shared bindings, whose length never reaches the params buffer.
    fn emit_index_lt_immediate_pred(&mut self, index_reg: Reg, count: u32) -> Reg {
        let bound = self.alloc(PtxType::U32);
        let in_bounds = self.alloc(PtxType::Bool);
        let _ = writeln!(self.text, "    mov.u32    {bound}, {count};");
        let _ = writeln!(
            self.text,
            "    setp.lt.u32    {in_bounds}, {index_reg}, {bound};"
        );
        in_bounds
    }

    fn atomic_value_reg(
        &mut self,
        atomic_op: AtomicOp,
        value_reg: Reg,
        elem_ty: PtxType,
    ) -> Result<Reg, EmitError> {
        if value_reg.0 == PtxType::Bool
            && matches!(
                atomic_op,
                AtomicOp::Exchange | AtomicOp::And | AtomicOp::Or | AtomicOp::Xor
            )
        {
            return Ok(self.coerce_for_store(value_reg, elem_ty));
        }
        Ok(value_reg)
    }

    /// Lower `Atomic { op: CompareExchange | CompareExchangeWeak }` to PTX
    /// `atom.<space>.cas.b32`. PTX CAS returns the prior value of the slot;
    /// callers compare it to `cmp` to decide whether the swap committed.
    /// Operands: `[slot, index, cmp_val, new_val]`.
    fn emit_atomic_cas(&mut self, op: &KernelOp) -> Result<(), EmitError> {
        let binding_slot = *op
            .operands
            .first()
            .ok_or_else(|| EmitError::InvalidDescriptor("AtomicCAS missing slot".into()))?;
        let index_op_id = *op
            .operands
            .get(1)
            .ok_or_else(|| EmitError::InvalidDescriptor("AtomicCAS missing index".into()))?;
        let cmp_op_id = *op
            .operands
            .get(2)
            .ok_or_else(|| EmitError::InvalidDescriptor("AtomicCAS missing cmp value".into()))?;
        let new_op_id = *op
            .operands
            .get(3)
            .ok_or_else(|| EmitError::InvalidDescriptor("AtomicCAS missing new value".into()))?;
        let binding = self.binding_for_slot(binding_slot)?;
        let element_type = binding.element_type.clone();
        let memory_class = binding.memory_class;
        let elem_ty = PtxType::from_dtype(&element_type)?;
        if !matches!(elem_ty, PtxType::U32 | PtxType::I32) {
            return Err(EmitError::UnsupportedDataType(format!(
                "atom.cas requires 32-bit element type; got {element_type:?}"
            )));
        }
        let index_reg = self.lookup_operand(index_op_id)?;
        let cmp_reg = self.lookup_operand(cmp_op_id)?;
        let new_reg = self.lookup_operand(new_op_id)?;
        let AtomicAddress {
            space,
            addr,
            in_bounds,
        } = self.emit_atomic_address(binding_slot, index_reg, &element_type, memory_class)?;
        let result_reg = self.alloc(elem_ty);
        let zero_lit = match elem_ty {
            PtxType::F32 => "0f00000000",
            _ => "0",
        };
        let _ = writeln!(
            self.text,
            "    mov.{}    {result_reg}, {zero_lit};",
            elem_ty.ptx_type_str()
        );
        let _ = writeln!(
            self.text,
            "    @{in_bounds} atom.{space}.cas.b32    {result_reg}, [{addr}], {cmp_reg}, {new_reg};"
        );
        self.bind_result(op, result_reg)
    }
}

fn atomic_type_suffix(atomic_op: AtomicOp, elem_ty: PtxType) -> Result<&'static str, EmitError> {
    if matches!(
        atomic_op,
        AtomicOp::Exchange | AtomicOp::And | AtomicOp::Or | AtomicOp::Xor
    ) {
        return match elem_ty {
            PtxType::U32 | PtxType::I32 => Ok("b32"),
            PtxType::U64 => Ok("b64"),
            other => Err(EmitError::UnsupportedDataType(format!(
                "atom.global bitwise/exchange requires a 32-bit or 64-bit integer element type; got {other:?}"
            ))),
        };
    }
    match elem_ty {
        PtxType::U32 => Ok("u32"),
        PtxType::I32 => Ok("s32"),
        PtxType::U64 => Ok("u64"),
        PtxType::F32 => Ok("f32"),
        other => Err(EmitError::UnsupportedDataType(format!(
            "atom.{atomic_op:?} requires a supported numeric element type; got {other:?}"
        ))),
    }
}
