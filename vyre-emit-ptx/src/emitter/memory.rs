use std::fmt::Write as _;
use std::num::NonZeroU32;

use rustc_hash::FxHashSet;
use vyre_foundation::ir::DataType;
use vyre_lower::analyses::{
    derive_shared_access_profiles, select_bank_conflict_strategy, BankConflictMitigation,
    SharedBindingAccessProfile, TargetBankGeometry,
};
use vyre_lower::KernelDescriptor;
use vyre_lower::MemoryClass;

use super::BodyCtx;
use crate::patterns::ldmatrix_cp_async;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

pub(super) struct MemAddress {
    pub(super) space: &'static str,
    pub(super) operand: MemOperand,
}

#[derive(Clone, Copy)]
pub(super) enum MemOperand {
    Reg(Reg),
    RegOffset(Reg, u64),
    SharedSlotOffset(u32, u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AsyncCopyDirection {
    Load,
    Store,
}

impl BodyCtx<'_> {
    pub(super) fn emit_load_value(
        &mut self,
        address: MemAddress,
        load_space: &str,
        element_type: &DataType,
        elem_ty: PtxType,
    ) -> Result<Reg, EmitError> {
        match element_type {
            DataType::U8 => {
                let out = self.alloc(PtxType::U32);
                let _ = write!(self.text, "    ld.{load_space}.u8    {out}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                Ok(out)
            }
            DataType::I8 => {
                let out = self.alloc(PtxType::I32);
                let _ = write!(self.text, "    ld.{load_space}.s8    {out}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                Ok(out)
            }
            DataType::U16 => {
                let out = self.alloc(PtxType::U32);
                let _ = write!(self.text, "    ld.{load_space}.u16    {out}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                Ok(out)
            }
            DataType::I16 => {
                let out = self.alloc(PtxType::I32);
                let _ = write!(self.text, "    ld.{load_space}.s16    {out}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                Ok(out)
            }
            DataType::Bool => {
                let word = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::Bool);
                let _ = write!(self.text, "    ld.{load_space}.u32    {word}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                let _ = writeln!(self.text, "    setp.ne.u32    {out}, {word}, 0;");
                Ok(out)
            }
            DataType::F16 => {
                let packed = self.alloc(PtxType::B16);
                let out = self.alloc(PtxType::F32);
                let _ = write!(self.text, "    ld.{load_space}.b16    {packed}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                let _ = writeln!(self.text, "    cvt.f32.f16    {out}, {packed};");
                Ok(out)
            }
            DataType::BF16 => {
                let packed = self.alloc(PtxType::B16);
                let out = self.alloc(PtxType::F32);
                let _ = write!(self.text, "    ld.{load_space}.b16    {packed}, ");
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                let _ = writeln!(self.text, "    cvt.f32.bf16    {out}, {packed};");
                Ok(out)
            }
            _ => {
                let val_reg = self.alloc(elem_ty);
                let _ = write!(
                    self.text,
                    "    ld.{load_space}.{}    {val_reg}, ",
                    elem_ty.ptx_type_str(),
                );
                self.write_mem_operand(address.operand)?;
                self.text.push_str(";\n");
                Ok(self.canonicalize_f32(val_reg))
            }
        }
    }

    pub(super) fn emit_store_value(
        &mut self,
        guard: Option<(String, Reg)>,
        address: MemAddress,
        element_type: &DataType,
        value_reg: Reg,
    ) -> Result<(), EmitError> {
        match element_type {
            DataType::U8 | DataType::I8 => self.emit_raw_store(guard, address, "u8", value_reg),
            DataType::U16 | DataType::I16 => self.emit_raw_store(guard, address, "u16", value_reg),
            DataType::Bool => {
                let pred = self.pred_from_boolish(value_reg);
                let word = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    selp.u32    {word}, 1, 0, {pred};");
                self.emit_raw_store(guard, address, "u32", word)
            }
            DataType::F16 => {
                let f32_value = self.ensure_f32_store_operand(value_reg);
                let packed = self.alloc(PtxType::B16);
                let _ = writeln!(self.text, "    cvt.rn.f16.f32    {packed}, {f32_value};");
                self.emit_raw_store(guard, address, "b16", packed)
            }
            DataType::BF16 => {
                let f32_value = self.ensure_f32_store_operand(value_reg);
                let packed = self.alloc(PtxType::B16);
                let _ = writeln!(self.text, "    cvt.rn.bf16.f32    {packed}, {f32_value};");
                self.emit_raw_store(guard, address, "b16", packed)
            }
            _ => {
                let elem_ty = PtxType::from_dtype(element_type)?;
                let value_reg = if elem_ty == PtxType::F32 {
                    self.canonicalize_f32(value_reg)
                } else {
                    value_reg
                };
                self.emit_raw_store(guard, address, elem_ty.ptx_type_str(), value_reg)
            }
        }
    }

    fn ensure_f32_store_operand(&mut self, value_reg: Reg) -> Reg {
        match value_reg.0 {
            PtxType::F32 => value_reg,
            PtxType::I32 => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rn.f32.s32    {out}, {value_reg};");
                out
            }
            PtxType::Bool => {
                let word = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    selp.u32    {word}, 1, 0, {value_reg};");
                let _ = writeln!(self.text, "    cvt.rn.f32.u32    {out}, {word};");
                out
            }
            PtxType::B16 | PtxType::U32 => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rn.f32.u32    {out}, {value_reg};");
                out
            }
            PtxType::U64 => {
                let out = self.alloc(PtxType::F32);
                // PTX cvt.rn.f32.u64 converts the full 64-bit value to F32
                // with round-to-nearest. The previous two-step path
                // (cvt.u32.u64 + cvt.rn.f32.u32) silently discarded the high
                // 32 bits before conversion, a silent truncation of any value
                // > 0xFFFFFFFF.
                let _ = writeln!(self.text, "    cvt.rn.f32.u64    {out}, {value_reg};");
                out
            }
        }
    }

    fn emit_raw_store(
        &mut self,
        guard: Option<(String, Reg)>,
        address: MemAddress,
        ptx_type: &str,
        value_reg: Reg,
    ) -> Result<(), EmitError> {
        match guard {
            Some((guard, pred)) => {
                let _ = write!(
                    self.text,
                    "    {guard}{pred} st.{}.{}    ",
                    address.space, ptx_type,
                );
            }
            None => {
                let _ = write!(self.text, "    st.{}.{}    ", address.space, ptx_type);
            }
        }
        self.write_mem_operand(address.operand)?;
        let _ = writeln!(self.text, ", {value_reg};");
        Ok(())
    }

    pub(super) fn emit_global_address_operand(
        &mut self,
        binding_slot: u32,
        index_op_id: u32,
        element_type: &DataType,
    ) -> Result<MemOperand, EmitError> {
        let global_ptr =
            *self
                .slot_to_ptr
                .get(&binding_slot)
                .ok_or_else(|| EmitError::InvalidBinding {
                    slot: binding_slot,
                    reason: "global pointer not preloaded".into(),
                })?;
        if let Some(byte_offset) = self.immediate_byte_offset(index_op_id, element_type)? {
            return Ok(MemOperand::RegOffset(global_ptr, byte_offset));
        }
        let index_reg = self.lookup_operand(index_op_id)?;
        let stride = element_type
            .size_bytes()
            .ok_or_else(|| EmitError::UnsupportedDataType(format!("{element_type:?}")))?;
        let addr_reg = self.alloc(PtxType::U64);
        let _ = writeln!(
            self.text,
            "    mul.wide.u32    {addr_reg}, {index_reg}, {stride};"
        );
        let final_addr = self.alloc(PtxType::U64);
        let _ = writeln!(
            self.text,
            "    add.u64    {final_addr}, {global_ptr}, {addr_reg};"
        );
        Ok(MemOperand::Reg(final_addr))
    }

    pub(super) fn emit_memory_address(
        &mut self,
        binding_slot: u32,
        index_op_id: u32,
        element_type: &DataType,
        memory_class: MemoryClass,
    ) -> Result<MemAddress, EmitError> {
        // The immediate path folds an element index straight into a byte
        // offset, so a permuted binding folds the permuted index. When the
        // offset does not fit an immediate this falls through to the register
        // path, which permutes at the shared address site instead, so either
        // route rewrites the index exactly once.
        if let Some(index) = self.u32_literals.get(&index_op_id).copied() {
            let addressed = match self.shared_permutation(binding_slot, memory_class) {
                Some(permutation) => permutation.apply(index),
                None => index,
            };
            if let Some(byte_offset) = self.byte_offset_of(addressed, element_type)? {
                return self.emit_memory_address_immediate(binding_slot, byte_offset, memory_class);
            }
        }
        let index_reg = self.lookup_operand(index_op_id)?;
        self.emit_memory_address_from_index_reg(binding_slot, index_reg, element_type, memory_class)
    }

    fn emit_memory_address_immediate(
        &self,
        binding_slot: u32,
        byte_offset: u64,
        memory_class: MemoryClass,
    ) -> Result<MemAddress, EmitError> {
        match memory_class {
            MemoryClass::Global | MemoryClass::Constant | MemoryClass::Uniform => {
                let global_ptr = *self.slot_to_ptr.get(&binding_slot).ok_or_else(|| {
                    EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "global pointer not preloaded".into(),
                    }
                })?;
                Ok(MemAddress {
                    space: "global",
                    operand: MemOperand::RegOffset(global_ptr, byte_offset),
                })
            }
            MemoryClass::Shared => {
                if !self.slot_to_shared_symbol.contains_key(&binding_slot) {
                    return Err(EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "shared symbol not allocated".into(),
                    });
                }
                Ok(MemAddress {
                    space: "shared",
                    operand: MemOperand::SharedSlotOffset(binding_slot, byte_offset),
                })
            }
            MemoryClass::Scratch => Err(EmitError::InvalidBinding {
                slot: binding_slot,
                reason: "scratch bindings must be resolved before PTX emission".into(),
            }),
        }
    }

    fn immediate_byte_offset(
        &self,
        index_op_id: u32,
        element_type: &DataType,
    ) -> Result<Option<u64>, EmitError> {
        let Some(index) = self.u32_literals.get(&index_op_id).copied() else {
            return Ok(None);
        };
        self.byte_offset_of(index, element_type)
    }

    /// Byte offset of element `index`, or `None` when it does not fit the
    /// immediate field an address operand carries.
    fn byte_offset_of(
        &self,
        index: u32,
        element_type: &DataType,
    ) -> Result<Option<u64>, EmitError> {
        let stride = element_type
            .size_bytes()
            .ok_or_else(|| EmitError::UnsupportedDataType(format!("{element_type:?}")))?;
        let Some(byte_offset) = u64::from(index).checked_mul(stride as u64) else {
            return Ok(None);
        };
        if byte_offset <= i32::MAX as u64 {
            Ok(Some(byte_offset))
        } else {
            Ok(None)
        }
    }

    pub(super) fn emit_memory_address_from_index_reg(
        &mut self,
        binding_slot: u32,
        index_reg: Reg,
        element_type: &DataType,
        memory_class: MemoryClass,
    ) -> Result<MemAddress, EmitError> {
        match memory_class {
            MemoryClass::Global => {
                let global_ptr = *self.slot_to_ptr.get(&binding_slot).ok_or_else(|| {
                    EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "global pointer not preloaded".into(),
                    }
                })?;
                let safe_index = self.clamp_index_to_buffer_length(binding_slot, index_reg);
                let byte_offset = self.emit_byte_offset(safe_index, element_type)?;
                let reg = self.alloc(PtxType::U64);
                let _ = writeln!(
                    self.text,
                    "    add.u64    {reg}, {global_ptr}, {byte_offset};"
                );
                Ok(MemAddress {
                    space: "global",
                    operand: MemOperand::Reg(reg),
                })
            }
            MemoryClass::Constant | MemoryClass::Uniform => {
                let global_ptr = *self.slot_to_ptr.get(&binding_slot).ok_or_else(|| {
                    EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "constant/uniform pointer not preloaded".into(),
                    }
                })?;
                let safe_index = self.clamp_index_to_buffer_length(binding_slot, index_reg);
                let byte_offset = self.emit_byte_offset(safe_index, element_type)?;
                let reg = self.alloc(PtxType::U64);
                let _ = writeln!(
                    self.text,
                    "    add.u64    {reg}, {global_ptr}, {byte_offset};"
                );
                Ok(MemAddress {
                    space: "global",
                    operand: MemOperand::Reg(reg),
                })
            }
            MemoryClass::Shared => {
                let symbol = self
                    .slot_to_shared_symbol
                    .get(&binding_slot)
                    .cloned()
                    .ok_or_else(|| EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "shared symbol not allocated".into(),
                    })?;
                let byte_offset =
                    self.emit_shared_byte_offset(binding_slot, index_reg, element_type)?;
                let base = self.alloc(PtxType::U32);
                let addr = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    mov.u32    {base}, {symbol};");
                let _ = writeln!(self.text, "    add.u32    {addr}, {base}, {byte_offset};");
                Ok(MemAddress {
                    space: "shared",
                    operand: MemOperand::Reg(addr),
                })
            }
            MemoryClass::Scratch => Err(EmitError::InvalidBinding {
                slot: binding_slot,
                reason: "scratch bindings must be resolved before PTX emission".into(),
            }),
        }
    }

    /// Clamp a runtime index register so the resulting address stays
    /// inside the buffer. PTX has no built-in bounds checking; without
    /// this, speculative loads emitted by `Expr::select` arms can fault
    /// with `CUDA_ERROR_ILLEGAL_ADDRESS`. WGSL/naga does an equivalent
    /// clamp via its bounds-check policy  -  this matches the contract
    /// across backends.
    ///
    /// Lowered as: `safe = (idx < len) ? idx : 0`. When `len == 0` the
    /// dispatcher rejects the launch upstream, so the `0` fallback
    /// always points at a valid byte.
    ///
    /// This makes an out-of-range access fault-free, not correct. The folded
    /// address is element 0, which a load may read and discard and a store may
    /// not write: a store carries its own bounds predicate
    /// (`store_guard::store_guard_for_index`) so the write is dropped rather
    /// than redirected onto a live element.
    fn clamp_index_to_buffer_length(&mut self, binding_slot: u32, raw_idx: Reg) -> Reg {
        let len_reg = self.ensure_buffer_length_reg(binding_slot);
        let in_bounds = self.alloc(PtxType::Bool);
        let safe_idx = self.alloc(PtxType::U32);
        let zero = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    mov.u32    {zero}, 0;");
        let _ = writeln!(
            self.text,
            "    setp.lt.u32    {in_bounds}, {raw_idx}, {len_reg};"
        );
        let _ = writeln!(
            self.text,
            "    selp.u32    {safe_idx}, {raw_idx}, {zero}, {in_bounds};"
        );
        safe_idx
    }

    pub(super) fn emit_index_in_bounds_pred(
        &mut self,
        binding_slot: u32,
        index_op_id: u32,
    ) -> Result<Reg, EmitError> {
        let raw_idx = self.lookup_operand(index_op_id)?;
        Ok(self.emit_index_reg_in_bounds_pred(binding_slot, raw_idx))
    }

    pub(super) fn emit_index_reg_in_bounds_pred(&mut self, binding_slot: u32, raw_idx: Reg) -> Reg {
        let len_reg = self.ensure_buffer_length_reg(binding_slot);
        let in_bounds = self.alloc(PtxType::Bool);
        let _ = writeln!(
            self.text,
            "    setp.lt.u32    {in_bounds}, {raw_idx}, {len_reg};"
        );
        in_bounds
    }

    /// `reg = operand(index_op_id) + offset`, or the operand itself when
    /// `offset` is zero.
    pub(super) fn emit_index_plus_immediate(
        &mut self,
        index_op_id: u32,
        offset: u32,
    ) -> Result<Reg, EmitError> {
        let base = self.lookup_operand(index_op_id)?;
        if offset == 0 {
            return Ok(base);
        }
        let reg = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    add.u32    {reg}, {base}, {offset};");
        Ok(reg)
    }

    fn ensure_buffer_length_reg(&mut self, binding_slot: u32) -> Reg {
        if let Some(&reg) = self.slot_to_length_reg.get(&binding_slot) {
            return reg;
        }
        let reg = self.alloc(PtxType::U32);
        // All valid slots must be registered by `preload_bindings`, which
        // performs checked arithmetic and returns Err on overflow. Reaching
        // this branch at all indicates a caller bug: a slot that was never
        // preloaded is being used at emit time.
        //
        // Law 10: we must NEVER silently emit a plausible-looking load when
        // the offset computation would overflow. We emit `trap;` first so the
        // thread terminates loudly rather than reading a garbage length
        // register and producing silently wrong bounds-check predicates.
        match binding_slot.checked_mul(4).and_then(|v| v.checked_add(4)) {
            None => {
                // Overflow: the slot cannot produce a valid params-buffer
                // offset. Emit a hard trap so the kernel fails closed instead
                // of reading a u32::MAX-offset address and silently using the
                // garbage as a bounds-check length.
                let _ = writeln!(
                    self.text,
                    "    // BUG: slot={binding_slot} byte offset overflows u32; \
                     preload_bindings should have rejected this slot, killing thread"
                );
                let _ = writeln!(self.text, "    trap;");
            }
            Some(byte_offset) => {
                let _ = writeln!(
                    self.text,
                    "    ld.global.ca.u32    {reg}, [%rd0 + {byte_offset}];"
                );
            }
        }
        self.slot_to_length_reg.insert(binding_slot, reg);
        reg
    }

    fn emit_byte_offset(
        &mut self,
        index_reg: Reg,
        element_type: &DataType,
    ) -> Result<Reg, EmitError> {
        let stride = element_type
            .size_bytes()
            .ok_or_else(|| EmitError::UnsupportedDataType(format!("{element_type:?}")))?;
        let byte_offset = self.alloc(PtxType::U64);
        let _ = writeln!(
            self.text,
            "    mul.wide.u32    {byte_offset}, {index_reg}, {stride};"
        );
        Ok(byte_offset)
    }

    fn emit_shared_byte_offset(
        &mut self,
        binding_slot: u32,
        index_reg: Reg,
        element_type: &DataType,
    ) -> Result<Reg, EmitError> {
        let stride = element_type
            .size_bytes()
            .ok_or_else(|| EmitError::UnsupportedDataType(format!("{element_type:?}")))?;
        let index_reg = self.emit_shared_permutation(binding_slot, index_reg);
        let byte_offset = self.alloc(PtxType::U32);
        let _ = writeln!(
            self.text,
            "    mul.lo.u32    {byte_offset}, {index_reg}, {stride};"
        );
        Ok(byte_offset)
    }

    pub(super) fn write_mem_operand(&mut self, operand: MemOperand) -> Result<(), EmitError> {
        match operand {
            MemOperand::Reg(reg) | MemOperand::RegOffset(reg, 0) => {
                let _ = write!(self.text, "[{reg}]");
            }
            MemOperand::RegOffset(reg, offset) => {
                let _ = write!(self.text, "[{reg}+{offset}]");
            }
            MemOperand::SharedSlotOffset(slot, 0) => {
                let symbol = self.slot_to_shared_symbol.get(&slot).ok_or_else(|| {
                    EmitError::InvalidBinding {
                        slot,
                        reason: "shared symbol not allocated".into(),
                    }
                })?;
                let _ = write!(self.text, "[{symbol}]");
            }
            MemOperand::SharedSlotOffset(slot, offset) => {
                let symbol = self.slot_to_shared_symbol.get(&slot).ok_or_else(|| {
                    EmitError::InvalidBinding {
                        slot,
                        reason: "shared symbol not allocated".into(),
                    }
                })?;
                let _ = write!(self.text, "[{symbol}+{offset}]");
            }
        }
        Ok(())
    }

    pub(super) fn require_u32_slot(
        &self,
        slot: u32,
        context: &str,
    ) -> Result<(DataType, MemoryClass), EmitError> {
        let binding = self.binding_for_slot(slot)?;
        if binding.element_type != DataType::U32 {
            return Err(EmitError::InvalidBinding {
                slot,
                reason: format!("{context} must be a U32 binding"),
            });
        }
        Ok((binding.element_type.clone(), binding.memory_class))
    }
}

/// Shared-memory bank geometry, as every CUDA target since Kepler reports it:
/// thirty-two four-byte banks, thirty-two lanes to a warp, four-byte native
/// access width.
const fn bank_geometry() -> TargetBankGeometry {
    TargetBankGeometry {
        bank_count: 32,
        bank_width_bytes: 4,
        subgroup_lanes: 32,
        instruction_word_bytes: 4,
    }
}

/// A bijective rewrite of a shared binding's element index.
///
/// Bijective is the whole requirement: two lanes whose element indices differ
/// must still differ after the rewrite, or the kernel computes different values
/// than the unpermuted one. Both arms are proven one-to-one over the binding's
/// element range by [`shared_permutation_for`], which refuses a strategy whose
/// preconditions the binding does not meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SharedPermutation {
    /// `(index >> row_log2) * (row + pad) + (index & (row - 1))`, which spreads
    /// each row of a tile across one more bank than it occupied.
    PadRows {
        /// Log2 of the unpadded row length in elements.
        row_log2: u32,
        /// Elements added per row.
        pad_elements: u32,
    },
    /// `index ^ ((index >> stride_shift) & mask)`, which rewrites only bits
    /// below the swizzle width and so keeps every index inside its own aligned
    /// block.
    XorSwizzle {
        /// Shift selecting the bits mixed into the low bits.
        stride_shift: u32,
        /// Mask of the low bits the swizzle rewrites.
        mask: u32,
    },
}

impl SharedPermutation {
    /// The element index this permutation maps `index` to.
    pub(super) fn apply(self, index: u32) -> u32 {
        match self {
            Self::PadRows {
                row_log2,
                pad_elements,
            } => {
                let row = 1_u32 << row_log2;
                (index >> row_log2)
                    .saturating_mul(row.saturating_add(pad_elements))
                    .saturating_add(index & (row - 1))
            }
            Self::XorSwizzle { stride_shift, mask } => index ^ ((index >> stride_shift) & mask),
        }
    }

    /// Elements the allocation needs so every permuted index stays in range.
    pub(super) fn extent(self, element_count: u32) -> u32 {
        match self {
            Self::PadRows {
                row_log2,
                pad_elements,
            } => {
                let row = 1_u32 << row_log2;
                (element_count >> row_log2).saturating_mul(row.saturating_add(pad_elements))
            }
            Self::XorSwizzle { .. } => element_count,
        }
    }
}

/// The permutation that applies `strategy` to `profile`, when it is one-to-one
/// over the binding's element range.
///
/// `None` keeps the unpermuted index. A strategy is refused rather than
/// approximated: an index rewrite that is not a bijection, or that can leave
/// the allocation, is a wrong kernel, not a slower one.
fn shared_permutation_for(
    profile: &SharedBindingAccessProfile,
    strategy: BankConflictMitigation,
) -> Option<SharedPermutation> {
    match strategy {
        BankConflictMitigation::NoRewrite => None,
        BankConflictMitigation::PadLines {
            pad_elements_per_row,
        } => {
            if pad_elements_per_row == 0 {
                return None;
            }
            // Padding is priced as stride plus pad, which is a row length only
            // when one row explains every strided phase. Two disagreeing
            // strides mean the price and the rewrite describe different tiles.
            let mut row: Option<u32> = None;
            for phase in &profile.phases {
                if phase.stride_elements <= 1 {
                    continue;
                }
                match row {
                    None => row = Some(phase.stride_elements),
                    Some(seen) if seen == phase.stride_elements => {}
                    Some(_) => return None,
                }
            }
            let row = row?;
            if !row.is_power_of_two()
                || row < 2
                || profile.element_count == 0
                || profile.element_count % row != 0
            {
                return None;
            }
            Some(SharedPermutation::PadRows {
                row_log2: row.trailing_zeros(),
                pad_elements: pad_elements_per_row,
            })
        }
        BankConflictMitigation::XorSwizzle {
            swizzle_bits,
            stride_shift,
        } => {
            // The rewritten bits have to sit strictly below the bits the shift
            // reads, or the extracted value depends on what the XOR changed and
            // the map stops being invertible.
            if swizzle_bits == 0 || swizzle_bits > 5 || stride_shift < swizzle_bits {
                return None;
            }
            let block = 1_u32 << swizzle_bits;
            if profile.element_count == 0 || profile.element_count % block != 0 {
                return None;
            }
            Some(SharedPermutation::XorSwizzle {
                stride_shift,
                mask: block - 1,
            })
        }
    }
}

impl BodyCtx<'_> {
    /// Choose one index permutation per permutable shared binding.
    ///
    /// Called before any shared declaration is written, because a padded
    /// binding is declared at its grown extent. A binding is permutable only
    /// when the neutral derivation proved every access to it is a scalar load
    /// or store with a known stride, and no fused bulk copy claims it: both
    /// route around the single address site the rewrite happens at.
    pub(super) fn plan_shared_permutations(&mut self, desc: &KernelDescriptor) {
        let geometry = bank_geometry();
        let Some(banks) = NonZeroU32::new(geometry.bank_count) else {
            return;
        };
        let bulk_copy_slots: FxHashSet<u32> = ldmatrix_cp_async::analyze(desc, self.options.target)
            .candidates
            .iter()
            .map(|candidate| candidate.shared_binding_slot)
            .collect();
        for profile in derive_shared_access_profiles(desc, banks) {
            if profile.blocked_by.is_some() || bulk_copy_slots.contains(&profile.binding_slot) {
                continue;
            }
            let selection = select_bank_conflict_strategy(&profile.phases, &geometry);
            if !selection.accepted {
                continue;
            }
            if let Some(permutation) = shared_permutation_for(&profile, selection.strategy) {
                self.slot_to_shared_permutation
                    .insert(profile.binding_slot, permutation);
            }
        }
    }

    /// The permutation chosen for `binding_slot`, if it is a shared binding
    /// and one was chosen.
    pub(super) fn shared_permutation(
        &self,
        binding_slot: u32,
        memory_class: MemoryClass,
    ) -> Option<SharedPermutation> {
        if !matches!(memory_class, MemoryClass::Shared) {
            return None;
        }
        self.slot_to_shared_permutation.get(&binding_slot).copied()
    }

    /// Rewrite a shared element index through the binding's permutation.
    ///
    /// Returns `index_reg` unchanged when the binding has none, so a kernel
    /// with nothing to permute emits exactly the instructions it did before.
    fn emit_shared_permutation(&mut self, binding_slot: u32, index_reg: Reg) -> Reg {
        let Some(permutation) = self.slot_to_shared_permutation.get(&binding_slot).copied() else {
            return index_reg;
        };
        match permutation {
            SharedPermutation::PadRows {
                row_log2,
                pad_elements,
            } => {
                let row = 1_u32 << row_log2;
                let padded = row.saturating_add(pad_elements);
                let rows = self.alloc(PtxType::U32);
                let offset = self.alloc(PtxType::U32);
                let scaled = self.alloc(PtxType::U32);
                let permuted = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    shr.u32    {rows}, {index_reg}, {row_log2};");
                let _ = writeln!(
                    self.text,
                    "    and.b32    {offset}, {index_reg}, {};",
                    row - 1
                );
                let _ = writeln!(self.text, "    mul.lo.u32    {scaled}, {rows}, {padded};");
                let _ = writeln!(self.text, "    add.u32    {permuted}, {scaled}, {offset};");
                permuted
            }
            SharedPermutation::XorSwizzle { stride_shift, mask } => {
                let high = self.alloc(PtxType::U32);
                let bits = self.alloc(PtxType::U32);
                let permuted = self.alloc(PtxType::U32);
                let _ = writeln!(
                    self.text,
                    "    shr.u32    {high}, {index_reg}, {stride_shift};"
                );
                let _ = writeln!(self.text, "    and.b32    {bits}, {high}, {mask};");
                let _ = writeln!(self.text, "    xor.b32    {permuted}, {index_reg}, {bits};");
                permuted
            }
        }
    }
}

// Inline: `SharedPermutation` and `shared_permutation_for` are crate-private,
// and the property they carry is arithmetic. Whether the emitted kernel applies
// them is proven from the emitted text in
// `tests/regression_emit_fixes.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_lower::analyses::{AccessPhase, AccessPhaseProfile};

    /// A profile stating one phase per entry of `strides`, all full width.
    fn profile(element_count: u32, strides: &[u32]) -> SharedBindingAccessProfile {
        SharedBindingAccessProfile {
            binding_slot: 1,
            element_count,
            phases: strides
                .iter()
                .map(|stride| AccessPhaseProfile {
                    phase: AccessPhase::ComputeRead,
                    stride_elements: *stride,
                    active_threads: 32,
                    access_weight: 1,
                })
                .collect(),
            blocked_by: None,
        }
    }

    /// Two lanes whose element indices differ must still differ after the
    /// rewrite, and no rewritten index may leave the allocation. Either failure
    /// is a kernel that computes different values than the unpermuted one, so
    /// both arms are checked over the whole declared range rather than sampled.
    #[test]
    fn both_permutations_are_one_to_one_inside_the_extent_they_declare() {
        let cases = [
            (
                SharedPermutation::PadRows {
                    row_log2: 5,
                    pad_elements: 1,
                },
                1024_u32,
            ),
            (
                SharedPermutation::PadRows {
                    row_log2: 3,
                    pad_elements: 4,
                },
                256,
            ),
            (
                SharedPermutation::XorSwizzle {
                    stride_shift: 3,
                    mask: 3,
                },
                1024,
            ),
            (
                SharedPermutation::XorSwizzle {
                    stride_shift: 5,
                    mask: 7,
                },
                2048,
            ),
        ];
        for (permutation, element_count) in cases {
            let extent = permutation.extent(element_count);
            let mut seen = FxHashSet::default();
            for index in 0..element_count {
                let mapped = permutation.apply(index);
                assert!(
                    mapped < extent,
                    "{permutation:?} maps {index} to {mapped}, outside an extent of {extent}"
                );
                assert!(
                    seen.insert(mapped),
                    "{permutation:?} maps two indices to {mapped}"
                );
            }
        }
    }

    /// The chain from bank geometry to emitted rewrite. A column walk over 32
    /// four-byte banks is a 32-way conflict, one element of padding per row is
    /// the cheapest accepted candidate, and that candidate becomes a row
    /// permutation over 32-element rows.
    #[test]
    fn a_column_walk_selects_one_element_of_padding_per_row() {
        let profile = profile(1024, &[32, 32]);
        let selection = select_bank_conflict_strategy(&profile.phases, &bank_geometry());
        assert!(selection.accepted);
        assert_eq!(
            selection.strategy,
            BankConflictMitigation::PadLines {
                pad_elements_per_row: 1
            }
        );
        assert_eq!(
            shared_permutation_for(&profile, selection.strategy),
            Some(SharedPermutation::PadRows {
                row_log2: 5,
                pad_elements: 1
            })
        );
    }

    /// A strategy is refused rather than approximated. Every precondition is
    /// checked against a binding that misses exactly that one, because a
    /// rewrite that is not a bijection, or that can leave the allocation, is a
    /// wrong kernel rather than a slower one.
    #[test]
    fn a_strategy_the_binding_cannot_carry_is_refused() {
        let pad = |pad_elements_per_row| BankConflictMitigation::PadLines {
            pad_elements_per_row,
        };
        let swizzle = |swizzle_bits, stride_shift| BankConflictMitigation::XorSwizzle {
            swizzle_bits,
            stride_shift,
        };
        let refused = [
            // Nothing to pad by.
            (profile(1024, &[32]), pad(0)),
            // Two strides mean the price and the rewrite describe different
            // tiles, so neither row length is the one that was ranked.
            (profile(1024, &[32, 16]), pad(1)),
            // A row length that is not a power of two has no shift form.
            (profile(1024, &[24]), pad(1)),
            // No strided phase states a row length at all.
            (profile(1024, &[1]), pad(1)),
            // The extent is not a whole number of rows, so the last row would
            // be padded past the allocation.
            (profile(1000, &[32]), pad(1)),
            (profile(0, &[32]), pad(1)),
            // The rewritten bits are not strictly below the bits the shift
            // reads, so the extracted value depends on what the XOR changed.
            (profile(1024, &[32]), swizzle(3, 2)),
            (profile(1024, &[32]), swizzle(0, 4)),
            (profile(1024, &[32]), swizzle(6, 8)),
            // The extent is not a whole number of swizzle blocks.
            (profile(12, &[32]), swizzle(3, 4)),
        ];
        for (binding, strategy) in refused {
            assert_eq!(
                shared_permutation_for(&binding, strategy),
                None,
                "{strategy:?} must be refused for a binding of {} elements with \
                 strides {:?}",
                binding.element_count,
                binding
                    .phases
                    .iter()
                    .map(|phase| phase.stride_elements)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            shared_permutation_for(&profile(1024, &[32]), BankConflictMitigation::NoRewrite),
            None
        );
    }
}
