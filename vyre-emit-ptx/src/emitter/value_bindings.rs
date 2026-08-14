//! Descriptor operand id to PTX register bindings.
//!
//! Owns the map between a descriptor's SSA namespace and the registers that
//! hold those values: allocating a register for a literal, resolving an
//! operand id to the register that carries it, and recording the register a
//! result id now names. It emits no instruction other than the literal
//! materialization it must allocate for.

use vyre_lower::{KernelOp, LiteralValue};

use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

impl BodyCtx<'_> {
    pub(super) fn alloc_literal(&mut self, lit: &LiteralValue) -> (Reg, String) {
        match lit {
            LiteralValue::U32(v) => {
                let reg = self.alloc(PtxType::U32);
                (reg, format!("{v}"))
            }
            LiteralValue::I32(v) => {
                let reg = self.alloc(PtxType::I32);
                (reg, format!("{v}"))
            }
            LiteralValue::F32(v) => {
                let reg = self.alloc(PtxType::F32);
                let bits = v.to_bits();
                (reg, format!("0f{bits:08X}"))
            }
            LiteralValue::Bool(v) => {
                let reg = self.alloc(PtxType::Bool);
                (reg, if *v { "1".into() } else { "0".into() })
            }
        }
    }

    pub(super) fn lookup_operand(&self, op_id: u32) -> Result<Reg, EmitError> {
        self.operand_to_reg.get(&op_id).copied().ok_or_else(|| {
            EmitError::InvalidDescriptor(format!("operand id {op_id} not yet emitted"))
        })
    }

    pub(super) fn bind_result(&mut self, op: &KernelOp, reg: Reg) -> Result<(), EmitError> {
        if let Some(result) = op.result {
            self.operand_to_reg.insert(result, reg);
        }
        Ok(())
    }

    pub(super) fn bind_consecutive_results(
        &mut self,
        op: &KernelOp,
        regs: &[Reg],
    ) -> Result<(), EmitError> {
        let Some(base) = op.result else {
            return Err(EmitError::InvalidDescriptor(
                "MatrixMma missing base result id. Fix: set result to the first accumulator id."
                    .into(),
            ));
        };
        for (offset, reg) in regs.iter().copied().enumerate() {
            let id = base.checked_add(offset as u32).ok_or_else(|| {
                EmitError::InvalidDescriptor(
                    "MatrixMma result id range overflows u32. Fix: allocate a lower base id."
                        .into(),
                )
            })?;
            self.operand_to_reg.insert(id, reg);
        }
        Ok(())
    }
}
