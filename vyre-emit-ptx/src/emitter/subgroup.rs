//! Subgroup reduction lowering.
//!
//! Owns the butterfly `shfl.sync.bfly` sequences that reduce a value across
//! the active subgroup and the lane mask those shuffles are issued with.
//! Ballot, shuffle, and broadcast are single instructions and are emitted
//! from the op dispatch directly.

use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;
use std::fmt::Write as _;
use vyre_foundation::ir::SubgroupReduceOp;

impl BodyCtx<'_> {
    pub(super) fn subgroup_lane_mask(&self) -> u32 {
        self.options.subgroup_size.saturating_sub(1)
    }

    pub(super) fn emit_subgroup_reduce(
        &mut self,
        op: SubgroupReduceOp,
        value: Reg,
    ) -> Result<Reg, EmitError> {
        if value.0 == PtxType::F32 {
            return self.emit_f32_subgroup_reduce(op, value);
        }
        // redux.sync handles add/min/max with the signed/unsigned type and
        // and/or/xor as bitwise `.b32`. It has no integer product reduction.
        let (op_str, type_str) = match op {
            SubgroupReduceOp::Add => ("add", value.0.ptx_type_str()),
            SubgroupReduceOp::Min => ("min", value.0.ptx_type_str()),
            SubgroupReduceOp::Max => ("max", value.0.ptx_type_str()),
            SubgroupReduceOp::And => ("and", "b32"),
            SubgroupReduceOp::Or => ("or", "b32"),
            SubgroupReduceOp::Xor => ("xor", "b32"),
            SubgroupReduceOp::Mul => {
                // No `redux.sync` integer product; reduce with the same
                // shfl.idx XOR butterfly the f32 path uses. Integers shuffle
                // directly (no f32<->b32 bitcast). `mul.lo` keeps the low 32
                // bits (matching the reference oracle's `wrapping_mul`).
                let combine = match value.0 {
                    PtxType::U32 => "mul.lo.u32",
                    PtxType::I32 => "mul.lo.s32",
                    other_ty => {
                        return Err(EmitError::InvalidDescriptor(format!(
                            "PTX subgroup product reduction needs a 32-bit integer operand, got {other_ty:?}. Fix: use u32/i32 (or an f32 operand)."
                        )))
                    }
                };
                return Ok(self.emit_subgroup_xor_butterfly(value, value.0, combine));
            }
            other => {
                return Err(EmitError::InvalidDescriptor(format!(
                    "PTX subgroup reduction operator {other:?} is not supported. Fix: extend emit_subgroup_reduce."
                )))
            }
        };
        let result = self.alloc(value.0);
        let mask = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    activemask.b32    {mask};");
        let _ = writeln!(
            self.text,
            "    redux.sync.{op_str}.{type_str}    {result}, {value}, {mask};"
        );
        Ok(result)
    }

    fn emit_f32_subgroup_reduce(
        &mut self,
        op: SubgroupReduceOp,
        value: Reg,
    ) -> Result<Reg, EmitError> {
        // f32 has no redux.sync; reduce with a shfl XOR butterfly whose combine
        // instruction is the reduction operator. Bitwise ops are undefined on
        // floats and fail closed.
        let combine = match op {
            SubgroupReduceOp::Add => "add.f32",
            SubgroupReduceOp::Mul => "mul.f32",
            SubgroupReduceOp::Min => "min.f32",
            SubgroupReduceOp::Max => "max.f32",
            SubgroupReduceOp::And | SubgroupReduceOp::Or | SubgroupReduceOp::Xor => {
                return Err(EmitError::InvalidDescriptor(
                    "subgroup bitwise reduction (and/or/xor) is undefined on f32. Fix: use an integer operand.".into(),
                ))
            }
            other => {
                return Err(EmitError::InvalidDescriptor(format!(
                    "PTX f32 subgroup reduction operator {other:?} is not supported. Fix: extend emit_f32_subgroup_reduce."
                )))
            }
        };
        Ok(self.emit_subgroup_xor_butterfly(value, PtxType::F32, combine))
    }

    /// XOR (butterfly) all-reduce shared by the f32 and integer-product paths.
    ///
    /// Every lane exchanges with `lane ^ offset` and applies `combine`, so after
    /// log2(width) steps EVERY lane holds the full reduction, matching the
    /// all-lane-broadcast contract of WGSL `subgroupAdd` / `redux.sync` / the
    /// reference oracle. (A `shfl.down` tree feeds only lane 0, which violates
    /// that contract for lanes 1..; this is verified all-lane on a live sm_120
    /// GPU by `vyre-driver-cuda`'s `subgroup_reduce_gpu_parity`.)
    ///
    /// The exchange uses `.idx` mode with an explicit `laneid ^ offset` source
    /// (the same instruction the verified `subgroup_shuffle` path uses), with
    /// `c = subgroup_size - 1` (the standard full-warp clamp, segmask=0). The
    /// XOR partner is always in range for a power-of-two width, so no
    /// predication is needed. `combine` is the full typed PTX op
    /// (`add.f32` / `max.f32` / `mul.lo.u32` / ...).
    fn emit_subgroup_xor_butterfly(&mut self, value: Reg, acc_ty: PtxType, combine: &str) -> Reg {
        let is_f32 = acc_ty == PtxType::F32;
        let acc = self.alloc(acc_ty);
        let mask = self.alloc(PtxType::U32);
        let lane = self.alloc(PtxType::U32);
        let idx_clamp = self.subgroup_lane_mask();
        // `mov.b32` is a width-preserving bit copy valid for f32 and integer
        // 32-bit registers alike.
        let _ = writeln!(self.text, "    mov.b32    {acc}, {value};");
        let _ = writeln!(self.text, "    activemask.b32    {mask};");
        let _ = writeln!(self.text, "    mov.u32    {lane}, %laneid;");

        let mut offset = self.options.subgroup_size / 2;
        while offset > 0 {
            let src_lane = self.alloc(PtxType::U32);
            let _ = writeln!(self.text, "    xor.b32    {src_lane}, {lane}, {offset};");
            if is_f32 {
                // Cross-lane shfl operates on `.b32`; bitcast f32<->b32 around it.
                let bits = self.alloc(PtxType::U32);
                let shuffled_bits = self.alloc(PtxType::U32);
                let shuffled = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    mov.b32    {bits}, {acc};");
                let _ = writeln!(
                    self.text,
                    "    shfl.sync.idx.b32    {shuffled_bits}, {bits}, {src_lane}, 0x{idx_clamp:x}, {mask};"
                );
                let _ = writeln!(self.text, "    mov.b32    {shuffled}, {shuffled_bits};");
                let _ = writeln!(self.text, "    {combine}    {acc}, {acc}, {shuffled};");
            } else {
                // Integer registers shuffle directly.
                let shuffled = self.alloc(acc_ty);
                let _ = writeln!(
                    self.text,
                    "    shfl.sync.idx.b32    {shuffled}, {acc}, {src_lane}, 0x{idx_clamp:x}, {mask};"
                );
                let _ = writeln!(self.text, "    {combine}    {acc}, {acc}, {shuffled};");
            }
            offset /= 2;
        }
        acc
    }
}
