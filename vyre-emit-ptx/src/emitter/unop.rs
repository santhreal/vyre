//! Unary-operator instruction selection.
//!
//! Owns the `UnOp` to PTX mapping for every register class, plus the
//! diagnostic spelling used when an operator has no lowering for the class
//! it was handed. Casts are a separate concept and live in `cast`.

use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;
use std::fmt::Write as _;
use vyre_foundation::ir::UnOp;

fn unop_name(op: &UnOp) -> &'static str {
    match op {
        UnOp::Negate => "negate",
        UnOp::LogicalNot => "logical_not",
        UnOp::BitNot => "bit_not",
        UnOp::Abs => "abs",
        UnOp::Sqrt => "sqrt",
        UnOp::InverseSqrt => "inverse_sqrt",
        UnOp::Reciprocal => "reciprocal",
        UnOp::Exp => "exp",
        UnOp::Log => "log",
        UnOp::Exp2 => "exp2",
        UnOp::Log2 => "log2",
        UnOp::Sin => "sin",
        UnOp::Cos => "cos",
        UnOp::Tanh => "tanh",
        UnOp::Floor => "floor",
        UnOp::Ceil => "ceil",
        UnOp::Round => "round",
        UnOp::Trunc => "trunc",
        UnOp::Popcount => "popcount",
        UnOp::Clz => "clz",
        UnOp::Ctz => "ctz",
        UnOp::ReverseBits => "reverse_bits",
        UnOp::IsNan => "is_nan",
        UnOp::IsInf => "is_inf",
        UnOp::IsFinite => "is_finite",
        _ => "unknown",
    }
}

impl BodyCtx<'_> {
    pub(super) fn emit_unop(&mut self, op: &UnOp, operand: Reg) -> Result<Reg, EmitError> {
        let out = match (op, operand.0) {
            (UnOp::Negate, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    neg.f32    {out}, {operand};");
                out
            }
            (UnOp::Negate, _) => {
                let out = self.alloc(PtxType::I32);
                let _ = writeln!(self.text, "    neg.s32    {out}, {operand};");
                out
            }
            (UnOp::BitNot, _) => {
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    not.b32    {out}, {operand};");
                out
            }
            (UnOp::LogicalNot, _) => {
                let out = self.alloc(PtxType::Bool);
                if operand.0 == PtxType::Bool {
                    let _ = writeln!(self.text, "    not.pred    {out}, {operand};");
                } else {
                    let _ = writeln!(self.text, "    setp.eq.u32    {out}, {operand}, 0;");
                }
                out
            }
            (UnOp::Abs, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    abs.f32    {out}, {operand};");
                out
            }
            (UnOp::Abs, _) => {
                let out = self.alloc(PtxType::I32);
                let _ = writeln!(self.text, "    abs.s32    {out}, {operand};");
                out
            }
            (UnOp::Sqrt, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    sqrt.rn.f32    {out}, {operand};");
                out
            }
            (UnOp::InverseSqrt, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                if self.options.ulp_budget.is_some_and(|budget| budget > 0) {
                    let _ = writeln!(self.text, "    rsqrt.approx.f32    {out}, {operand};");
                } else {
                    let _ = writeln!(self.text, "    sqrt.rn.f32    {out}, {operand};");
                    let _ = writeln!(self.text, "    rcp.rn.f32    {out}, {out};");
                }
                out
            }
            (UnOp::Reciprocal, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                if self.options.ulp_budget.is_some_and(|budget| budget > 0) {
                    let _ = writeln!(self.text, "    rcp.approx.f32    {out}, {operand};");
                } else {
                    let _ = writeln!(self.text, "    rcp.rn.f32    {out}, {operand};");
                }
                out
            }
            (UnOp::Tanh, PtxType::F32) => {
                if !self.options.ulp_budget.is_some_and(|budget| budget > 0) {
                    return Err(EmitError::PtxConstructionFailed(
                        "CUDA PTX `tanh` lowering requires approximate transcendental instructions, but ulp_budget is not positive. Fix: set an explicit ULP budget for this dispatch or route to strict lowering.".into(),
                    ));
                }
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    tanh.approx.f32    {out}, {operand};");
                out
            }
            (UnOp::Exp, PtxType::F32)
            | (UnOp::Log, PtxType::F32)
            | (UnOp::Exp2, PtxType::F32)
            | (UnOp::Log2, PtxType::F32)
            | (UnOp::Sin, PtxType::F32)
            | (UnOp::Cos, PtxType::F32) => {
                if !self.options.ulp_budget.is_some_and(|budget| budget > 0) {
                    return Err(EmitError::PtxConstructionFailed(format!(
                        "CUDA PTX `{op:?}` lowering requires approximate transcendental instructions, but ulp_budget is not positive. Fix: set an explicit ULP budget for this dispatch or route to strict lowering."
                    )));
                }
                let out = self.alloc(PtxType::F32);
                match op {
                    UnOp::Exp => {
                        let tmp = self.alloc(PtxType::F32);
                        let _ = writeln!(self.text, "    mul.f32    {tmp}, {operand}, 0f3FB8AA3B;");
                        let _ = writeln!(self.text, "    ex2.approx.f32    {out}, {tmp};");
                    }
                    UnOp::Log => {
                        let tmp = self.alloc(PtxType::F32);
                        let _ = writeln!(self.text, "    lg2.approx.f32    {tmp}, {operand};");
                        let _ = writeln!(self.text, "    mul.f32    {out}, {tmp}, 0f3F317218;");
                    }
                    UnOp::Exp2 => {
                        let _ = writeln!(self.text, "    ex2.approx.f32    {out}, {operand};");
                    }
                    UnOp::Log2 => {
                        let _ = writeln!(self.text, "    lg2.approx.f32    {out}, {operand};");
                    }
                    UnOp::Sin => {
                        let _ = writeln!(self.text, "    sin.approx.f32    {out}, {operand};");
                    }
                    UnOp::Cos => {
                        let _ = writeln!(self.text, "    cos.approx.f32    {out}, {operand};");
                    }
                    _ => {}
                }
                out
            }
            (UnOp::Floor, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rmi.f32.f32    {out}, {operand};");
                out
            }
            (UnOp::Ceil, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rpi.f32.f32    {out}, {operand};");
                out
            }
            (UnOp::Round, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rni.f32.f32    {out}, {operand};");
                out
            }
            (UnOp::Trunc, PtxType::F32) => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    cvt.rzi.f32.f32    {out}, {operand};");
                out
            }
            (UnOp::Popcount, _) => {
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    popc.b32    {out}, {operand};");
                out
            }
            (UnOp::Clz, _) => {
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    clz.b32    {out}, {operand};");
                out
            }
            (UnOp::Ctz, _) => {
                let reversed = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    brev.b32    {reversed}, {operand};");
                let _ = writeln!(self.text, "    clz.b32    {out}, {reversed};");
                out
            }
            (UnOp::ReverseBits, _) => {
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    brev.b32    {out}, {operand};");
                out
            }
            (UnOp::IsNan, PtxType::F32) => {
                let out = self.alloc(PtxType::Bool);
                let _ = writeln!(
                    self.text,
                    "    setp.nan.f32    {out}, {operand}, {operand};"
                );
                out
            }
            (UnOp::IsInf, PtxType::F32) => {
                let bits = self.alloc(PtxType::U32);
                let abs = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::Bool);
                let _ = writeln!(self.text, "    mov.b32    {bits}, {operand};");
                let _ = writeln!(self.text, "    and.b32    {abs}, {bits}, 0x7fffffff;");
                let _ = writeln!(self.text, "    setp.eq.u32    {out}, {abs}, 0x7f800000;");
                out
            }
            (UnOp::IsFinite, PtxType::F32) => {
                let bits = self.alloc(PtxType::U32);
                let abs = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::Bool);
                let _ = writeln!(self.text, "    mov.b32    {bits}, {operand};");
                let _ = writeln!(self.text, "    and.b32    {abs}, {bits}, 0x7fffffff;");
                let _ = writeln!(self.text, "    setp.lt.u32    {out}, {abs}, 0x7f800000;");
                out
            }
            other => {
                return Err(EmitError::PtxConstructionFailed(format!(
                    "UnOp `{}` on {:?} has no PTX lowering. Fix: add descriptor PTX emission before enabling this op on CUDA.",
                    unop_name(other.0),
                    other.1
                )));
            }
        };
        Ok(self.canonicalize_f32(out))
    }
}
