//! Binary-operator instruction selection.
//!
//! Owns the `BinOp` to PTX mapping, including the comparison forms that
//! produce a predicate and the total-function division and remainder
//! sequences that PTX has no single instruction for. Literal-operand
//! strength reduction is not here: it lives in `const_strength_reduction`
//! and runs before this fallback.

use super::type_suffix::ptx_binop_suffix;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;
use std::fmt::Write as _;
use vyre_foundation::ir::BinOp;

impl BodyCtx<'_> {
    pub(super) fn emit_binop(
        &mut self,
        op: BinOp,
        left: Reg,
        right: Reg,
    ) -> Result<(Reg, PtxType), EmitError> {
        let ty = left.0;
        if ty == PtxType::Bool && matches!(op, BinOp::Eq | BinOp::Ne) {
            let out = self.alloc(PtxType::Bool);
            let xor = self.alloc(PtxType::Bool);
            let _ = writeln!(self.text, "    xor.pred    {xor}, {left}, {right};");
            if matches!(op, BinOp::Eq) {
                let _ = writeln!(self.text, "    not.pred    {out}, {xor};");
            } else {
                let _ = writeln!(self.text, "    mov.pred    {out}, {xor};");
            }
            return Ok((out, PtxType::Bool));
        }
        if matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        ) {
            let out = self.alloc(PtxType::Bool);
            let cmp = match op {
                BinOp::Eq => "eq",
                BinOp::Ne if ty == PtxType::F32 => "neu",
                BinOp::Ne => "ne",
                BinOp::Lt => "lt",
                BinOp::Le => "le",
                BinOp::Gt => "gt",
                BinOp::Ge => "ge",
                other => {
                    return Err(EmitError::InvalidDescriptor(format!(
                        "comparison lowering received non-comparison operator {other:?}. \
                         Fix: route arithmetic operators through the arithmetic PTX lowering path."
                    )));
                }
            };
            let suffix = if ty == PtxType::F32 {
                "f32"
            } else if ty == PtxType::I32 {
                "s32"
            } else {
                "u32"
            };
            let _ = writeln!(
                self.text,
                "    setp.{cmp}.{suffix}    {out}, {left}, {right};"
            );
            return Ok((out, PtxType::Bool));
        }
        match op {
            BinOp::AbsDiff if ty == PtxType::U32 || ty == PtxType::Bool => {
                let left_ge_right = self.alloc(PtxType::Bool);
                let hi = self.alloc(PtxType::U32);
                let lo = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(
                    self.text,
                    "    setp.ge.u32    {left_ge_right}, {left}, {right};"
                );
                let _ = writeln!(
                    self.text,
                    "    selp.u32    {hi}, {left}, {right}, {left_ge_right};"
                );
                let _ = writeln!(
                    self.text,
                    "    selp.u32    {lo}, {right}, {left}, {left_ge_right};"
                );
                let _ = writeln!(self.text, "    sub.u32    {out}, {hi}, {lo};");
                Ok((out, PtxType::U32))
            }
            BinOp::RotateLeft | BinOp::RotateRight
                if ty == PtxType::U32 || ty == PtxType::I32 || ty == PtxType::Bool =>
            {
                let out = self.alloc(ty);
                let direction = if matches!(op, BinOp::RotateLeft) {
                    "l"
                } else {
                    "r"
                };
                let _ = writeln!(
                    self.text,
                    "    shf.{direction}.wrap.b32    {out}, {left}, {left}, {right};"
                );
                Ok((out, ty))
            }
            BinOp::Shl | BinOp::Shr if ty == PtxType::U32 || ty == PtxType::I32 => {
                let out = self.alloc(ty);
                let masked_shift = self.alloc(PtxType::U32);
                let mnemonic = if matches!(op, BinOp::Shl) {
                    "shl"
                } else {
                    "shr"
                };
                let suffix = ptx_binop_suffix(op, ty);
                let _ = writeln!(self.text, "    and.b32    {masked_shift}, {right}, 31;");
                let _ = writeln!(
                    self.text,
                    "    {mnemonic}.{suffix}    {out}, {left}, {masked_shift};"
                );
                Ok((out, ty))
            }
            BinOp::Div if ty == PtxType::U32 || ty == PtxType::Bool => {
                let out = self.emit_total_u32_binary(left, right, u32::MAX, "u32_div_done", "div");
                Ok((out, PtxType::U32))
            }
            BinOp::Mod if ty == PtxType::U32 || ty == PtxType::Bool => {
                let out = self.emit_total_u32_binary(left, right, 0, "u32_mod_done", "rem");
                Ok((out, PtxType::U32))
            }
            BinOp::Div if ty == PtxType::I32 => {
                let out = self.emit_total_i32_div(left, right);
                Ok((out, PtxType::I32))
            }
            BinOp::Div if ty == PtxType::F32 => {
                let out = self.alloc(PtxType::F32);
                let _ = writeln!(self.text, "    div.rn.f32    {out}, {left}, {right};");
                Ok((self.canonicalize_f32(out), PtxType::F32))
            }
            BinOp::Mod if ty == PtxType::I32 => {
                let out = self.emit_total_i32_mod(left, right);
                Ok((out, PtxType::I32))
            }
            BinOp::SaturatingAdd if ty == PtxType::U32 || ty == PtxType::Bool => {
                let sum = self.alloc(PtxType::U32);
                let overflow = self.alloc(PtxType::Bool);
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    add.u32    {sum}, {left}, {right};");
                let _ = writeln!(self.text, "    setp.lt.u32    {overflow}, {sum}, {left};");
                let _ = writeln!(
                    self.text,
                    "    selp.u32    {out}, 0xffffffff, {sum}, {overflow};"
                );
                Ok((out, PtxType::U32))
            }
            BinOp::SaturatingSub if ty == PtxType::U32 || ty == PtxType::Bool => {
                let underflow = self.alloc(PtxType::Bool);
                let diff = self.alloc(PtxType::U32);
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(
                    self.text,
                    "    setp.lt.u32    {underflow}, {left}, {right};"
                );
                let _ = writeln!(self.text, "    sub.u32    {diff}, {left}, {right};");
                let _ = writeln!(self.text, "    selp.u32    {out}, 0, {diff}, {underflow};");
                Ok((out, PtxType::U32))
            }
            BinOp::SaturatingMul if ty == PtxType::U32 || ty == PtxType::Bool => {
                // Full 64-bit product via the native widening multiply, clamped to
                // u32::MAX when it overflows 32 bits, i.e. when the product's high
                // word is non-zero. Byte-for-byte `u32::saturating_mul` and the
                // oracle's `select(b != 0 && a > MAX/b, MAX, a*b)` contract, without
                // an emulated division.
                let prod = self.alloc(PtxType::U64);
                let prod_hi64 = self.alloc(PtxType::U64);
                let lo = self.alloc(PtxType::U32);
                let hi = self.alloc(PtxType::U32);
                let overflow = self.alloc(PtxType::Bool);
                let out = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    mul.wide.u32    {prod}, {left}, {right};");
                let _ = writeln!(self.text, "    cvt.u32.u64    {lo}, {prod};");
                let _ = writeln!(self.text, "    shr.u64    {prod_hi64}, {prod}, 32;");
                let _ = writeln!(self.text, "    cvt.u32.u64    {hi}, {prod_hi64};");
                let _ = writeln!(self.text, "    setp.ne.u32    {overflow}, {hi}, 0;");
                let _ = writeln!(
                    self.text,
                    "    selp.u32    {out}, 0xffffffff, {lo}, {overflow};"
                );
                Ok((out, PtxType::U32))
            }
            _ => {
                let out_ty = if matches!(op, BinOp::And | BinOp::Or) && ty == PtxType::Bool {
                    PtxType::Bool
                } else {
                    ty
                };
                let out = self.alloc(out_ty);
                let mnemonic = match op {
                    BinOp::Add | BinOp::WrappingAdd => "add",
                    BinOp::Sub | BinOp::WrappingSub => "sub",
                    BinOp::Mul => {
                        if ty == PtxType::F32 {
                            "mul"
                        } else {
                            "mul.lo"
                        }
                    }
                    BinOp::MulHigh => "mul.hi",
                    BinOp::BitAnd | BinOp::And => "and",
                    BinOp::BitOr | BinOp::Or => "or",
                    BinOp::BitXor => "xor",
                    BinOp::Shl => "shl",
                    BinOp::Shr => "shr",
                    BinOp::Min => "min",
                    BinOp::Max => "max",
                    other => {
                        return Err(EmitError::PtxConstructionFailed(format!(
                            "BinOp `{other:?}` has no PTX lowering. Fix: add descriptor PTX emission before enabling this op on CUDA."
                        )));
                    }
                };
                let suffix = ptx_binop_suffix(op, ty);
                let _ = writeln!(
                    self.text,
                    "    {mnemonic}.{suffix}    {out}, {left}, {right};"
                );
                Ok((self.canonicalize_f32(out), out_ty))
            }
        }
    }

    fn emit_total_u32_binary(
        &mut self,
        left: Reg,
        right: Reg,
        zero_divisor_result: u32,
        done_label: &str,
        mnemonic: &str,
    ) -> Reg {
        let out = self.alloc(PtxType::U32);
        let pred = self.alloc(PtxType::Bool);
        let done = self.alloc_label(done_label);
        let _ = writeln!(self.text, "    mov.u32    {out}, {zero_divisor_result};");
        let _ = writeln!(self.text, "    setp.eq.u32    {pred}, {right}, 0;");
        let _ = writeln!(self.text, "    @{pred} bra {done};");
        let _ = writeln!(self.text, "    {mnemonic}.u32    {out}, {left}, {right};");
        let _ = writeln!(self.text, "{done}:");
        out
    }

    fn emit_total_i32_div(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_total_i32_binary(
            left,
            right,
            "div",
            "i32_div_done",
            Some(("i32_div_min_overflow", 0x8000_0000)),
        )
    }

    fn emit_total_i32_mod(&mut self, left: Reg, right: Reg) -> Reg {
        self.emit_total_i32_binary(left, right, "rem", "i32_mod_done", None)
    }

    fn emit_total_i32_binary(
        &mut self,
        left: Reg,
        right: Reg,
        mnemonic: &str,
        done_label: &str,
        overflow_case: Option<(&str, u32)>,
    ) -> Reg {
        let out = self.alloc(PtxType::I32);
        let zero = self.alloc(PtxType::Bool);
        let min = self.alloc(PtxType::Bool);
        let neg_one = self.alloc(PtxType::Bool);
        let overflow = self.alloc(PtxType::Bool);
        let done = self.alloc_label(done_label);
        let overflow_label = overflow_case.map(|(label, value)| (self.alloc_label(label), value));
        let _ = writeln!(self.text, "    mov.s32    {out}, 0;");
        let _ = writeln!(self.text, "    setp.eq.s32    {zero}, {right}, 0;");
        let _ = writeln!(self.text, "    @{zero} bra {done};");
        let _ = writeln!(self.text, "    setp.eq.u32    {min}, {left}, 0x80000000;");
        let _ = writeln!(
            self.text,
            "    setp.eq.u32    {neg_one}, {right}, 0xffffffff;"
        );
        let _ = writeln!(self.text, "    and.pred    {overflow}, {min}, {neg_one};");
        if let Some((label, _)) = &overflow_label {
            let _ = writeln!(self.text, "    @{overflow} bra {label};");
        } else {
            let _ = writeln!(self.text, "    @{overflow} bra {done};");
        }
        let _ = writeln!(self.text, "    {mnemonic}.s32    {out}, {left}, {right};");
        if let Some((label, value)) = overflow_label {
            let _ = writeln!(self.text, "    bra {done};");
            let _ = writeln!(self.text, "{label}:");
            let _ = writeln!(self.text, "    mov.u32    {out}, 0x{value:08x};");
        }
        let _ = writeln!(self.text, "{done}:");
        out
    }
}
