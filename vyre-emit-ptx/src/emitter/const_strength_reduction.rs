//! Strength reduction for multiply, divide, and remainder by a literal.
//!
//! Owns the shift-and-add, magic-multiply, and mask rewrites that replace a
//! `mul`/`div`/`rem` whose right operand is a known `u32` constant. Each
//! entry point returns `None` when the constant has no profitable form, and
//! the caller then falls back to the general lowering in `binop`.

use std::fmt::Write as _;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};

impl BodyCtx<'_> {
    pub(super) fn emit_small_u32_const_mul(&mut self, value: Reg, constant: u32) -> Option<Reg> {
        if !matches!(value.0, PtxType::U32 | PtxType::Bool) {
            return None;
        }
        if constant == 0 {
            let out = self.alloc(PtxType::U32);
            let _ = writeln!(self.text, "    mov.u32    {out}, 0;");
            return Some(out);
        }
        if constant == 1 {
            return Some(value);
        }
        if constant.is_power_of_two() {
            let out = self.alloc(PtxType::U32);
            let shift = constant.trailing_zeros();
            let _ = writeln!(self.text, "    shl.b32    {out}, {value}, {shift};");
            return Some(out);
        }
        if constant.count_ones() > 4 {
            return None;
        }
        let mut acc = None;
        for shift in 0..u32::BITS {
            if (constant & (1u32 << shift)) == 0 {
                continue;
            }
            let term = if shift == 0 {
                value
            } else {
                let shifted = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    shl.b32    {shifted}, {value}, {shift};");
                shifted
            };
            acc = Some(match acc {
                Some(prev) => {
                    let out = self.alloc(PtxType::U32);
                    let _ = writeln!(self.text, "    add.u32    {out}, {prev}, {term};");
                    out
                }
                None => term,
            });
        }
        acc
    }

    pub(super) fn emit_fast_u32_const_div(&mut self, value: Reg, divisor: u32) -> Option<Reg> {
        if !matches!(value.0, PtxType::U32 | PtxType::Bool) || divisor == 0 {
            return None;
        }
        if divisor == 1 {
            return Some(value);
        }
        if divisor.is_power_of_two() {
            let out = self.alloc(PtxType::U32);
            let shift = divisor.trailing_zeros();
            let _ = writeln!(self.text, "    shr.u32    {out}, {value}, {shift};");
            return Some(out);
        }
        if divisor == 3 {
            let magic = self.alloc(PtxType::U32);
            let high = self.alloc(PtxType::U32);
            let out = self.alloc(PtxType::U32);
            let _ = writeln!(self.text, "    mov.u32    {magic}, 0xaaaaaaab;");
            let _ = writeln!(self.text, "    mul.hi.u32    {high}, {value}, {magic};");
            let _ = writeln!(self.text, "    shr.u32    {out}, {high}, 1;");
            return Some(out);
        }
        None
    }

    pub(super) fn emit_fast_u32_const_mod(&mut self, value: Reg, divisor: u32) -> Option<Reg> {
        if !matches!(value.0, PtxType::U32 | PtxType::Bool) || divisor == 0 {
            return None;
        }
        let out = self.alloc(PtxType::U32);
        if divisor == 1 {
            let _ = writeln!(self.text, "    mov.u32    {out}, 0;");
            return Some(out);
        }
        if divisor.is_power_of_two() {
            let mask = divisor - 1;
            let _ = writeln!(self.text, "    and.b32    {out}, {value}, {mask};");
            return Some(out);
        }
        None
    }
}
