//! Conversions between PTX register classes.
//!
//! Owns the three coercions the rest of the emitter needs when a value's
//! register class does not match the class an instruction or a memory
//! element demands: f32 subnormal/NaN canonicalization, predicate-to-word
//! widening for stores, and word-to-predicate narrowing. It owns no
//! instruction selection; every arithmetic lowering lives elsewhere.

use std::fmt::Write as _;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};

impl BodyCtx<'_> {
    pub(super) fn canonicalize_f32(&mut self, value: Reg) -> Reg {
        if value.0 != PtxType::F32 {
            return value;
        }
        let flushed = self.alloc(PtxType::F32);
        let nan = self.alloc(PtxType::Bool);
        let out = self.alloc(PtxType::F32);
        let _ = writeln!(
            self.text,
            "    mul.ftz.f32    {flushed}, {value}, 0f3f800000;"
        );
        let _ = writeln!(
            self.text,
            "    setp.nan.f32    {nan}, {flushed}, {flushed};"
        );
        let _ = writeln!(
            self.text,
            "    selp.f32    {out}, 0f7fc00000, {flushed}, {nan};"
        );
        out
    }

    pub(super) fn coerce_for_store(&mut self, value: Reg, elem_ty: PtxType) -> Reg {
        if value.0 != PtxType::Bool || elem_ty == PtxType::Bool {
            return value;
        }
        let out = self.alloc(PtxType::U32);
        let _ = writeln!(self.text, "    selp.u32    {out}, 1, 0, {value};");
        out
    }

    pub(super) fn pred_from_boolish(&mut self, value: Reg) -> Reg {
        if value.0 == PtxType::Bool {
            return value;
        }
        let pred = self.alloc(PtxType::Bool);
        let _ = writeln!(self.text, "    setp.ne.u32    {pred}, {value}, 0;");
        pred
    }
}
