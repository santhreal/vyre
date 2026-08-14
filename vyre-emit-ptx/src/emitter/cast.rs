//! Scalar conversion between PTX register classes.
//!
//! Owns `Cast` lowering: the `cvt` form for each source/target register
//! class pair, the saturating float-to-integer path, and the truthiness
//! rules for casts to and from a predicate. Register-class coercion that a
//! non-cast instruction needs lives in `coercion`.

use std::fmt::Write as _;
use vyre_foundation::ir::DataType;
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

impl BodyCtx<'_> {
    pub(super) fn emit_cast(&mut self, src: Reg, target: &DataType) -> Result<Reg, EmitError> {
        let dst_ty = PtxType::from_dtype(target)?;
        // A float source has no defined narrowing integer conversion: it converts
        // only to u32/i32 (saturating `cvt.rzi`, below), bool (truthy), or f32.
        // `from_dtype` collapses U8/U16->U32 and I8/I16->I32, so WITHOUT this guard
        // an f32->u8 would silently emit a non-narrowing `cvt.rzi.u32.f32` (a full
        // u32-range value claimed as a u8). The foundation cast table
        // (`validate::cast::cast_is_valid`) rejects these casts, but the
        // no-validation emit path can reach here. Fail closed (Law 10), matching
        // the naga emitter and the `Bytes` arm in `from_dtype`. (f32->u64/i64 maps
        // to PtxType::U64 != F32, so it already fails closed via the unmatched `_`
        // arm below; this closes the narrow-int holes that `from_dtype` masks.)
        if src.0 == PtxType::F32
            && matches!(
                target,
                DataType::U8 | DataType::U16 | DataType::I8 | DataType::I16
            )
        {
            return Err(EmitError::PtxConstructionFailed(format!(
                "cast from f32 to `{target:?}` has no defined conversion: a float \
                 source converts only to u32/i32 (saturating), bool (truthy), or f32. \
                 Fix: cast the f32 to u32 or i32 first, then narrow the integer."
            )));
        }
        // Narrowing to a sub-word integer (u8/u16/i8/i16). `from_dtype` collapses
        // these to U32/I32, so the `src.0 == dst_ty` identity check below would
        // treat `u32 -> u8` as a no-op and KEEP the high bits, a silent
        // non-narrowing that diverges from Rust `as`, the V035 contract, and the
        // reference oracle. PTX has no sub-word register, but it has the canonical
        // narrowing converts: `cvt.u32.u8` zero-extends the low byte (== `& 0xFF`)
        // and `cvt.s32.s8` sign-extends it (`200 as i8 == -56`). Reduce the source
        // to its low 32-bit word first, then emit the narrowing convert.
        let narrow = match target {
            DataType::U8 => Some(("u32", "u8")),
            DataType::U16 => Some(("u32", "u16")),
            DataType::I8 => Some(("s32", "s8")),
            DataType::I16 => Some(("s32", "s16")),
            _ => None,
        };
        if let Some((dst_t, src_t)) = narrow {
            let base = match src.0 {
                PtxType::U32 | PtxType::I32 => src,
                PtxType::Bool => {
                    let word = self.alloc(PtxType::U32);
                    let _ = writeln!(self.text, "    selp.u32    {word}, 1, 0, {src};");
                    word
                }
                PtxType::U64 => {
                    let word = self.alloc(PtxType::U32);
                    let _ = writeln!(self.text, "    cvt.u32.u64    {word}, {src};");
                    word
                }
                PtxType::F32 | PtxType::B16 => {
                    // F32 is already rejected by the guard above; B16 (packed
                    // f16/bf16) is not an integer-like narrowing source. Fail
                    // closed rather than reinterpret float bits as an integer.
                    return Err(EmitError::PtxConstructionFailed(format!(
                        "cast from {:?} to `{target:?}` has no defined integer \
                         narrowing: only an integer-like scalar source narrows. \
                         Fix: cast to u32 or i32 first, then narrow.",
                        src.0
                    )));
                }
            };
            let dst = self.alloc(dst_ty);
            let _ = writeln!(self.text, "    cvt.{dst_t}.{src_t}    {dst}, {base};");
            return Ok(dst);
        }
        if src.0 == dst_ty {
            return Ok(src);
        }
        let dst = self.alloc(dst_ty);
        match (src.0, dst_ty) {
            (PtxType::U32, PtxType::F32) => {
                let _ = writeln!(self.text, "    cvt.rn.f32.u32    {dst}, {src};");
            }
            (PtxType::I32, PtxType::F32) => {
                let _ = writeln!(self.text, "    cvt.rn.f32.s32    {dst}, {src};");
            }
            (PtxType::Bool, PtxType::F32) => {
                let word = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    selp.u32    {word}, 1, 0, {src};");
                let _ = writeln!(self.text, "    cvt.rn.f32.u32    {dst}, {word};");
            }
            (PtxType::F32, PtxType::U32) => {
                let _ = writeln!(self.text, "    cvt.rzi.u32.f32    {dst}, {src};");
            }
            (PtxType::F32, PtxType::I32) => {
                let _ = writeln!(self.text, "    cvt.rzi.s32.f32    {dst}, {src};");
            }
            (PtxType::Bool, PtxType::U32) => {
                let _ = writeln!(self.text, "    selp.u32    {dst}, 1, 0, {src};");
            }
            (PtxType::Bool, PtxType::I32) => {
                let _ = writeln!(self.text, "    selp.u32    {dst}, 1, 0, {src};");
            }
            (PtxType::U32 | PtxType::I32, PtxType::Bool) => {
                let _ = writeln!(self.text, "    setp.ne.u32    {dst}, {src}, 0;");
            }
            (PtxType::F32, PtxType::Bool) => {
                let _ = writeln!(self.text, "    setp.neu.f32    {dst}, {src}, 0f00000000;");
            }
            (PtxType::U32, PtxType::I32) | (PtxType::I32, PtxType::U32) => {
                let _ = writeln!(self.text, "    mov.b32    {dst}, {src};");
            }
            (PtxType::U32, PtxType::U64) => {
                // Zero-extend 32→64.
                let _ = writeln!(self.text, "    cvt.u64.u32    {dst}, {src};");
            }
            (PtxType::I32, PtxType::U64) => {
                // Sign-extend 32→64; the 64-bit two's-complement bit pattern is
                // written into the .u64 (`%rd`) register.
                let _ = writeln!(self.text, "    cvt.s64.s32    {dst}, {src};");
            }
            (PtxType::U64, PtxType::U32) => {
                // Explicit narrowing: keep the low 32 bits.
                let _ = writeln!(self.text, "    cvt.u32.u64    {dst}, {src};");
            }
            (PtxType::U64, PtxType::I32) => {
                // Narrowing to a signed 32-bit value: keep the low 32 bits; the
                // bit pattern of the low word IS the i32 (matches naga's low-word
                // bitcast and the reference's low-word narrowing). NEVER fail
                // closed here (wgpu/naga supports u64->i32, so CUDA must too).
                let _ = writeln!(self.text, "    cvt.u32.u64    {dst}, {src};");
            }
            (PtxType::U64, PtxType::Bool) => {
                // Truthiness over the FULL 64 bits (not just the low word):
                // matches the reference `value != 0` and naga's (low | high) != 0.
                let _ = writeln!(self.text, "    setp.ne.u64    {dst}, {src}, 0;");
            }
            (PtxType::U64, PtxType::F32) => {
                // Full 64-bit → F32 with round-to-nearest. NEVER narrow to u32
                // first: that silently discards the high 32 bits.
                let _ = writeln!(self.text, "    cvt.rn.f32.u64    {dst}, {src};");
            }
            _ => {
                return Err(EmitError::PtxConstructionFailed(format!(
                    "unsupported PTX cast from {:?} to {:?}. Fix: validate casts before CUDA emission.",
                    src.0, dst_ty
                )));
            }
        }
        Ok(dst)
    }
}
