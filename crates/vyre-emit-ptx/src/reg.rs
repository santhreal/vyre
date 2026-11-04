use crate::EmitError;
use std::fmt;
use std::fmt::Write as _;
use vyre_foundation::ir::DataType;

/// PTX scalar register classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PtxType {
    /// `.b16` register (`%h<N>`)  -  used for packed f16/bf16 memory values.
    B16,
    /// `.u32` register (`%r<N>`).
    U32,
    /// `.s32` register (`%s<N>`).
    I32,
    /// `.f32` register (`%f<N>`).
    F32,
    /// `.pred` register (`%p<N>`).
    Bool,
    /// `.u64` register (`%rd<N>`)  -  used for pointers.
    U64,
}

impl PtxType {
    pub(crate) fn ptx_type_str(self) -> &'static str {
        self.descriptor().0
    }

    pub(crate) fn reg_prefix(self) -> &'static str {
        self.descriptor().1
    }

    fn descriptor(self) -> (&'static str, &'static str) {
        match self {
            Self::B16 => ("b16", "h"),
            Self::U32 => ("u32", "r"),
            Self::I32 => ("s32", "s"),
            Self::F32 => ("f32", "f"),
            Self::Bool => ("pred", "p"),
            Self::U64 => ("u64", "rd"),
        }
    }

    pub(crate) fn from_dtype(dt: &DataType) -> Result<Self, EmitError> {
        match dt {
            DataType::Bool => Ok(Self::Bool),
            DataType::U8 | DataType::U16 | DataType::U32 => Ok(Self::U32),
            DataType::I8 | DataType::I16 | DataType::I32 => Ok(Self::I32),
            DataType::F16 | DataType::BF16 | DataType::F32 => Ok(Self::F32),
            // PTX 64-bit registers (`%rd`, `.u64`) are typeless bit containers
            //: signedness is per-instruction, not per-register, so both U64
            // and I64 map here. `validate::cast::cast_is_valid` allows
            // `i32 -> i64`, and `emit_cast`'s `(I32, U64) => cvt.s64.s32`
            // sign-extend / `(U32, U64) => cvt.u64.u32` zero-extend arms then
            // produce the correct 64-bit pattern. Before this, I64 fell to the
            // `other` arm and a valid `Cast { target: I64 }` errored.
            DataType::U64 | DataType::I64 => Ok(Self::U64),
            // `Bytes` is a packed-byte buffer-element marker, NOT a scalar
            // register type. Folding it into `.u32` here would silently
            // reinterpret a byte stream as a word (Law 10): a `Bytes` buffer
            // load would index words instead of bytes, and a `Cast { Bytes }`
            // of a u32 would no-op (src == dst == .u32). It needs a pack-to-u32
            // pre-pass before emission, so fail closed and name the fix.
            DataType::Bytes => Err(EmitError::UnsupportedDataType(
                "Bytes is a packed-byte buffer element, not a scalar register \
                 type; it requires a pack-to-u32 pre-pass before PTX emission \
                 and must never be reinterpreted as a u32 word"
                    .to_owned(),
            )),
            other => Err(EmitError::UnsupportedDataType(format!("{other:?}"))),
        }
    }
}

/// One named PTX register: a (type, index) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Reg(pub(crate) PtxType, pub(crate) u32);

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}{}", self.0.reg_prefix(), self.1)
    }
}

/// Write a brace-delimited PTX register tuple: `{%r1, %r2}`.
pub(crate) fn write_reg_tuple(out: &mut String, regs: &[Reg]) {
    out.push('{');
    for (idx, reg) in regs.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{reg}");
    }
    out.push('}');
}

// Inline: covers the crate-private `PtxType` and `Reg`, which no integration
// test can reach.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_type_from_dtype_covers_scalars() {
        assert_eq!(PtxType::from_dtype(&DataType::Bool).unwrap(), PtxType::Bool);
        assert_eq!(PtxType::from_dtype(&DataType::U8).unwrap(), PtxType::U32);
        assert_eq!(PtxType::from_dtype(&DataType::I8).unwrap(), PtxType::I32);
        assert_eq!(PtxType::from_dtype(&DataType::U16).unwrap(), PtxType::U32);
        assert_eq!(PtxType::from_dtype(&DataType::I16).unwrap(), PtxType::I32);
        assert_eq!(PtxType::from_dtype(&DataType::U32).unwrap(), PtxType::U32);
        assert_eq!(PtxType::from_dtype(&DataType::I32).unwrap(), PtxType::I32);
        assert_eq!(PtxType::from_dtype(&DataType::F32).unwrap(), PtxType::F32);
    }

    #[test]
    fn ptx_type_from_dtype_rejects_unsupported() {
        assert!(matches!(
            PtxType::from_dtype(&DataType::Tensor),
            Err(EmitError::UnsupportedDataType(_))
        ));
    }

    #[test]
    fn ptx_type_from_dtype_rejects_bytes_instead_of_silent_u32() {
        // `Bytes` is a packed-byte buffer element, not a scalar register type.
        // Before this guard it was folded into `.u32` (grouped with U8/U16/U32),
        // silently reinterpreting a byte stream as a word: a `Bytes` buffer load
        // would index words not bytes, and a `Cast { Bytes }` of a u32 would no-op
        // (src == dst). The emitter must fail closed and name the pack-to-u32 fix.
        let err = PtxType::from_dtype(&DataType::Bytes)
            .expect_err("from_dtype(Bytes) must fail closed, not silently map to .u32");
        let EmitError::UnsupportedDataType(msg) = &err else {
            panic!("Bytes rejection must be UnsupportedDataType; got {err:?}");
        };
        assert!(
            msg.contains("Bytes") && msg.contains("pack-to-u32 pre-pass"),
            "Bytes rejection must name the type and the fix (pack-to-u32 pre-pass); got: {msg}"
        );
    }

    #[test]
    fn reg_display_uses_correct_prefix() {
        assert_eq!(format!("{}", Reg(PtxType::U32, 5)), "%r5");
        assert_eq!(format!("{}", Reg(PtxType::I32, 0)), "%s0");
        assert_eq!(format!("{}", Reg(PtxType::F32, 3)), "%f3");
        assert_eq!(format!("{}", Reg(PtxType::Bool, 1)), "%p1");
        assert_eq!(format!("{}", Reg(PtxType::U64, 7)), "%rd7");
    }
}
