//! Closed orthogonal scalar types for the semantic type system.

/// Closed scalar type representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ScalarType {
    /// 1-bit boolean truth value.
    Bool,
    /// Signed or unsigned integer with explicit bit width.
    Int {
        /// Whether this integer is signed.
        signed: bool,
        /// Bit width (e.g. 8, 16, 32, 64).
        bits: u16,
    },
    /// IEEE or specialized floating point scalar.
    Float {
        /// Floating point format bit width (e.g. 16 for f16/bf16, 32 for f32, 64 for f64).
        bits: u16,
    },
    /// Complex number composed of paired real and imaginary floats.
    Complex {
        /// Component floating point bit width.
        bits: u16,
    },
    /// Substrate-independent memory index / address offset.
    Index,
}

impl ScalarType {
    /// Unsigned 8-bit integer.
    #[must_use]
    pub const fn u8() -> Self {
        Self::Int {
            signed: false,
            bits: 8,
        }
    }

    /// Unsigned 16-bit integer.
    #[must_use]
    pub const fn u16() -> Self {
        Self::Int {
            signed: false,
            bits: 16,
        }
    }

    /// Unsigned 32-bit integer.
    #[must_use]
    pub const fn u32() -> Self {
        Self::Int {
            signed: false,
            bits: 32,
        }
    }

    /// Unsigned 64-bit integer.
    #[must_use]
    pub const fn u64() -> Self {
        Self::Int {
            signed: false,
            bits: 64,
        }
    }

    /// Signed 8-bit integer.
    #[must_use]
    pub const fn i8() -> Self {
        Self::Int {
            signed: true,
            bits: 8,
        }
    }

    /// Signed 16-bit integer.
    #[must_use]
    pub const fn i16() -> Self {
        Self::Int {
            signed: true,
            bits: 16,
        }
    }

    /// Signed 32-bit integer.
    #[must_use]
    pub const fn i32() -> Self {
        Self::Int {
            signed: true,
            bits: 32,
        }
    }

    /// Signed 64-bit integer.
    #[must_use]
    pub const fn i64() -> Self {
        Self::Int {
            signed: true,
            bits: 64,
        }
    }

    /// 16-bit float (f16).
    #[must_use]
    pub const fn f16() -> Self {
        Self::Float { bits: 16 }
    }

    /// 32-bit float (f32).
    #[must_use]
    pub const fn f32() -> Self {
        Self::Float { bits: 32 }
    }

    /// 64-bit float (f64).
    #[must_use]
    pub const fn f64() -> Self {
        Self::Float { bits: 64 }
    }

    /// 1-bit boolean.
    #[must_use]
    pub const fn bool() -> Self {
        Self::Bool
    }

    /// Index type.
    #[must_use]
    pub const fn index() -> Self {
        Self::Index
    }

    /// Bit width of this scalar type.
    #[must_use]
    pub const fn bit_width(&self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::Int { bits, .. } | Self::Float { bits } => *bits as u32,
            Self::Complex { bits } => (*bits as u32) * 2,
            Self::Index => 64,
        }
    }

    /// Byte size in memory (rounded up to whole bytes).
    #[must_use]
    pub const fn byte_width(&self) -> u32 {
        let bits = self.bit_width();
        (bits + 7) / 8
    }

    /// Whether this is an integer scalar.
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(self, Self::Int { .. } | Self::Index)
    }

    /// Whether this is a floating point scalar.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float { .. })
    }

    /// Whether this is a signed integer.
    #[must_use]
    pub const fn is_signed_integer(&self) -> bool {
        match self {
            Self::Int { signed, .. } => *signed,
            _ => false,
        }
    }
}
