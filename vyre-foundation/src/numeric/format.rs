//! The scalar formats a numeric contract is stated over.
//!
//! [`DataType`] covers everything a buffer may hold, including tensors, sparse
//! layouts and extension types. A numeric contract is stated over the scalar
//! arithmetic a region performs, so it names a scalar format and reads every
//! rounding, overflow, subnormal and special-value rule from the one semantics
//! table in `vyre-spec` rather than restating any of them.

use serde::{Deserialize, Serialize};
use vyre_spec::{numeric_semantics_for, DataType, NumericSemantics};

/// One scalar arithmetic format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub enum ScalarFormat {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 4-bit integer.
    I4,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
    /// 8-bit float, four exponent bits and three mantissa bits.
    F8E4M3,
    /// 8-bit float, five exponent bits and two mantissa bits.
    F8E5M2,
    /// 4-bit float.
    FP4,
    /// 4-bit normal-float codebook.
    NF4,
}

impl ScalarFormat {
    /// Every scalar format, in declaration order.
    pub const ALL: [Self; 17] = [
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::I4,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
        Self::F8E4M3,
        Self::F8E5M2,
        Self::FP4,
        Self::NF4,
    ];

    /// The data type this format names.
    #[must_use]
    pub fn data_type(self) -> DataType {
        match self {
            Self::U8 => DataType::U8,
            Self::U16 => DataType::U16,
            Self::U32 => DataType::U32,
            Self::U64 => DataType::U64,
            Self::I4 => DataType::I4,
            Self::I8 => DataType::I8,
            Self::I16 => DataType::I16,
            Self::I32 => DataType::I32,
            Self::I64 => DataType::I64,
            Self::F16 => DataType::F16,
            Self::BF16 => DataType::BF16,
            Self::F32 => DataType::F32,
            Self::F64 => DataType::F64,
            Self::F8E4M3 => DataType::F8E4M3,
            Self::F8E5M2 => DataType::F8E5M2,
            Self::FP4 => DataType::FP4,
            Self::NF4 => DataType::NF4,
        }
    }

    /// The scalar format a data type names, if it names one.
    ///
    /// A composite type carries its element type, so the element of a vector,
    /// a tensor, a sparse layout or a quantized value answers for the whole.
    #[must_use]
    pub fn of(dtype: &DataType) -> Option<Self> {
        match dtype {
            DataType::U8 => Some(Self::U8),
            DataType::U16 => Some(Self::U16),
            DataType::U32 | DataType::Vec2U32 | DataType::Vec4U32 => Some(Self::U32),
            DataType::U64 => Some(Self::U64),
            DataType::I4 => Some(Self::I4),
            DataType::I8 => Some(Self::I8),
            DataType::I16 => Some(Self::I16),
            DataType::I32 => Some(Self::I32),
            DataType::I64 => Some(Self::I64),
            DataType::F16 => Some(Self::F16),
            DataType::BF16 => Some(Self::BF16),
            DataType::F32 => Some(Self::F32),
            DataType::F64 => Some(Self::F64),
            DataType::F8E4M3 => Some(Self::F8E4M3),
            DataType::F8E5M2 => Some(Self::F8E5M2),
            DataType::FP4 => Some(Self::FP4),
            DataType::NF4 => Some(Self::NF4),
            DataType::Vec { element, .. }
            | DataType::TensorShaped { element, .. }
            | DataType::SparseCsr { element }
            | DataType::SparseCoo { element }
            | DataType::SparseBsr { element, .. } => Self::of(element),
            DataType::Quantized { storage, .. } => Self::of(storage),
            // `DataType` is `#[non_exhaustive]`, so a cross-crate match needs a
            // rest arm. The arm answers None, and the closure that a new variant
            // must be classified is a test that reads the enum's own source.
            DataType::Bool
            | DataType::Bytes
            | DataType::Array { .. }
            | DataType::Tensor
            | DataType::Handle(_)
            | DataType::DeviceMesh { .. }
            | DataType::Opaque(_) => None,
            _ => None,
        }
    }

    /// The authoritative semantics of this format.
    #[must_use]
    pub fn semantics(self) -> NumericSemantics {
        numeric_semantics_for(&self.data_type())
    }

    /// Whether arithmetic in this format is exact.
    #[must_use]
    pub fn is_exact(self) -> bool {
        self.data_type().arithmetic_is_exact()
    }

    /// The relative fraction one unit in the last place spans.
    ///
    /// An exact format has none: every value it holds is the value itself, so a
    /// ULP bound over it counts exact steps rather than fractions.
    #[must_use]
    pub fn ulp_fraction(self) -> Option<f64> {
        self.semantics()
            .mantissa_bits
            .map(|bits| 2.0_f64.powi(-i32::try_from(bits).unwrap_or(i32::MAX)))
    }
}

impl core::fmt::Display for ScalarFormat {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.data_type())
    }
}
