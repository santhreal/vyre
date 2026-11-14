//! Scalar literal value representation shared across IR width and identity rules.

use serde::{Deserialize, Serialize};
use vyre_spec::DataType;

/// Scalar literal constant across all supported integer, float, and boolean widths.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum ScalarLiteral {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit IEEE float.
    F32(f32),
    /// 64-bit IEEE float.
    F64(f64),
    /// Boolean proposition.
    Bool(bool),
}

impl Eq for ScalarLiteral {}

impl std::hash::Hash for ScalarLiteral {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::U32(v) => (0u8, *v).hash(state),
            Self::I32(v) => (1u8, *v).hash(state),
            Self::U64(v) => (2u8, *v).hash(state),
            Self::I64(v) => (3u8, *v).hash(state),
            Self::F32(v) => (4u8, v.to_bits()).hash(state),
            Self::F64(v) => (5u8, v.to_bits()).hash(state),
            Self::Bool(v) => (6u8, *v).hash(state),
        }
    }
}

impl ScalarLiteral {
    /// Infer the primitive data type of this literal.
    #[must_use]
    pub const fn data_type(&self) -> DataType {
        match self {
            Self::U32(_) => DataType::U32,
            Self::I32(_) => DataType::I32,
            Self::U64(_) => DataType::U64,
            Self::I64(_) => DataType::I64,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::Bool(_) => DataType::Bool,
        }
    }
}
