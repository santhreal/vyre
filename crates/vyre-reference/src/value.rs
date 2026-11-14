//! Runtime values accepted and returned by the core reference interpreter.
//!
//! Every payload a value carries is immutable and reference counted, so
//! sharing a value costs one refcount bump and no payload copy. The evaluator
//! binds the same value into a nested scope, hands it to a peer lane as part
//! of a subgroup snapshot, and reads it back out of a local, all without
//! duplicating the bytes behind it.
//! [`Value::shares_payload_with`](crate::value::Value::shares_payload_with) reports
//! whether two values stand on one payload.

use std::sync::Arc;

/// A concrete value passed into or returned from the reference interpreter.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Value {
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Boolean value.
    Bool(bool),
    /// Raw little-endian storage bytes.
    Bytes(Arc<[u8]>),
    /// Floating-point value represented with stable host bits.
    Float(f64),
    /// Fixed-size array of values, shared so a rebind copies a refcount
    /// rather than every element.
    Array(Arc<[Value]>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            // Elementwise, and therefore reflexive. Without this arm an
            // `Array` fell through to the catch-all and compared unequal to
            // itself, which breaks the `Eq` this type also claims and makes
            // every oracle comparison of an array output report a mismatch.
            (Self::Array(a), Self::Array(b)) => a == b,
            (
                Self::U32(_)
                | Self::I32(_)
                | Self::U64(_)
                | Self::Bool(_)
                | Self::Bytes(_)
                | Self::Float(_)
                | Self::Array(_),
                _,
            ) => false,
        }
    }
}

impl Eq for Value {}

impl Value {
    /// Build an array value from owned elements.
    ///
    /// The elements move into shared storage once; every later rebind,
    /// snapshot or tile read copies a refcount.
    #[must_use]
    pub fn array(elements: impl Into<Arc<[Self]>>) -> Self {
        Self::Array(elements.into())
    }

    /// True when `self` and `other` stand on one reference-counted payload.
    ///
    /// A share of [`Value::Bytes`] or [`Value::Array`] is a refcount bump, so
    /// the two values that come out of it point at the same allocation and
    /// this reports `true`. A value rebuilt from the same elements is a
    /// second allocation and reports `false`; equality cannot tell a share
    /// from a copy.
    ///
    /// A scalar carries no payload, so two scalars never share one and this
    /// reports `false` for every pair of them, including a value and itself.
    #[must_use]
    pub fn shares_payload_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bytes(left), Self::Bytes(right)) => Arc::ptr_eq(left, right),
            (Self::Array(left), Self::Array(right)) => Arc::ptr_eq(left, right),
            (
                Self::U32(_)
                | Self::I32(_)
                | Self::U64(_)
                | Self::Bool(_)
                | Self::Bytes(_)
                | Self::Float(_)
                | Self::Array(_),
                _,
            ) => false,
        }
    }

    /// Interpret the value using the IR truth convention.
    #[must_use]
    pub fn truthy(&self) -> bool {
        match self {
            Self::Array(values) => !values.is_empty(),
            Self::Float(value) => *value != 0.0,
            Self::U32(_) | Self::I32(_) | Self::U64(_) | Self::Bool(_) | Self::Bytes(_) => {
                self.try_as_u32().unwrap_or(1) != 0
            }
        }
    }

    /// Return this value as little-endian bytes for buffer initialization.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::U32(value) => value.to_le_bytes().into(),
            Self::I32(value) => value.to_le_bytes().into(),
            Self::U64(value) => value.to_le_bytes().into(),
            Self::Bool(value) => u32::from(*value).to_le_bytes().into(),
            Self::Bytes(bytes) => bytes.as_ref().into(),
            Self::Float(value) => value.to_le_bytes().into(),
            Self::Array(values) => values.iter().flat_map(Self::to_bytes).collect(),
        }
    }

    /// Return this value encoded at the declared input width.
    #[must_use]
    pub fn to_bytes_width(&self, declared_width: usize) -> Vec<u8> {
        let mut bytes = self.to_bytes();
        if declared_width == 0 {
            return bytes;
        }
        bytes.resize(declared_width, 0);
        bytes.truncate(declared_width);
        bytes
    }

    /// Write this value into an existing fixed-width byte slot.
    ///
    /// For non-empty targets, this is equivalent to
    /// `target.copy_from_slice(&self.to_bytes_width(target.len()))` but avoids
    /// allocating a temporary vector on store-heavy reference paths. Empty
    /// targets are a no-op because they cannot carry the variable-width
    /// `to_bytes_width(0)` payload.
    pub fn write_bytes_width_into(&self, target: &mut [u8]) {
        target.fill(0);
        let mut cursor = 0usize;
        self.copy_raw_bytes_prefix(target, &mut cursor);
    }

    /// Append this value encoded at the declared input width without
    /// allocating a temporary byte vector for the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination length would overflow.
    pub fn extend_bytes_width(
        &self,
        declared_width: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), crate::ReferenceError> {
        let start_len = out.len();
        let fixed_next_len = if declared_width == 0 {
            None
        } else {
            Some(start_len.checked_add(declared_width).ok_or_else(|| {
                crate::ReferenceError::new(
                    "encoded value byte size overflows usize. Fix: reduce the argument count or byte payload size.",
                )
            })?)
        };
        match self {
            Self::U32(value) => extend_fixed_width(&value.to_le_bytes(), declared_width, out),
            Self::I32(value) => extend_fixed_width(&value.to_le_bytes(), declared_width, out),
            Self::U64(value) => extend_fixed_width(&value.to_le_bytes(), declared_width, out),
            Self::Bool(value) => {
                extend_fixed_width(&u32::from(*value).to_le_bytes(), declared_width, out);
            }
            Self::Bytes(bytes) => extend_fixed_width(bytes, declared_width, out),
            Self::Float(value) => extend_fixed_width(&value.to_le_bytes(), declared_width, out),
            Self::Array(values) => {
                for value in values.iter() {
                    value.extend_bytes_width(0, out)?;
                }
                if let Some(next_len) = fixed_next_len {
                    out.truncate(start_len + declared_width.min(out.len() - start_len));
                    out.resize(next_len, 0);
                }
            }
        }
        if let Some(next_len) = fixed_next_len {
            debug_assert_eq!(out.len(), next_len);
        }
        Ok(())
    }

    fn copy_raw_bytes_prefix(&self, target: &mut [u8], cursor: &mut usize) {
        match self {
            Self::U32(value) => copy_bytes_prefix(&value.to_le_bytes(), target, cursor),
            Self::I32(value) => copy_bytes_prefix(&value.to_le_bytes(), target, cursor),
            Self::U64(value) => copy_bytes_prefix(&value.to_le_bytes(), target, cursor),
            Self::Bool(value) => {
                copy_bytes_prefix(&u32::from(*value).to_le_bytes(), target, cursor);
            }
            Self::Bytes(bytes) => copy_bytes_prefix(bytes, target, cursor),
            Self::Float(value) => copy_bytes_prefix(&value.to_le_bytes(), target, cursor),
            Self::Array(values) => {
                for value in values.iter() {
                    if *cursor >= target.len() {
                        break;
                    }
                    value.copy_raw_bytes_prefix(target, cursor);
                }
            }
        }
    }

    /// Try to interpret the value as the IR's scalar `u32` word.
    #[must_use]
    pub fn try_as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(*value),
            Self::I32(value) => u32::try_from(*value).ok(),
            Self::U64(value) => u32::try_from(*value).ok(),
            Self::Bool(value) => Some(u32::from(*value)),
            Self::Bytes(bytes) => (bytes.len() <= 4).then(|| read_u32_prefix(bytes)),
            Self::Float(value) => f64_to_u32(*value),
            Self::Array(_) => None,
        }
    }

    /// Interpret the value as the IR's scalar `u32` word.
    #[must_use]
    pub fn as_u32(&self) -> u32 {
        self.try_as_u32().unwrap_or(0)
    }

    /// Try to interpret the value as a full `u64`.
    #[must_use]
    pub fn try_as_u64(&self) -> Option<u64> {
        match self {
            Self::U32(value) => Some(u64::from(*value)),
            Self::I32(value) => u64::try_from(*value).ok(),
            Self::U64(value) => Some(*value),
            Self::Bool(value) => Some(u64::from(*value)),
            Self::Bytes(bytes) => (bytes.len() <= 8).then(|| read_u64_prefix(bytes)),
            Self::Float(value) => f64_to_u64(*value),
            Self::Array(_) => None,
        }
    }

    /// Interpret the value as a full `u64`.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        self.try_as_u64().unwrap_or(0)
    }

    /// Try to interpret the value as an `f32`.
    #[must_use]
    pub fn try_as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(value) => Some(*value as f32),
            Self::U32(value) => Some(f32::from_bits(*value)),
            Self::I32(_) | Self::U64(_) | Self::Bool(_) | Self::Bytes(_) | Self::Array(_) => None,
        }
    }
    /// Try to interpret the value as an `f64`.
    #[must_use]
    pub fn try_as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            Self::U32(value) => Some(f64::from(f32::from_bits(*value))),
            Self::I32(value) => Some(f64::from(*value)),
            Self::U64(value) => Some(f64::from_bits(*value)),
            Self::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
            Self::Bytes(_) | Self::Array(_) => None,
        }
    }

    /// Return the full value payload as little-endian bytes.
    #[must_use]
    pub fn wide_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }

    /// Try to create a zero value for the given data type.
    #[must_use]
    pub fn try_zero_for(ty: vyre_foundation::ir::DataType) -> Option<Self> {
        Self::try_zero_for_ref(&ty)
    }

    /// Borrowed counterpart of [`Self::try_zero_for`] for hot load paths.
    #[must_use]
    pub(crate) fn try_zero_for_ref(ty: &vyre_foundation::ir::DataType) -> Option<Self> {
        match ty {
            vyre_foundation::ir::DataType::U32 => Some(Self::U32(0)),
            vyre_foundation::ir::DataType::I32 => Some(Self::I32(0)),
            vyre_foundation::ir::DataType::U64 => Some(Self::U64(0)),
            vyre_foundation::ir::DataType::Bool => Some(Self::Bool(false)),
            vyre_foundation::ir::DataType::Bytes => Some(Self::Bytes(Arc::from([]))),
            vyre_foundation::ir::DataType::F32 => Some(Self::Float(0.0)),
            vyre_foundation::ir::DataType::F64 => Some(Self::Float(0.0)),
            vyre_foundation::ir::DataType::Vec2U32 => Some(Self::Bytes(Arc::from([0u8; 8]))),
            vyre_foundation::ir::DataType::Vec4U32 => Some(Self::Bytes(Arc::from([0u8; 16]))),
            // Every remaining declared `DataType` is named rather than
            // absorbed, so adding a variant to the spec fails to compile here
            // instead of silently taking the storage-width fallback.
            other @ (vyre_foundation::ir::DataType::U8
            | vyre_foundation::ir::DataType::U16
            | vyre_foundation::ir::DataType::I8
            | vyre_foundation::ir::DataType::I16
            | vyre_foundation::ir::DataType::I64
            | vyre_foundation::ir::DataType::F16
            | vyre_foundation::ir::DataType::BF16
            | vyre_foundation::ir::DataType::F8E4M3
            | vyre_foundation::ir::DataType::F8E5M2
            | vyre_foundation::ir::DataType::I4
            | vyre_foundation::ir::DataType::FP4
            | vyre_foundation::ir::DataType::NF4
            | vyre_foundation::ir::DataType::Tensor
            | vyre_foundation::ir::DataType::Handle(_)
            | vyre_foundation::ir::DataType::Array { .. }
            | vyre_foundation::ir::DataType::Vec { .. }
            | vyre_foundation::ir::DataType::TensorShaped { .. }
            | vyre_foundation::ir::DataType::SparseCsr { .. }
            | vyre_foundation::ir::DataType::SparseCoo { .. }
            | vyre_foundation::ir::DataType::SparseBsr { .. }
            | vyre_foundation::ir::DataType::DeviceMesh { .. }
            | vyre_foundation::ir::DataType::Quantized { .. }
            | vyre_foundation::ir::DataType::Opaque(_)) => fixed_scalar_storage_width(other)
                .map(|width| Self::Bytes(Arc::from(vec![0; width]))),
        }
    }

    /// Create a value from element bytes for the given data type.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte slice is too short for the declared type.
    pub fn from_element_bytes(
        ty: vyre_foundation::ir::DataType,
        bytes: &[u8],
    ) -> Result<Self, String> {
        Self::from_element_bytes_ref(&ty, bytes)
    }

    /// Borrowed counterpart of [`Self::from_element_bytes`] for hot load paths.
    pub(crate) fn from_element_bytes_ref(
        ty: &vyre_foundation::ir::DataType,
        bytes: &[u8],
    ) -> Result<Self, String> {
        match ty {
            vyre_foundation::ir::DataType::U32 => Ok(Self::U32(u32::from_le_bytes(require_bytes(
                bytes,
                "u32 requires 4 bytes",
            )?))),
            vyre_foundation::ir::DataType::I32 => Ok(Self::I32(i32::from_le_bytes(require_bytes(
                bytes,
                "i32 requires 4 bytes",
            )?))),
            vyre_foundation::ir::DataType::U64 => Ok(Self::U64(u64::from_le_bytes(require_bytes(
                bytes,
                "u64 requires 8 bytes",
            )?))),
            vyre_foundation::ir::DataType::Bool => Ok(Self::Bool(
                u32::from_le_bytes(require_bytes(bytes, "bool requires 4 bytes")?) != 0,
            )),
            vyre_foundation::ir::DataType::Vec2U32 => {
                let arr: [u8; 8] = require_bytes(bytes, "vec2u32 requires 8 bytes")?;
                Ok(Self::Bytes(Arc::from(arr)))
            }
            vyre_foundation::ir::DataType::Vec4U32 => {
                let arr: [u8; 16] = require_bytes(bytes, "vec4u32 requires 16 bytes")?;
                Ok(Self::Bytes(Arc::from(arr)))
            }
            vyre_foundation::ir::DataType::F32 => {
                let value = f32::from_le_bytes(require_bytes(bytes, "f32 requires 4 bytes")?);
                Ok(Self::Float(f64::from(
                    crate::execution::typed_ops::canonical_f32(value),
                )))
            }
            vyre_foundation::ir::DataType::F64 => Ok(Self::Float(f64::from_le_bytes(
                require_bytes(bytes, "f64 requires 8 bytes")?,
            ))),
            vyre_foundation::ir::DataType::Bytes => Ok(Self::Bytes(Arc::from(bytes))),
            // Every remaining declared `DataType` is named rather than
            // absorbed, so adding a variant to the spec fails to compile here
            // instead of silently decoding as opaque storage bytes.
            other @ (vyre_foundation::ir::DataType::U8
            | vyre_foundation::ir::DataType::U16
            | vyre_foundation::ir::DataType::I8
            | vyre_foundation::ir::DataType::I16
            | vyre_foundation::ir::DataType::I64
            | vyre_foundation::ir::DataType::F16
            | vyre_foundation::ir::DataType::BF16
            | vyre_foundation::ir::DataType::F8E4M3
            | vyre_foundation::ir::DataType::F8E5M2
            | vyre_foundation::ir::DataType::I4
            | vyre_foundation::ir::DataType::FP4
            | vyre_foundation::ir::DataType::NF4
            | vyre_foundation::ir::DataType::Tensor
            | vyre_foundation::ir::DataType::Handle(_)
            | vyre_foundation::ir::DataType::Array { .. }
            | vyre_foundation::ir::DataType::Vec { .. }
            | vyre_foundation::ir::DataType::TensorShaped { .. }
            | vyre_foundation::ir::DataType::SparseCsr { .. }
            | vyre_foundation::ir::DataType::SparseCoo { .. }
            | vyre_foundation::ir::DataType::SparseBsr { .. }
            | vyre_foundation::ir::DataType::DeviceMesh { .. }
            | vyre_foundation::ir::DataType::Quantized { .. }
            | vyre_foundation::ir::DataType::Opaque(_)) => {
                match fixed_scalar_storage_width(other) {
                    Some(width) => {
                        if bytes.len() < width {
                            return Err(format!("{other} requires {width} bytes"));
                        }
                        Ok(Self::Bytes(Arc::from(&bytes[..width])))
                    }
                    None => Ok(Self::Bytes(Arc::from(bytes))),
                }
            }
        }
    }
}

fn require_bytes<const N: usize>(bytes: &[u8], msg: &'static str) -> Result<[u8; N], String> {
    bytes
        .get(..N)
        .and_then(|s| <[u8; N]>::try_from(s).ok())
        .ok_or_else(|| String::from(msg))
}

fn fixed_scalar_storage_width(ty: &vyre_foundation::ir::DataType) -> Option<usize> {
    match ty {
        vyre_foundation::ir::DataType::U8
        | vyre_foundation::ir::DataType::I8
        | vyre_foundation::ir::DataType::F8E4M3
        | vyre_foundation::ir::DataType::F8E5M2
        | vyre_foundation::ir::DataType::I4
        | vyre_foundation::ir::DataType::FP4
        | vyre_foundation::ir::DataType::NF4 => Some(1),
        vyre_foundation::ir::DataType::U16
        | vyre_foundation::ir::DataType::I16
        | vyre_foundation::ir::DataType::F16
        | vyre_foundation::ir::DataType::BF16 => Some(2),
        vyre_foundation::ir::DataType::Handle(_)
        | vyre_foundation::ir::DataType::DeviceMesh { .. } => Some(4),
        vyre_foundation::ir::DataType::I64 => Some(8),
        vyre_foundation::ir::DataType::Array { element_size } => Some(*element_size as usize),
        vyre_foundation::ir::DataType::Vec { element, count } => {
            fixed_scalar_storage_width(element)
                .and_then(|width| width.checked_mul(usize::from(*count)))
        }
        vyre_foundation::ir::DataType::TensorShaped { element, shape } => {
            let element_width = fixed_scalar_storage_width(element)?;
            shape
                .iter()
                .try_fold(element_width, |width, &dim| width.checked_mul(dim as usize))
        }
        vyre_foundation::ir::DataType::Quantized { storage, .. } => {
            fixed_scalar_storage_width(storage)
        }
        // Every remaining declared `DataType` is named rather than absorbed,
        // so adding a variant to the spec fails to compile here instead of
        // silently reporting no fixed storage width.
        vyre_foundation::ir::DataType::U32
        | vyre_foundation::ir::DataType::I32
        | vyre_foundation::ir::DataType::U64
        | vyre_foundation::ir::DataType::Vec2U32
        | vyre_foundation::ir::DataType::Vec4U32
        | vyre_foundation::ir::DataType::Bool
        | vyre_foundation::ir::DataType::Bytes
        | vyre_foundation::ir::DataType::F32
        | vyre_foundation::ir::DataType::F64
        | vyre_foundation::ir::DataType::Tensor
        | vyre_foundation::ir::DataType::SparseCsr { .. }
        | vyre_foundation::ir::DataType::SparseCoo { .. }
        | vyre_foundation::ir::DataType::SparseBsr { .. }
        | vyre_foundation::ir::DataType::Opaque(_) => None,
    }
}

fn extend_fixed_width(bytes: &[u8], declared_width: usize, out: &mut Vec<u8>) {
    if declared_width == 0 {
        out.extend_from_slice(bytes);
        return;
    }
    let copied = bytes.len().min(declared_width);
    out.extend_from_slice(&bytes[..copied]);
    out.resize(out.len() + (declared_width - copied), 0);
}

fn copy_bytes_prefix(bytes: &[u8], target: &mut [u8], cursor: &mut usize) {
    if *cursor >= target.len() {
        return;
    }
    let len = (target.len() - *cursor).min(bytes.len());
    target[*cursor..*cursor + len].copy_from_slice(&bytes[..len]);
    *cursor += len;
}

fn f64_to_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX)).then(|| value as u32)
}

fn f64_to_u64(value: f64) -> Option<u64> {
    const U64_EXCLUSIVE_MAX_AS_F64: f64 = 18_446_744_073_709_551_616.0;
    (value.is_finite() && value >= 0.0 && value < U64_EXCLUSIVE_MAX_AS_F64).then(|| value as u64)
}

impl From<Vec<u8>> for Value {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(Arc::from(bytes))
    }
}

impl From<&[u8]> for Value {
    fn from(bytes: &[u8]) -> Self {
        Self::Bytes(Arc::from(bytes))
    }
}

fn read_u32_prefix(bytes: &[u8]) -> u32 {
    let mut padded = [0u8; 4];
    let len = bytes.len().min(4);
    padded[..len].copy_from_slice(&bytes[..len]);
    u32::from_le_bytes(padded)
}

fn read_u64_prefix(bytes: &[u8]) -> u64 {
    let mut padded = [0u8; 8];
    let len = bytes.len().min(8);
    padded[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(padded)
}
