//! The generic schema member kind, its typed value, wire tag, and compatibility.
//!
//! A generic schema member is one value kind an external schema may declare for
//! a field. The kind is the unit every stage of the schema contract records a
//! decision for: the public API lists it, the wire format assigns it a stable
//! tag, the traversal decodes it into a typed value the visitor observes, the
//! validator parses and bounds-checks it, and the compatibility relation states
//! which declared members admit it across independently versioned consumers.
//!
//! [`FieldType::ALL`] is the roster every stage closure iterates. The
//! `variant-list-closure` gate compares that roster against the enum
//! declaration in this file, so a member added without extending the roster is
//! reported rather than skipped in silence.

use std::fmt;
use std::num::ParseIntError;

use super::schema::SchemaTranslationError;

/// Value data types for fields in an external dialect contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FieldType {
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE-754 32-bit float.
    F32,
    /// IEEE-754 64-bit float.
    F64,
    /// Boolean flag.
    Bool,
    /// UTF-8 string value.
    String,
    /// Opaque byte string.
    Bytes,
    /// Buffer identifier reference.
    Buffer,
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One integer width an external schema field can declare.
trait IntegerLiteral: Sized {
    /// Width name as it appears in a diagnostic.
    const NAME: &'static str;

    fn from_decimal(raw: &str) -> Result<Self, ParseIntError>;

    fn from_hex(digits: &str) -> Result<Self, ParseIntError>;
}

macro_rules! integer_literal {
    ($($ty:ty),+ $(,)?) => {$(
        impl IntegerLiteral for $ty {
            const NAME: &'static str = stringify!($ty);

            fn from_decimal(raw: &str) -> Result<Self, ParseIntError> {
                raw.parse()
            }

            fn from_hex(digits: &str) -> Result<Self, ParseIntError> {
                Self::from_str_radix(digits, 16)
            }
        }
    )+};
}

integer_literal!(u32, i32, u64, i64);

/// Accept a decimal literal or a `0x`-prefixed hexadecimal one, rejecting overflow.
fn decode_integer<T: IntegerLiteral>(raw: &str) -> Result<T, String> {
    let parsed = match raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        Some(digits) => T::from_hex(digits),
        None => T::from_decimal(raw),
    };
    parsed.map_err(|e| format!("invalid {} value `{raw}`: {e}", T::NAME))
}

/// Accept a finite decimal float, rejecting an infinity or a NaN.
fn decode_f32(raw: &str) -> Result<f32, String> {
    let parsed = raw
        .parse::<f32>()
        .map_err(|e| format!("invalid f32 value `{raw}`: {e}"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("f32 value `{raw}` is non-finite"))
    }
}

/// Accept a finite decimal float, rejecting an infinity or a NaN.
fn decode_f64(raw: &str) -> Result<f64, String> {
    let parsed = raw
        .parse::<f64>()
        .map_err(|e| format!("invalid f64 value `{raw}`: {e}"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("f64 value `{raw}` is non-finite"))
    }
}

impl FieldType {
    /// Every generic schema member kind an external schema can declare.
    pub const ALL: [Self; 10] = [
        Self::U32,
        Self::I32,
        Self::U64,
        Self::I64,
        Self::F32,
        Self::F64,
        Self::Bool,
        Self::String,
        Self::Bytes,
        Self::Buffer,
    ];

    /// Canonical lowercase name of this member kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Bytes => "bytes",
            Self::Buffer => "buffer",
        }
    }

    /// Stable wire tag identifying this member kind across schema versions.
    ///
    /// The tag is the identity a consumer built against another schema version
    /// reads, and it is what [`super::ExternalSchema::canonical_identity`]
    /// hashes, so a tag is assigned once and never reused for another member.
    #[must_use]
    pub const fn wire_tag(self) -> u16 {
        match self {
            Self::U32 => 1,
            Self::I32 => 2,
            Self::U64 => 3,
            Self::I64 => 4,
            Self::F32 => 5,
            Self::F64 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Bytes => 9,
            Self::Buffer => 10,
        }
    }

    /// Resolve a wire tag back to the member kind that carries it.
    ///
    /// The reverse table is [`FieldType::ALL`] read through
    /// [`FieldType::wire_tag`], so the two directions cannot disagree.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaTranslationError::UnknownMemberTag`] for a tag no
    /// declared member carries.
    pub fn from_wire_tag(
        dialect: &'static str,
        tag: u16,
    ) -> Result<Self, SchemaTranslationError> {
        Self::ALL
            .into_iter()
            .find(|member| member.wire_tag() == tag)
            .ok_or(SchemaTranslationError::UnknownMemberTag { dialect, tag })
    }

    /// Whether a field contract declaring `self` admits an external field
    /// declaring `external`.
    ///
    /// A member admits itself, and an integer or float member admits the
    /// narrower member of the same family, whose every value is representable
    /// without loss. Widening across families, and narrowing in any family, is
    /// refused: both change the value a consumer reads.
    #[must_use]
    pub const fn accepts(self, external: Self) -> bool {
        match self {
            Self::U32 => matches!(external, Self::U32),
            Self::I32 => matches!(external, Self::I32),
            Self::U64 => matches!(external, Self::U64 | Self::U32),
            Self::I64 => matches!(external, Self::I64 | Self::I32),
            Self::F32 => matches!(external, Self::F32),
            Self::F64 => matches!(external, Self::F64 | Self::F32),
            Self::Bool => matches!(external, Self::Bool),
            Self::String => matches!(external, Self::String),
            Self::Bytes => matches!(external, Self::Bytes),
            Self::Buffer => matches!(external, Self::Buffer),
        }
    }

    /// Decode a raw external field value into a typed value of this member kind.
    ///
    /// # Errors
    ///
    /// Returns a string describing the parse or bounds failure.
    pub fn decode(self, raw: &str) -> Result<FieldValue, String> {
        match self {
            Self::U32 => decode_integer::<u32>(raw).map(FieldValue::U32),
            Self::I32 => decode_integer::<i32>(raw).map(FieldValue::I32),
            Self::U64 => decode_integer::<u64>(raw).map(FieldValue::U64),
            Self::I64 => decode_integer::<i64>(raw).map(FieldValue::I64),
            Self::F32 => decode_f32(raw).map(FieldValue::F32),
            Self::F64 => decode_f64(raw).map(FieldValue::F64),
            Self::Bool => raw
                .parse::<bool>()
                .map(FieldValue::Bool)
                .map_err(|e| format!("invalid bool value `{raw}`: {e}")),
            Self::String => Ok(FieldValue::String(raw.to_string())),
            Self::Bytes => Ok(FieldValue::Bytes(raw.as_bytes().to_vec())),
            Self::Buffer => Ok(FieldValue::Buffer(raw.to_string())),
        }
    }

    /// Validate that a raw string value can be parsed into this field type without overflow.
    ///
    /// # Errors
    ///
    /// Returns a string describing the parse or bounds failure.
    pub fn parse_and_validate(&self, raw: &str) -> Result<(), String> {
        self.decode(raw).map(|_| ())
    }
}

/// A decoded external schema field value.
///
/// The traversal produces one of these for every field it visits, so a consumer
/// reads a typed value rather than re-parsing the raw text the external schema
/// carried.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    /// Unsigned 32-bit integer value.
    U32(u32),
    /// Signed 32-bit integer value.
    I32(i32),
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Signed 64-bit integer value.
    I64(i64),
    /// IEEE-754 32-bit float value.
    F32(f32),
    /// IEEE-754 64-bit float value.
    F64(f64),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string value.
    String(String),
    /// Opaque byte string value.
    Bytes(Vec<u8>),
    /// Buffer identifier reference.
    Buffer(String),
}

impl FieldValue {
    /// The member kind this value carries.
    #[must_use]
    pub const fn field_type(&self) -> FieldType {
        match self {
            Self::U32(_) => FieldType::U32,
            Self::I32(_) => FieldType::I32,
            Self::U64(_) => FieldType::U64,
            Self::I64(_) => FieldType::I64,
            Self::F32(_) => FieldType::F32,
            Self::F64(_) => FieldType::F64,
            Self::Bool(_) => FieldType::Bool,
            Self::String(_) => FieldType::String,
            Self::Bytes(_) => FieldType::Bytes,
            Self::Buffer(_) => FieldType::Buffer,
        }
    }
}

/// Validate that an external field's declared member is admissible for the
/// member a dialect field contract declares.
///
/// # Errors
///
/// Returns [`SchemaTranslationError::IncompatibleFieldMember`] when the
/// contract member does not accept the external member.
pub fn validate_member_compatibility(
    dialect: &'static str,
    node_op: &str,
    field: &str,
    contract_member: FieldType,
    external_member: FieldType,
) -> Result<(), SchemaTranslationError> {
    if contract_member.accepts(external_member) {
        return Ok(());
    }
    Err(SchemaTranslationError::IncompatibleFieldMember {
        dialect,
        node_op: node_op.to_string(),
        field: field.to_string(),
        contract_member,
        external_member,
    })
}
