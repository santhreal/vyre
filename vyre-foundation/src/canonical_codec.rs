//! Canonical deterministic binary codec and verification for schema registry records.
//!
//! WHY: closes the class "schema codecs allow non-canonical bytes, duplicate fields,
//! or unchecked bounds to deserialize into valid-looking records".
//! Decodes and encodes with fixed-width types, canonical field numbers, hard size/depth bounds,
//! and domain-separated cryptographic hashing.

use core::fmt;
use std::string::String;
use std::vec::Vec;

use vyre_spec::{FieldType, SchemaId, SchemaRegistry};

/// Value representation for canonical record fields.
#[derive(Clone, Debug, PartialEq)]
pub enum CanonicalValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 32-bit integer.
    I32(i32),
    /// Signed 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// Fixed-length byte vector.
    FixedBytes(Vec<u8>),
    /// Variable byte vector.
    VarBytes(Vec<u8>),
    /// UTF-8 string.
    Utf8String(String),
    /// List of canonical values.
    List(Vec<CanonicalValue>),
}

/// A structured canonical record mapping schema fields to values.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalRecord {
    /// Schema identifier for this record.
    pub schema_id: SchemaId,
    /// Field values stored by field number (1-based).
    pub fields: Vec<(u32, CanonicalValue)>,
}

/// Error returned during canonical encoding or decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Unknown schema id.
    UnknownSchema(SchemaId),
    /// Record exceeds maximum allowed payload bytes.
    PayloadOversized {
        /// Size in bytes.
        size: usize,
        /// Limit in bytes.
        limit: usize,
    },
    /// Record exceeds maximum allowed nesting depth.
    DepthExceeded {
        /// Nesting depth observed.
        depth: usize,
        /// Nesting depth limit.
        limit: usize,
    },
    /// List field exceeds maximum allowed element count.
    ElementCountExceeded {
        /// Count observed.
        count: usize,
        /// Element limit.
        limit: usize,
    },
    /// Field numbers are out of order.
    NonCanonicalFieldOrder {
        /// Field number expected after previous field.
        expected_after: u32,
        /// Field number received.
        got: u32,
    },
    /// Duplicate field key was detected.
    DuplicateKey {
        /// Duplicate field number.
        field_number: u32,
    },
    /// Non-canonical encoding representation was encountered (e.g. non-boolean byte, non-normalized float).
    NonCanonicalEncoding {
        /// Field number.
        field_number: u32,
        /// Error detail.
        details: &'static str,
    },
    /// Type of field does not match declared schema.
    TypeMismatch {
        /// Field number.
        field_number: u32,
        /// Field name.
        field_name: &'static str,
    },
    /// Required field was not present in the record.
    MissingRequiredField {
        /// Field number.
        field_number: u32,
        /// Field name.
        field_name: &'static str,
    },
    /// Truncated bytes encountered while decoding.
    UnexpectedEof {
        /// Expected bytes.
        expected: usize,
        /// Remaining bytes.
        remaining: usize,
    },
    /// Invalid UTF-8 encoding in string field.
    InvalidUtf8 {
        /// Field number.
        field_number: u32,
    },
    /// Stale schema version or obsolete fixture rejected by name.
    StaleSchemaVersion {
        /// Schema ID.
        schema_id: SchemaId,
        /// Stale version string found.
        found: String,
    },
    /// Signature verification failed under domain separator.
    SignatureVerificationFailed,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSchema(id) => write!(f, "Fix: unknown schema id {id:?}"),
            Self::PayloadOversized { size, limit } => write!(
                f,
                "Fix: record size {size} bytes exceeds declared schema bound of {limit} bytes"
            ),
            Self::DepthExceeded { depth, limit } => write!(
                f,
                "Fix: record nesting depth {depth} exceeds declared schema limit of {limit}"
            ),
            Self::ElementCountExceeded { count, limit } => write!(
                f,
                "Fix: list element count {count} exceeds declared schema limit of {limit}"
            ),
            Self::NonCanonicalFieldOrder { expected_after, got } => write!(
                f,
                "Fix: non-canonical field ordering; field {got} appeared after {expected_after} (fields must be strictly ascending)"
            ),
            Self::DuplicateKey { field_number } => write!(
                f,
                "Fix: duplicate field key {field_number} detected; duplicate keys are forbidden"
            ),
            Self::NonCanonicalEncoding { field_number, details } => write!(
                f,
                "Fix: non-canonical encoding for field {field_number}: {details}"
            ),
            Self::TypeMismatch { field_number, field_name } => write!(
                f,
                "Fix: type mismatch for field {field_number} ('{field_name}')"
            ),
            Self::MissingRequiredField { field_number, field_name } => write!(
                f,
                "Fix: missing required field {field_number} ('{field_name}')"
            ),
            Self::UnexpectedEof { expected, remaining } => write!(
                f,
                "Fix: truncated payload; expected {expected} bytes but only {remaining} remain"
            ),
            Self::InvalidUtf8 { field_number } => write!(
                f,
                "Fix: invalid UTF-8 bytes in field {field_number}"
            ),
            Self::StaleSchemaVersion { schema_id, found } => write!(
                f,
                "Fix: stale schema version `{found}` rejected for {schema_id:?}; migrate to current schema"
            ),
            Self::SignatureVerificationFailed => write!(
                f,
                "Fix: cryptographic signature verification failed under schema domain separator"
            ),
        }
    }
}


/// Canonical binary encoder for schema records.
pub struct CanonicalEncoder;

impl CanonicalEncoder {
    /// Encode a canonical record into deterministic bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if validation against the declared schema fails.
    pub fn encode(record: &CanonicalRecord) -> Result<Vec<u8>, CodecError> {
        let schema = SchemaRegistry::lookup(record.schema_id)
            .ok_or(CodecError::UnknownSchema(record.schema_id))?;

        let mut out = Vec::new();
        // Magic header: "VYRE" + schema_id (u32) + semver (u32, u32, u32)
        out.extend_from_slice(b"VYRE");
        out.extend_from_slice(&(record.schema_id as u32).to_le_bytes());
        out.extend_from_slice(&schema.semver.major.to_le_bytes());
        out.extend_from_slice(&schema.semver.minor.to_le_bytes());
        out.extend_from_slice(&schema.semver.patch.to_le_bytes());

        let mut last_field_num = 0;
        for (field_num, val) in &record.fields {
            if *field_num == last_field_num {
                return Err(CodecError::DuplicateKey {
                    field_number: *field_num,
                });
            }
            if *field_num < last_field_num {
                return Err(CodecError::NonCanonicalFieldOrder {
                    expected_after: last_field_num,
                    got: *field_num,
                });
            }
            last_field_num = *field_num;

            // Check for stale fixture versions in string fields
            if let CanonicalValue::Utf8String(s) = val {
                if schema.stale_fixtures.iter().any(|&stale| stale == s.as_str()) {
                    return Err(CodecError::StaleSchemaVersion {
                        schema_id: record.schema_id,
                        found: s.clone(),
                    });
                }
            }

            // Find field def in schema
            let field_def = schema
                .fields
                .iter()
                .find(|f| f.number == *field_num)
                .ok_or(CodecError::NonCanonicalFieldOrder {
                    expected_after: last_field_num,
                    got: *field_num,
                })?;

            out.extend_from_slice(&field_num.to_le_bytes());
            Self::encode_value(val, field_def.field_type, schema, 1, &mut out)?;
        }
        // Verify all required fields were encoded
        for req_field in schema.fields.iter().filter(|f| f.required) {
            if !record.fields.iter().any(|(num, _)| *num == req_field.number) {
                return Err(CodecError::MissingRequiredField {
                    field_number: req_field.number,
                    field_name: req_field.name,
                });
            }
        }

        if out.len() > schema.bounds.max_bytes {
            return Err(CodecError::PayloadOversized {
                size: out.len(),
                limit: schema.bounds.max_bytes,
            });
        }

        Ok(out)
    }

    fn encode_value(
        val: &CanonicalValue,
        expected_type: FieldType,
        schema: &vyre_spec::SchemaDefinition,
        depth: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        if depth > schema.bounds.max_depth {
            return Err(CodecError::DepthExceeded {
                depth,
                limit: schema.bounds.max_depth,
            });
        }
        match (val, &expected_type) {
            (CanonicalValue::U8(v), FieldType::U8) => out.push(*v),
            (CanonicalValue::U16(v), FieldType::U16) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::U32(v), FieldType::U32) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::U64(v), FieldType::U64) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::I32(v), FieldType::I32) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::I64(v), FieldType::I64) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::F32(v), FieldType::F32) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::F64(v), FieldType::F64) => out.extend_from_slice(&v.to_le_bytes()),
            (CanonicalValue::Bool(v), FieldType::Bool) => out.push(if *v { 1 } else { 0 }),
            (CanonicalValue::FixedBytes(b), FieldType::FixedBytes(n)) => {
                if b.len() != *n {
                    return Err(CodecError::TypeMismatch {
                        field_number: 0,
                        field_name: "FixedBytes length mismatch",
                    });
                }
                out.extend_from_slice(b.as_slice());
            }
            (CanonicalValue::VarBytes(b), FieldType::VarBytes) => {
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(b.as_slice());
            }
            (CanonicalValue::Utf8String(s), FieldType::Utf8String) => {
                let bytes = s.as_bytes();
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(bytes);
            }
            (CanonicalValue::List(items), FieldType::List(elem_type)) => {
                if items.len() > schema.bounds.max_elements {
                    return Err(CodecError::ElementCountExceeded {
                        count: items.len(),
                        limit: schema.bounds.max_elements,
                    });
                }
                out.extend_from_slice(&(items.len() as u32).to_le_bytes());
                for item in items.iter() {
                    Self::encode_value(item, **elem_type, schema, depth + 1, out)?;
                }
            }
            _ => {
                return Err(CodecError::TypeMismatch {
                    field_number: 0,
                    field_name: "value type does not match schema field type",
                });
            }
        }
        Ok(())
    }
}

/// Canonical binary decoder for schema records.
pub struct CanonicalDecoder;

impl CanonicalDecoder {
    /// Decode deterministic canonical bytes into a typed [`CanonicalRecord`].
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if payload is malformed, oversized, truncated, or out of order.
    pub fn decode(bytes: &[u8]) -> Result<CanonicalRecord, CodecError> {
        if bytes.len() < 20 {
            return Err(CodecError::UnexpectedEof {
                expected: 20,
                remaining: bytes.len(),
            });
        }
        if &bytes[0..4] != b"VYRE" {
            return Err(CodecError::UnknownSchema(SchemaId::ConformanceCertificate));
        }

        let schema_id_raw = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let schema_id = SchemaId::ALL
            .iter()
            .copied()
            .find(|&s| (s as u32) == schema_id_raw)
            .ok_or(CodecError::UnknownSchema(SchemaId::ConformanceCertificate))?;

        let schema = SchemaRegistry::lookup(schema_id)
            .ok_or(CodecError::UnknownSchema(schema_id))?;

        if bytes.len() > schema.bounds.max_bytes {
            return Err(CodecError::PayloadOversized {
                size: bytes.len(),
                limit: schema.bounds.max_bytes,
            });
        }

        let mut offset = 20; // 4 magic + 4 schema_id + 12 semver
        let mut fields = Vec::new();
        let mut last_field_num = 0;

        while offset < bytes.len() {
            if offset + 4 > bytes.len() {
                return Err(CodecError::UnexpectedEof {
                    expected: 4,
                    remaining: bytes.len() - offset,
                });
            }
            let field_num = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;

            if field_num == last_field_num {
                return Err(CodecError::DuplicateKey {
                    field_number: field_num,
                });
            }
            if field_num < last_field_num {
                return Err(CodecError::NonCanonicalFieldOrder {
                    expected_after: last_field_num,
                    got: field_num,
                });
            }
            last_field_num = field_num;

            let field_def = schema
                .fields
                .iter()
                .find(|f| f.number == field_num)
                .ok_or(CodecError::NonCanonicalFieldOrder {
                    expected_after: last_field_num,
                    got: field_num,
                })?;

            let (val, consumed) = Self::decode_value(&bytes[offset..], field_def.field_type, field_num, schema, 1)?;
            offset += consumed;

            // Check for stale version strings
            if let CanonicalValue::Utf8String(s) = &val {
                if schema.stale_fixtures.iter().any(|&stale| stale == s.as_str()) {
                    return Err(CodecError::StaleSchemaVersion {
                        schema_id,
                        found: s.clone(),
                    });
                }
            }

            fields.push((field_num, val));
        }

        // Verify required fields
        for req_field in schema.fields.iter().filter(|f| f.required) {
            if !fields.iter().any(|(num, _)| *num == req_field.number) {
                return Err(CodecError::MissingRequiredField {
                    field_number: req_field.number,
                    field_name: req_field.name,
                });
            }
        }

        Ok(CanonicalRecord { schema_id, fields })
    }

    fn decode_value(
        bytes: &[u8],
        field_type: FieldType,
        field_num: u32,
        schema: &vyre_spec::SchemaDefinition,
        depth: usize,
    ) -> Result<(CanonicalValue, usize), CodecError> {
        if depth > schema.bounds.max_depth {
            return Err(CodecError::DepthExceeded {
                depth,
                limit: schema.bounds.max_depth,
            });
        }
        match field_type {
            FieldType::U8 => {
                if bytes.is_empty() {
                    return Err(CodecError::UnexpectedEof { expected: 1, remaining: 0 });
                }
                Ok((CanonicalValue::U8(bytes[0]), 1))
            }
            FieldType::U16 => {
                if bytes.len() < 2 {
                    return Err(CodecError::UnexpectedEof { expected: 2, remaining: bytes.len() });
                }
                Ok((CanonicalValue::U16(u16::from_le_bytes(bytes[0..2].try_into().unwrap())), 2))
            }
            FieldType::U32 => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                Ok((CanonicalValue::U32(u32::from_le_bytes(bytes[0..4].try_into().unwrap())), 4))
            }
            FieldType::U64 => {
                if bytes.len() < 8 {
                    return Err(CodecError::UnexpectedEof { expected: 8, remaining: bytes.len() });
                }
                Ok((CanonicalValue::U64(u64::from_le_bytes(bytes[0..8].try_into().unwrap())), 8))
            }
            FieldType::I32 => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                Ok((CanonicalValue::I32(i32::from_le_bytes(bytes[0..4].try_into().unwrap())), 4))
            }
            FieldType::I64 => {
                if bytes.len() < 8 {
                    return Err(CodecError::UnexpectedEof { expected: 8, remaining: bytes.len() });
                }
                Ok((CanonicalValue::I64(i64::from_le_bytes(bytes[0..8].try_into().unwrap())), 8))
            }
            FieldType::F32 => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                Ok((CanonicalValue::F32(f32::from_le_bytes(bytes[0..4].try_into().unwrap())), 4))
            }
            FieldType::F64 => {
                if bytes.len() < 8 {
                    return Err(CodecError::UnexpectedEof { expected: 8, remaining: bytes.len() });
                }
                Ok((CanonicalValue::F64(f64::from_le_bytes(bytes[0..8].try_into().unwrap())), 8))
            }
            FieldType::Bool => {
                if bytes.is_empty() {
                    return Err(CodecError::UnexpectedEof { expected: 1, remaining: 0 });
                }
                if bytes[0] > 1 {
                    return Err(CodecError::NonCanonicalEncoding {
                        field_number: field_num,
                        details: "boolean byte must be 0 or 1",
                    });
                }
                Ok((CanonicalValue::Bool(bytes[0] == 1), 1))
            }
            FieldType::FixedBytes(n) => {
                if bytes.len() < n {
                    return Err(CodecError::UnexpectedEof { expected: n, remaining: bytes.len() });
                }
                Ok((CanonicalValue::FixedBytes(bytes[0..n].to_vec()), n))
            }
            FieldType::VarBytes => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                if bytes.len() < 4 + len {
                    return Err(CodecError::UnexpectedEof { expected: 4 + len, remaining: bytes.len() });
                }
                Ok((CanonicalValue::VarBytes(bytes[4..4 + len].to_vec()), 4 + len))
            }
            FieldType::Utf8String => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                if bytes.len() < 4 + len {
                    return Err(CodecError::UnexpectedEof { expected: 4 + len, remaining: bytes.len() });
                }
                let s = core::str::from_utf8(&bytes[4..4 + len])
                    .map_err(|_| CodecError::InvalidUtf8 { field_number: field_num })?;
                Ok((CanonicalValue::Utf8String(String::from(s)), 4 + len))
            }
            FieldType::List(elem_type) => {
                if bytes.len() < 4 {
                    return Err(CodecError::UnexpectedEof { expected: 4, remaining: bytes.len() });
                }
                let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                if count > schema.bounds.max_elements {
                    return Err(CodecError::ElementCountExceeded {
                        count,
                        limit: schema.bounds.max_elements,
                    });
                }
                let mut offset = 4;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    let (item, consumed) = Self::decode_value(&bytes[offset..], *elem_type, field_num, schema, depth + 1)?;
                    offset += consumed;
                    items.push(item);
                }
                Ok((CanonicalValue::List(items), offset))
            }
            _ => Err(CodecError::TypeMismatch {
                field_number: field_num,
                field_name: "unsupported or future field type",
            }),
        }
    }
}

/// Canonical record signer.
pub struct CanonicalSigner;

impl CanonicalSigner {
    /// Compute the cryptographic digest over identity fields of a canonical record.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if encoding fails.
    pub fn compute_identity_digest(record: &CanonicalRecord) -> Result<[u8; 32], CodecError> {
        let schema = SchemaRegistry::lookup(record.schema_id)
            .ok_or(CodecError::UnknownSchema(record.schema_id))?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(schema.domain_separator.as_bytes());
        hasher.update(&(record.schema_id as u32).to_le_bytes());

        for field_def in schema.fields.iter().filter(|f| f.is_identity) {
            if let Some((_, val)) = record.fields.iter().find(|(num, _)| *num == field_def.number) {
                let mut buf = Vec::new();
                CanonicalEncoder::encode_value(val, field_def.field_type, schema, 1, &mut buf)?;
                hasher.update(&field_def.number.to_le_bytes());
                hasher.update(&buf);
            }
        }

        Ok(*hasher.finalize().as_bytes())
    }
}
