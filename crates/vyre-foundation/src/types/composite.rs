//! Closed record and variant composite types.

use super::scalar::ScalarType;
use super::SemanticType;

/// One typed field in a record structure.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct RecordField {
    /// Field identifier.
    pub name: String,
    /// Field semantic type.
    pub ty: Box<SemanticType>,
    /// Optional explicit physical byte offset.
    pub offset_bytes: Option<u64>,
}

/// Closed record / product type with ordered, typed fields.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct RecordType {
    /// Fields in declaration order.
    pub fields: Vec<RecordField>,
}

impl RecordType {
    /// Create a new record type from field definitions.
    #[must_use]
    pub fn new(fields: Vec<RecordField>) -> Self {
        Self { fields }
    }

    /// Look up a field by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&RecordField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Number of declared fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether this record type is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// One tagged case in a discriminated variant / sum type.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct VariantCase {
    /// Variant case name.
    pub name: String,
    /// Discriminant integer tag.
    pub tag: u64,
    /// Optional payload type.
    pub payload: Option<Box<SemanticType>>,
}

/// Closed discriminated union / sum type.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct VariantType {
    /// Scalar type used for discriminant tag storage.
    pub discriminant_type: ScalarType,
    /// Declared variant cases.
    pub variants: Vec<VariantCase>,
}

impl VariantType {
    /// Create a new variant type.
    #[must_use]
    pub fn new(discriminant_type: ScalarType, variants: Vec<VariantCase>) -> Self {
        Self {
            discriminant_type,
            variants,
        }
    }

    /// Look up a variant case by name.
    #[must_use]
    pub fn case_by_name(&self, name: &str) -> Option<&VariantCase> {
        self.variants.iter().find(|v| v.name == name)
    }

    /// Look up a variant case by integer tag.
    #[must_use]
    pub fn case_by_tag(&self, tag: u64) -> Option<&VariantCase> {
        self.variants.iter().find(|v| v.tag == tag)
    }
}
