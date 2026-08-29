//! Domain-neutral external schema, field, and resource ABI translation contracts.
//!
//! Provides validation and mapping from versioned external schemas into semantic dialect
//! operations and resource ABIs. Adheres to strict exhaustive closure: unknown fields,
//! duplicate fields, missing required fields, incomplete resource rosters, and incompatible
//! identities are rejected with actionable `Fix:` diagnostics.

use std::collections::BTreeSet;

use crate::ir::{BufferAccess, DataType};

/// Value data types for fields in an external dialect contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// IEEE-754 32-bit float.
    F32,
    /// Boolean flag.
    Bool,
    /// UTF-8 string value.
    String,
    /// Opaque byte string.
    Bytes,
    /// Buffer identifier reference.
    Buffer,
}

/// Declared field specification for an operation in a dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldContract {
    /// Field name.
    pub name: &'static str,
    /// Expected field type.
    pub field_type: FieldType,
    /// Whether this field must be present in the external schema.
    pub required: bool,
}

/// Resource binding declaration for an operation's ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBinding {
    /// Resource identifier / name.
    pub name: &'static str,
    /// Access mode (ReadOnly, WriteOnly, ReadWrite).
    pub access: BufferAccess,
    /// Data type of the elements in this resource.
    pub element_type: DataType,
    /// Minimum required alignment in bytes.
    pub alignment: u32,
    /// Minimum byte capacity required for valid execution.
    pub minimum_bytes: u64,
}

/// Resource ABI defining the memory and buffer contract for an operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceAbi {
    /// Declared resources required by the operation.
    pub resources: &'static [ResourceBinding],
}

impl ResourceAbi {
    /// Empty resource ABI with no buffer dependencies.
    pub const EMPTY: Self = Self { resources: &[] };

    /// Find a resource binding by name.
    #[must_use]
    pub fn find_resource(&self, name: &str) -> Option<&'static ResourceBinding> {
        self.resources.iter().find(|res| res.name == name)
    }

    /// Check if all required resources are present in the provided bound names.
    ///
    /// # Errors
    ///
    /// Returns the name of the first missing resource binding.
    pub fn verify_complete_roster(&self, bound: &[&str]) -> Result<(), &'static str> {
        for res in self.resources {
            if !bound.contains(&res.name) {
                return Err(res.name);
            }
        }
        Ok(())
    }
}

/// External schema node representation used to validate neutral external models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalSchemaNode {
    /// Operation name in the external schema.
    pub op_name: String,
    /// Raw field key-value pairs (ordered to detect duplicates).
    pub raw_fields: Vec<(String, String)>,
    /// Bound resource names provided by the external model.
    pub bound_resources: Vec<String>,
}

/// Errors arising during external schema translation and validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchemaTranslationError {
    /// An unknown field was present in the external schema.
    #[error("Unknown field `{field}` in external schema node `{node_op}` for dialect `{dialect}`. Fix: remove `{field}` or declare it in the dialect field contract.")]
    UnknownField {
        /// Owning dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Unknown field name.
        field: String,
    },
    /// A duplicate field was encountered in the external schema node.
    #[error("Duplicate field `{field}` in external schema node `{node_op}` for dialect `{dialect}`. Fix: deduplicate the field definition in the external schema payload.")]
    DuplicateField {
        /// Owning dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Duplicate field name.
        field: String,
    },
    /// A required field is missing from the external schema node.
    #[error("Missing required field `{field}` in external schema node `{node_op}` for dialect `{dialect}`. Fix: provide `{field}` with valid data in the external schema payload.")]
    MissingRequiredField {
        /// Owning dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Missing field name.
        field: String,
    },
    /// The resource roster is incomplete: a declared resource is not bound.
    #[error("Incomplete resource roster for external schema node `{node_op}` in dialect `{dialect}`: expected resource `{expected_resource}` is not bound. Fix: bind all declared resources ({declared:?}) in the external resource ABI.")]
    IncompleteResourceRoster {
        /// Owning dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Missing resource name.
        expected_resource: &'static str,
        /// List of all declared resources.
        declared: Vec<&'static str>,
    },
    /// The external schema identity is incompatible with the target dialect.
    #[error("Incompatible schema identity `{schema_id}` (version `{found_version}`) for dialect `{dialect}` (requires `{required_dialect}`, schema version `{expected_version}`). Fix: migrate external schema to dialect `{required_dialect}` version `{expected_version}`.")]
    IncompatibleIdentity {
        /// Target dialect identifier.
        dialect: &'static str,
        /// Found external schema identifier.
        schema_id: String,
        /// Expected dialect identifier.
        required_dialect: &'static str,
        /// Found schema version.
        found_version: u32,
        /// Expected schema version.
        expected_version: u32,
    },
    /// An external schema node has no corresponding operation in the dialect.
    #[error("Unmapped external schema node `{node_op}` in dialect `{dialect}` (exhaustive mapping required). Fix: map node `{node_op}` to a dialect operation or register a dialect extension.")]
    UnmappedNode {
        /// Dialect identifier.
        dialect: &'static str,
        /// Unmapped node operation name.
        node_op: String,
    },
}

/// Validate external schema node fields against a declared field contract.
///
/// # Errors
///
/// Returns [`SchemaTranslationError`] on unknown, duplicate, or missing required fields.
pub fn validate_node_fields(
    dialect: &'static str,
    node_op: &str,
    raw_fields: &[(String, String)],
    declared_fields: &[FieldContract],
) -> Result<(), SchemaTranslationError> {
    let mut seen_fields = BTreeSet::new();
    let declared_names: BTreeSet<&'static str> =
        declared_fields.iter().map(|f| f.name).collect();

    for (field_name, _) in raw_fields {
        if !declared_names.contains(field_name.as_str()) {
            return Err(SchemaTranslationError::UnknownField {
                dialect,
                node_op: node_op.to_string(),
                field: field_name.clone(),
            });
        }
        if !seen_fields.insert(field_name.as_str()) {
            return Err(SchemaTranslationError::DuplicateField {
                dialect,
                node_op: node_op.to_string(),
                field: field_name.clone(),
            });
        }
    }

    for contract in declared_fields {
        if contract.required && !seen_fields.contains(contract.name) {
            return Err(SchemaTranslationError::MissingRequiredField {
                dialect,
                node_op: node_op.to_string(),
                field: contract.name.to_string(),
            });
        }
    }

    Ok(())
}

/// Validate external schema resource bindings against an operation's ResourceAbi.
///
/// # Errors
///
/// Returns [`SchemaTranslationError::IncompleteResourceRoster`] when a declared resource is not bound.
pub fn validate_node_resources(
    dialect: &'static str,
    node_op: &str,
    bound_resources: &[String],
    abi: &ResourceAbi,
) -> Result<(), SchemaTranslationError> {
    let bound_slice: Vec<&str> = bound_resources.iter().map(String::as_str).collect();
    if let Err(missing) = abi.verify_complete_roster(&bound_slice) {
        let declared: Vec<&'static str> = abi.resources.iter().map(|r| r.name).collect();
        return Err(SchemaTranslationError::IncompleteResourceRoster {
            dialect,
            node_op: node_op.to_string(),
            expected_resource: missing,
            declared,
        });
    }
    Ok(())
}

/// Validate an external schema header identity.
///
/// # Errors
///
/// Returns [`SchemaTranslationError::IncompatibleIdentity`] if the schema id or version does not match.
pub fn validate_schema_identity(
    dialect: &'static str,
    found_schema_id: &str,
    found_version: u32,
    expected_dialect: &'static str,
    expected_version: u32,
) -> Result<(), SchemaTranslationError> {
    if found_schema_id != expected_dialect || found_version != expected_version {
        return Err(SchemaTranslationError::IncompatibleIdentity {
            dialect,
            schema_id: found_schema_id.to_string(),
            required_dialect: expected_dialect,
            found_version,
            expected_version,
        });
    }
    Ok(())
}
