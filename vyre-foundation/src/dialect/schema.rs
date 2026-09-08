//! Domain-neutral external schema, field, resource, layout, and translation contracts.
//!
//! Provides validation and mapping from versioned external schemas into semantic dialect
//! operations, layouts, and resource ABIs. Adheres to strict exhaustive closure: unknown
//! fields, duplicate fields, missing required fields, overflowing values, unused declared
//! resources/layouts, incomplete resource rosters, and incompatible identities are rejected
//! with actionable `Fix:` diagnostics before compilation.

use std::collections::BTreeSet;

use crate::ir::{BufferAccess, DataType};

/// Value data types for fields in an external dialect contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl FieldType {
    /// Validate that a raw string value can be parsed into this field type without overflow.
    ///
    /// # Errors
    ///
    /// Returns a string describing the parse or bounds failure.
    pub fn parse_and_validate(&self, raw: &str) -> Result<(), String> {
        match self {
            Self::U32 => raw
                .parse::<u32>()
                .map(|_| ())
                .map_err(|e| format!("invalid u32 value `{raw}`: {e}")),
            Self::I32 => raw
                .parse::<i32>()
                .map(|_| ())
                .map_err(|e| format!("invalid i32 value `{raw}`: {e}")),
            Self::U64 => raw
                .parse::<u64>()
                .map(|_| ())
                .map_err(|e| format!("invalid u64 value `{raw}`: {e}")),
            Self::I64 => raw
                .parse::<i64>()
                .map(|_| ())
                .map_err(|e| format!("invalid i64 value `{raw}`: {e}")),
            Self::F32 => raw
                .parse::<f32>()
                .map_err(|e| format!("invalid f32 value `{raw}`: {e}"))
                .and_then(|f| {
                    if f.is_finite() {
                        Ok(())
                    } else {
                        Err(format!("f32 value `{raw}` is non-finite"))
                    }
                }),
            Self::F64 => raw
                .parse::<f64>()
                .map_err(|e| format!("invalid f64 value `{raw}`: {e}"))
                .and_then(|f| {
                    if f.is_finite() {
                        Ok(())
                    } else {
                        Err(format!("f64 value `{raw}` is non-finite"))
                    }
                }),
            Self::Bool => raw
                .parse::<bool>()
                .map(|_| ())
                .map_err(|e| format!("invalid bool value `{raw}`: {e}")),
            Self::String | Self::Bytes | Self::Buffer => Ok(()),
        }
    }
}

/// Declared field specification for an operation in a dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldContract {
    /// Field name.
    pub name: &'static str,
    /// Expected field type.
    pub field_type: FieldType,
    /// Whether this field must be present in the external schema.
    pub required: bool,
}

/// Resource binding declaration for an operation's ABI.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// Layout contract describing tensor/buffer shape, strides, and memory packing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayoutContract {
    /// Layout name or identifier.
    pub name: &'static str,
    /// Element data type.
    pub element_type: DataType,
    /// Dimensions of the layout extent.
    pub shape: &'static [u64],
    /// Strides along each dimension in element units.
    pub strides: &'static [u64],
    /// Byte alignment.
    pub alignment: u32,
    /// Whether the layout is contiguous in memory.
    pub contiguous: bool,
}

impl LayoutContract {
    /// Compute minimum buffer capacity required by this layout in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaTranslationError::OverflowingLayout`] if extent math overflows `u64`.
    pub fn compute_capacity(&self, dialect: &'static str) -> Result<u64, SchemaTranslationError> {
        let elem_size = self.element_type.min_bytes() as u64;
        let mut max_offset = 0_u64;
        for (&d, &s) in self.shape.iter().zip(self.strides.iter()) {
            if d == 0 {
                continue;
            }
            let span = (d - 1).checked_mul(s).ok_or_else(|| {
                SchemaTranslationError::OverflowingLayout {
                    dialect,
                    resource: self.name.to_string(),
                }
            })?;
            max_offset = max_offset.checked_add(span).ok_or_else(|| {
                SchemaTranslationError::OverflowingLayout {
                    dialect,
                    resource: self.name.to_string(),
                }
            })?;
        }
        let bytes = (max_offset.checked_add(1).ok_or_else(|| {
            SchemaTranslationError::OverflowingLayout {
                dialect,
                resource: self.name.to_string(),
            }
        })?)
        .checked_mul(elem_size)
        .ok_or_else(|| SchemaTranslationError::OverflowingLayout {
            dialect,
            resource: self.name.to_string(),
        })?;
        Ok(bytes)
    }
}

/// Layout binding association between a declared resource and its layout contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayoutBinding {
    /// Resource identifier.
    pub resource_name: &'static str,
    /// Layout contract specification.
    pub layout: LayoutContract,
}

/// Resource ABI defining the memory and buffer contract for an operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceAbi {
    /// Declared resources required by the operation.
    pub resources: &'static [ResourceBinding],
    /// Declared layout contracts for resources.
    pub layouts: &'static [LayoutBinding],
}

impl ResourceAbi {
    /// Empty resource ABI with no buffer dependencies.
    pub const EMPTY: Self = Self {
        resources: &[],
        layouts: &[],
    };

    /// Construct a resource ABI with resources only.
    #[must_use]
    pub const fn with_resources(resources: &'static [ResourceBinding]) -> Self {
        Self {
            resources,
            layouts: &[],
        }
    }

    /// Construct a resource ABI with resources and layout bindings.
    #[must_use]
    pub const fn with_resources_and_layouts(
        resources: &'static [ResourceBinding],
        layouts: &'static [LayoutBinding],
    ) -> Self {
        Self { resources, layouts }
    }

    /// Find a resource binding by name.
    #[must_use]
    pub fn find_resource(&self, name: &str) -> Option<&'static ResourceBinding> {
        self.resources.iter().find(|res| res.name == name)
    }

    /// Find a layout contract for a resource by name.
    #[must_use]
    pub fn find_layout(&self, resource_name: &str) -> Option<&'static LayoutContract> {
        self.layouts
            .iter()
            .find(|b| b.resource_name == resource_name)
            .map(|b| &b.layout)
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

/// External resource declaration in a complete external schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalResourceDeclaration {
    /// Resource name.
    pub name: String,
    /// Buffer access mode.
    pub access: BufferAccess,
    /// Data type.
    pub element_type: DataType,
    /// Byte alignment.
    pub alignment: u32,
    /// Byte capacity.
    pub byte_capacity: u64,
}

/// External layout declaration in a complete external schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalLayoutDeclaration {
    /// Target resource name.
    pub resource_name: String,
    /// Element data type.
    pub element_type: DataType,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Strides along each dimension.
    pub strides: Vec<u64>,
    /// Byte alignment.
    pub alignment: u32,
}

impl ExternalLayoutDeclaration {
    /// Compute extent bytes required by this layout declaration.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaTranslationError::OverflowingLayout`] on extent arithmetic overflow.
    pub fn compute_extent_bytes(&self, dialect: &'static str) -> Result<u64, SchemaTranslationError> {
        let elem_size = self.element_type.min_bytes() as u64;
        let mut max_offset = 0_u64;
        for (&d, &s) in self.shape.iter().zip(self.strides.iter()) {
            if d == 0 {
                continue;
            }
            let span = (d - 1).checked_mul(s).ok_or_else(|| {
                SchemaTranslationError::OverflowingLayout {
                    dialect,
                    resource: self.resource_name.clone(),
                }
            })?;
            max_offset = max_offset.checked_add(span).ok_or_else(|| {
                SchemaTranslationError::OverflowingLayout {
                    dialect,
                    resource: self.resource_name.clone(),
                }
            })?;
        }
        let bytes = (max_offset.checked_add(1).ok_or_else(|| {
            SchemaTranslationError::OverflowingLayout {
                dialect,
                resource: self.resource_name.clone(),
            }
        })?)
        .checked_mul(elem_size)
        .ok_or_else(|| SchemaTranslationError::OverflowingLayout {
            dialect,
            resource: self.resource_name.clone(),
        })?;
        Ok(bytes)
    }
}

/// Versioned domain-neutral external schema container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalSchema {
    /// Dialect / schema identifier.
    pub schema_id: String,
    /// Schema format version.
    pub version: u32,
    /// Operation nodes in the schema.
    pub nodes: Vec<ExternalSchemaNode>,
    /// Declared external resources.
    pub declared_resources: Vec<ExternalResourceDeclaration>,
    /// Declared layout specifications.
    pub declared_layouts: Vec<ExternalLayoutDeclaration>,
}

impl ExternalSchema {
    /// Compute deterministic blake3 canonical identity hash of the schema.
    #[must_use]
    pub fn canonical_identity(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre.external_schema.v1\0");
        hasher.update(self.schema_id.as_bytes());
        hasher.update(&[0]);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&(self.nodes.len() as u64).to_le_bytes());
        for node in &self.nodes {
            hasher.update(node.op_name.as_bytes());
            hasher.update(&[0]);
            hasher.update(&(node.raw_fields.len() as u64).to_le_bytes());
            for (k, v) in &node.raw_fields {
                hasher.update(k.as_bytes());
                hasher.update(&[0]);
                hasher.update(v.as_bytes());
                hasher.update(&[0]);
            }
            hasher.update(&(node.bound_resources.len() as u64).to_le_bytes());
            for res in &node.bound_resources {
                hasher.update(res.as_bytes());
                hasher.update(&[0]);
            }
        }
        hasher.update(&(self.declared_resources.len() as u64).to_le_bytes());
        for res in &self.declared_resources {
            hasher.update(res.name.as_bytes());
            hasher.update(&[0]);
            hasher.update(format!("{:?}", res.access).as_bytes());
            hasher.update(&[0]);
            hasher.update(format!("{:?}", res.element_type).as_bytes());
            hasher.update(&[0]);
            hasher.update(&res.alignment.to_le_bytes());
            hasher.update(&res.byte_capacity.to_le_bytes());
        }
        hasher.update(&(self.declared_layouts.len() as u64).to_le_bytes());
        for layout in &self.declared_layouts {
            hasher.update(layout.resource_name.as_bytes());
            hasher.update(&[0]);
            hasher.update(format!("{:?}", layout.element_type).as_bytes());
            hasher.update(&[0]);
            hasher.update(&layout.alignment.to_le_bytes());
            hasher.update(&(layout.shape.len() as u64).to_le_bytes());
            for dim in &layout.shape {
                hasher.update(&dim.to_le_bytes());
            }
            hasher.update(&(layout.strides.len() as u64).to_le_bytes());
            for stride in &layout.strides {
                hasher.update(&stride.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Exhaustively visit every component in this schema.
    pub fn accept<V: ExternalSchemaVisitor>(&self, visitor: &mut V) -> Result<(), V::Error> {
        visitor.visit_schema(&self.schema_id, self.version)?;
        for res in &self.declared_resources {
            visitor.visit_resource_declaration(res)?;
        }
        for layout in &self.declared_layouts {
            visitor.visit_layout_declaration(layout)?;
        }
        for node in &self.nodes {
            visitor.visit_node(node)?;
            for (k, v) in &node.raw_fields {
                visitor.visit_field(&node.op_name, k, v)?;
            }
            for res in &node.bound_resources {
                visitor.visit_resource_binding(&node.op_name, res)?;
            }
        }
        Ok(())
    }
}

/// Exhaustive visitor trait over external schema components.
pub trait ExternalSchemaVisitor {
    /// Error type produced by the visitor.
    type Error;

    /// Visit schema header.
    fn visit_schema(&mut self, schema_id: &str, version: u32) -> Result<(), Self::Error>;

    /// Visit a schema operation node.
    fn visit_node(&mut self, node: &ExternalSchemaNode) -> Result<(), Self::Error>;

    /// Visit a field key-value pair within a node.
    fn visit_field(
        &mut self,
        node_op: &str,
        field_name: &str,
        field_value: &str,
    ) -> Result<(), Self::Error>;

    /// Visit a bound resource name within a node.
    fn visit_resource_binding(
        &mut self,
        node_op: &str,
        resource_name: &str,
    ) -> Result<(), Self::Error>;

    /// Visit an external resource declaration.
    fn visit_resource_declaration(
        &mut self,
        resource: &ExternalResourceDeclaration,
    ) -> Result<(), Self::Error>;

    /// Visit an external layout declaration.
    fn visit_layout_declaration(
        &mut self,
        layout: &ExternalLayoutDeclaration,
    ) -> Result<(), Self::Error>;
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
    /// A field value overflows or violates type bounds.
    #[error("Overflowing value `{value}` for field `{field}` of type `{field_type:?}` in node `{node_op}` for dialect `{dialect}`: {reason}. Fix: ensure field values fit within declared type range.")]
    OverflowingField {
        /// Dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Field name.
        field: String,
        /// Declared field type.
        field_type: FieldType,
        /// Raw value.
        value: String,
        /// Reason for overflow/parse rejection.
        reason: String,
    },
    /// A layout specification overflows arithmetic bounds.
    #[error("Overflowing layout for resource `{resource}` in dialect `{dialect}`: extent or capacity computation overflows u64. Fix: check dimensions and stride parameters.")]
    OverflowingLayout {
        /// Dialect identifier.
        dialect: &'static str,
        /// Resource name.
        resource: String,
    },
    /// A declared resource is not used/bound by any node in the schema.
    #[error("Unused declared resource `{resource}` in external schema `{schema_id}`. Fix: bind the declared resource to a schema node or remove unused declaration.")]
    UnusedResource {
        /// Schema identifier.
        schema_id: String,
        /// Unused resource name.
        resource: String,
    },
    /// A declared layout is not used/bound by any node in the schema.
    #[error("Unused declared layout for resource `{resource}` in external schema `{schema_id}`. Fix: bind the resource or remove unused layout declaration.")]
    UnusedLayout {
        /// Schema identifier.
        schema_id: String,
        /// Unused resource name.
        resource: String,
    },
    /// A layout's element type is incompatible with its declared resource.
    #[error("Incompatible layout for resource `{resource}` in dialect `{dialect}`: element type `{declared_type:?}` does not match resource element type `{resource_type:?}`. Fix: align layout element type with resource declaration.")]
    IncompatibleLayout {
        /// Dialect identifier.
        dialect: &'static str,
        /// Resource name.
        resource: String,
        /// Layout element type.
        declared_type: DataType,
        /// Resource element type.
        resource_type: DataType,
    },
}

/// Validate external schema node fields against a declared field contract.
///
/// # Errors
///
/// Returns [`SchemaTranslationError`] on unknown, duplicate, missing required, or overflowing fields.
pub fn validate_node_fields(
    dialect: &'static str,
    node_op: &str,
    raw_fields: &[(String, String)],
    declared_fields: &[FieldContract],
) -> Result<(), SchemaTranslationError> {
    let mut seen_fields = BTreeSet::new();
    let declared_names: BTreeSet<&'static str> = declared_fields.iter().map(|f| f.name).collect();

    for (field_name, field_val) in raw_fields {
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
        if let Some(contract) = declared_fields.iter().find(|f| f.name == field_name.as_str()) {
            if let Err(reason) = contract.field_type.parse_and_validate(field_val) {
                return Err(SchemaTranslationError::OverflowingField {
                    dialect,
                    node_op: node_op.to_string(),
                    field: field_name.clone(),
                    field_type: contract.field_type,
                    value: field_val.clone(),
                    reason,
                });
            }
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

/// Validate a complete external schema against a dialect specification.
///
/// Checks schema identity, validates all nodes through `node_validator`, verifies
/// all declared resources and layouts are used without dangling unused members, and
/// checks layout compatibility and bounds.
///
/// # Errors
///
/// Returns [`SchemaTranslationError`] on any validation failure.
pub fn validate_external_schema<F>(
    dialect: &'static str,
    schema: &ExternalSchema,
    expected_version: u32,
    mut node_validator: F,
) -> Result<BTreeSet<String>, SchemaTranslationError>
where
    F: FnMut(&ExternalSchemaNode) -> Result<(), SchemaTranslationError>,
{
    validate_schema_identity(
        dialect,
        &schema.schema_id,
        schema.version,
        dialect,
        expected_version,
    )?;

    let mut bound_resources = BTreeSet::new();

    for node in &schema.nodes {
        node_validator(node)?;
        for res in &node.bound_resources {
            bound_resources.insert(res.clone());
        }
    }

    let declared_resource_names: BTreeSet<String> = schema
        .declared_resources
        .iter()
        .map(|r| r.name.clone())
        .collect();

    for decl in &schema.declared_resources {
        if !bound_resources.contains(&decl.name) {
            return Err(SchemaTranslationError::UnusedResource {
                schema_id: schema.schema_id.clone(),
                resource: decl.name.clone(),
            });
        }
    }

    for layout in &schema.declared_layouts {
        if !declared_resource_names.contains(&layout.resource_name) {
            return Err(SchemaTranslationError::UnusedLayout {
                schema_id: schema.schema_id.clone(),
                resource: layout.resource_name.clone(),
            });
        }
        if let Some(res) = schema
            .declared_resources
            .iter()
            .find(|r| r.name == layout.resource_name)
        {
            if res.element_type != layout.element_type {
                return Err(SchemaTranslationError::IncompatibleLayout {
                    dialect,
                    resource: layout.resource_name.clone(),
                    declared_type: layout.element_type.clone(),
                    resource_type: res.element_type.clone(),
                });
            }
        }
        layout.compute_extent_bytes(dialect)?;
    }

    Ok(bound_resources)
}
