//! Domain-neutral external schema, field, resource, layout, and translation contracts.
//!
//! Provides validation and mapping from versioned external schemas into semantic dialect
//! operations, layouts, and resource ABIs. Adheres to strict exhaustive closure: unknown
//! fields, duplicate fields, missing required fields, overflowing values, unused declared
//! resources/layouts, incomplete resource rosters, and incompatible identities are rejected
//! with actionable `Fix:` diagnostics before compilation.

use std::collections::BTreeSet;

use super::member::{FieldType, FieldValue};
use crate::ir::{BufferAccess, DataType};

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

/// Byte extent a strided layout spans, including the final element.
///
/// The one owner of the extent arithmetic. A static [`LayoutContract`] and a
/// decoded [`ExternalLayoutDeclaration`] describe the same geometry and must
/// agree on the capacity it needs, so both read this.
///
/// # Errors
///
/// Returns [`SchemaTranslationError::OverflowingLayout`] when the extent or the
/// byte capacity overflows `u64`.
fn layout_extent_bytes(
    dialect: &'static str,
    resource: &str,
    element_type: &DataType,
    shape: &[u64],
    strides: &[u64],
) -> Result<u64, SchemaTranslationError> {
    let overflow = || SchemaTranslationError::OverflowingLayout {
        dialect,
        resource: resource.to_string(),
    };
    let elem_size = element_type.min_bytes() as u64;
    let mut max_offset = 0_u64;
    for (&extent, &stride) in shape.iter().zip(strides.iter()) {
        if extent == 0 {
            continue;
        }
        let span = (extent - 1).checked_mul(stride).ok_or_else(overflow)?;
        max_offset = max_offset.checked_add(span).ok_or_else(overflow)?;
    }
    max_offset
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(elem_size))
        .ok_or_else(overflow)
}

impl LayoutContract {
    /// Compute minimum buffer capacity required by this layout in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaTranslationError::OverflowingLayout`] if extent math overflows `u64`.
    pub fn compute_capacity(&self, dialect: &'static str) -> Result<u64, SchemaTranslationError> {
        layout_extent_bytes(
            dialect,
            self.name,
            &self.element_type,
            self.shape,
            self.strides,
        )
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

/// One field an external schema node carries.
///
/// The declared member is the kind the external schema states the value has.
/// It is separate from the member a dialect field contract declares, and the
/// two are reconciled by [`super::validate_member_compatibility`], so a
/// consumer on another schema version cannot widen a field past what the
/// contract admits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalField {
    /// Field name.
    pub name: String,
    /// Member kind the external schema declares for this field.
    pub declared_member: FieldType,
    /// Raw textual value as the external schema carried it.
    pub raw_value: String,
}

impl ExternalField {
    /// Decode this field into a typed value of its declared member kind.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaTranslationError::OverflowingField`] when the raw text
    /// does not parse into the declared member without loss.
    pub fn decode(
        &self,
        dialect: &'static str,
        node_op: &str,
    ) -> Result<FieldValue, SchemaTranslationError> {
        self.declared_member.decode(&self.raw_value).map_err(|reason| {
            SchemaTranslationError::OverflowingField {
                dialect,
                node_op: node_op.to_string(),
                field: self.name.clone(),
                field_type: self.declared_member,
                value: self.raw_value.clone(),
                reason,
            }
        })
    }
}

/// External schema node representation used to validate neutral external models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalSchemaNode {
    /// Operation name in the external schema.
    pub op_name: String,
    /// Declared fields in schema order, so a duplicate is observable.
    pub fields: Vec<ExternalField>,
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
    pub fn compute_extent_bytes(
        &self,
        dialect: &'static str,
    ) -> Result<u64, SchemaTranslationError> {
        layout_extent_bytes(
            dialect,
            &self.resource_name,
            &self.element_type,
            &self.shape,
            &self.strides,
        )
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
        hasher.update(b"vyre.external_schema.v2\0");
        hasher.update(self.schema_id.as_bytes());
        hasher.update(&[0]);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&(self.nodes.len() as u64).to_le_bytes());
        for node in &self.nodes {
            hasher.update(node.op_name.as_bytes());
            hasher.update(&[0]);
            hasher.update(&(node.fields.len() as u64).to_le_bytes());
            for field in &node.fields {
                hasher.update(field.name.as_bytes());
                hasher.update(&[0]);
                hasher.update(&field.declared_member.wire_tag().to_le_bytes());
                hasher.update(field.raw_value.as_bytes());
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
    ///
    /// Each field is visited twice: once as the raw declaration the external
    /// schema carried, and once as the typed value its declared member decodes
    /// to, so a consumer reads a value of a known member kind rather than
    /// re-parsing text.
    ///
    /// # Errors
    ///
    /// Returns the visitor's error, including the
    /// [`SchemaTranslationError::OverflowingField`] raised when a raw value
    /// does not parse into the member the external schema declared for it.
    pub fn accept<V: ExternalSchemaVisitor>(
        &self,
        dialect: &'static str,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit_schema(&self.schema_id, self.version)?;
        for res in &self.declared_resources {
            visitor.visit_resource_declaration(res)?;
        }
        for layout in &self.declared_layouts {
            visitor.visit_layout_declaration(layout)?;
        }
        for node in &self.nodes {
            visitor.visit_node(node)?;
            for field in &node.fields {
                visitor.visit_field(&node.op_name, field)?;
                let value = field.decode(dialect, &node.op_name)?;
                visitor.visit_field_value(&node.op_name, &field.name, &value)?;
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
    type Error: From<SchemaTranslationError>;

    /// Visit schema header.
    fn visit_schema(&mut self, schema_id: &str, version: u32) -> Result<(), Self::Error>;

    /// Visit a schema operation node.
    fn visit_node(&mut self, node: &ExternalSchemaNode) -> Result<(), Self::Error>;

    /// Visit a declared field within a node, before it is decoded.
    fn visit_field(&mut self, node_op: &str, field: &ExternalField)
        -> Result<(), Self::Error>;

    /// Visit the typed value a declared field decodes to.
    fn visit_field_value(
        &mut self,
        node_op: &str,
        field_name: &str,
        value: &FieldValue,
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
    /// An external field declares a member kind the field contract refuses.
    #[error("Incompatible member `{external_member}` for field `{field}` in external schema node `{node_op}` for dialect `{dialect}`: the field contract declares `{contract_member}`. Fix: declare `{field}` as `{contract_member}` in the external schema, or widen the dialect field contract.")]
    IncompatibleFieldMember {
        /// Dialect identifier.
        dialect: &'static str,
        /// Node operation name.
        node_op: String,
        /// Field name.
        field: String,
        /// Member kind the dialect field contract declares.
        contract_member: FieldType,
        /// Member kind the external schema declares.
        external_member: FieldType,
    },
    /// A wire tag names no declared generic schema member.
    #[error("Unknown schema member wire tag `{tag}` for dialect `{dialect}`. Fix: rebuild the producer against a schema revision that declares tag `{tag}`, or re-encode the field with a member this revision declares.")]
    UnknownMemberTag {
        /// Dialect identifier.
        dialect: &'static str,
        /// Wire tag no declared member carries.
        tag: u16,
    },
}

/// Validate external schema node fields against a declared field contract.
///
/// # Errors
///
/// Returns [`SchemaTranslationError`] on unknown, duplicate, missing required,
/// member-incompatible, or overflowing fields.
pub fn validate_node_fields(
    dialect: &'static str,
    node_op: &str,
    fields: &[ExternalField],
    declared_fields: &[FieldContract],
) -> Result<(), SchemaTranslationError> {
    let mut seen_fields = BTreeSet::new();

    for field in fields {
        if !seen_fields.insert(field.name.as_str()) {
            return Err(SchemaTranslationError::DuplicateField {
                dialect,
                node_op: node_op.to_string(),
                field: field.name.clone(),
            });
        }
        let Some(contract) = declared_fields.iter().find(|f| f.name == field.name) else {
            return Err(SchemaTranslationError::UnknownField {
                dialect,
                node_op: node_op.to_string(),
                field: field.name.clone(),
            });
        };
        super::member::validate_member_compatibility(
            dialect,
            node_op,
            &field.name,
            contract.field_type,
            field.declared_member,
        )?;
        if let Err(reason) = contract.field_type.parse_and_validate(&field.raw_value) {
            return Err(SchemaTranslationError::OverflowingField {
                dialect,
                node_op: node_op.to_string(),
                field: field.name.clone(),
                field_type: contract.field_type,
                value: field.raw_value.clone(),
                reason,
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
