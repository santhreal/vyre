//! Dialect external schema, field contract, resource ABI, layout, and translation error closure contracts.
//!
//! The contract requires versioned domain-neutral schema, field, resource, layout,
//! and translation contracts with exhaustive visitors and canonical identity, proving
//! unknown, duplicate, missing, incompatible, overflowing, unused, and unmapped members
//! fail closed before compilation.
//!
//! A generic schema member is one [`FieldType`] an external schema may declare
//! for a field. [`FieldType::ALL`] is the roster, held to the enum declaration
//! at run time by the `variant-list-closure` gate, and the stage closures below
//! read that roster rather than a list of their own. Adding a member therefore
//! turns the wire, visitor, validator, and compatibility closures red until each
//! stage records a decision for it, and turns the public API snapshot red until
//! it is regenerated.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::dialect::{
    validate_external_schema, validate_member_compatibility, validate_node_fields,
    validate_node_resources, validate_schema_identity, ExternalField, ExternalLayoutDeclaration,
    ExternalResourceDeclaration, ExternalSchema, ExternalSchemaNode, ExternalSchemaVisitor,
    FieldContract, FieldType, FieldValue, LayoutContract, ResourceAbi, ResourceBinding,
    SchemaTranslationError,
};
use vyre_foundation::ir::{BufferAccess, DataType};

/// Dialect identifier every fixture in this file translates against.
const DIALECT: &str = "test::dialect";

/// Wire tag pinned for each generic schema member.
///
/// The tag is the cross-version identity of the member: a consumer built
/// against another schema revision resolves a field by this number. Moving a
/// tag renames every field already encoded with it, so the assignment is
/// recorded once here and a member without a row is refused.
const PINNED_MEMBER_WIRE_TAGS: &[(FieldType, u16)] = &[
    (FieldType::U32, 1),
    (FieldType::I32, 2),
    (FieldType::U64, 3),
    (FieldType::I64, 4),
    (FieldType::F32, 5),
    (FieldType::F64, 6),
    (FieldType::Bool, 7),
    (FieldType::String, 8),
    (FieldType::Bytes, 9),
    (FieldType::Buffer, 10),
];

/// Raw text the traversal closure declares for each member, and the typed value
/// the traversal must hand the visitor for it.
const MEMBER_TRAVERSAL_SAMPLES: &[(FieldType, &str, FieldValueShape)] = &[
    (FieldType::U32, "7", FieldValueShape::U32(7)),
    (FieldType::I32, "-7", FieldValueShape::I32(-7)),
    (FieldType::U64, "0xFF", FieldValueShape::U64(255)),
    (FieldType::I64, "-9", FieldValueShape::I64(-9)),
    (FieldType::F32, "1.5", FieldValueShape::F32(1.5)),
    (FieldType::F64, "2.5", FieldValueShape::F64(2.5)),
    (FieldType::Bool, "true", FieldValueShape::Bool(true)),
    (FieldType::String, "text", FieldValueShape::Str("text")),
    (FieldType::Bytes, "raw", FieldValueShape::Bytes(b"raw")),
    (FieldType::Buffer, "buf0", FieldValueShape::Buffer("buf0")),
];

/// The decoded value a traversal sample is required to produce.
#[derive(Clone, Copy, Debug)]
enum FieldValueShape {
    /// Unsigned 32-bit value.
    U32(u32),
    /// Signed 32-bit value.
    I32(i32),
    /// Unsigned 64-bit value.
    U64(u64),
    /// Signed 64-bit value.
    I64(i64),
    /// IEEE-754 32-bit value.
    F32(f32),
    /// IEEE-754 64-bit value.
    F64(f64),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string value.
    Str(&'static str),
    /// Opaque byte string value.
    Bytes(&'static [u8]),
    /// Buffer identifier value.
    Buffer(&'static str),
}

impl FieldValueShape {
    /// The decoded value this shape stands for.
    fn expected(self) -> FieldValue {
        match self {
            Self::U32(v) => FieldValue::U32(v),
            Self::I32(v) => FieldValue::I32(v),
            Self::U64(v) => FieldValue::U64(v),
            Self::I64(v) => FieldValue::I64(v),
            Self::F32(v) => FieldValue::F32(v),
            Self::F64(v) => FieldValue::F64(v),
            Self::Bool(v) => FieldValue::Bool(v),
            Self::Str(v) => FieldValue::String(v.to_string()),
            Self::Bytes(v) => FieldValue::Bytes(v.to_vec()),
            Self::Buffer(v) => FieldValue::Buffer(v.to_string()),
        }
    }
}

/// One value each member admits, and one it refuses.
///
/// A member whose raw text is unconstrained records `None` for the refusal and
/// is held to accepting an arbitrary payload instead.
const MEMBER_VALIDATION_SAMPLES: &[(FieldType, &str, Option<&str>)] = &[
    (FieldType::U32, "4294967295", Some("4294967296")),
    (FieldType::I32, "-2147483648", Some("2147483648")),
    (
        FieldType::U64,
        "18446744073709551615",
        Some("18446744073709551616"),
    ),
    (
        FieldType::I64,
        "-9223372036854775808",
        Some("9223372036854775808"),
    ),
    (FieldType::F32, "1.5", Some("not_a_number")),
    (FieldType::F64, "1.5e10", Some("inf")),
    (FieldType::Bool, "false", Some("yes")),
    (FieldType::String, "text", None),
    (FieldType::Bytes, "data", None),
    (FieldType::Buffer, "buf", None),
];

/// The external members each contract member admits across schema versions.
const MEMBER_COMPATIBILITY: &[(FieldType, &[FieldType])] = &[
    (FieldType::U32, &[FieldType::U32]),
    (FieldType::I32, &[FieldType::I32]),
    (FieldType::U64, &[FieldType::U64, FieldType::U32]),
    (FieldType::I64, &[FieldType::I64, FieldType::I32]),
    (FieldType::F32, &[FieldType::F32]),
    (FieldType::F64, &[FieldType::F64, FieldType::F32]),
    (FieldType::Bool, &[FieldType::Bool]),
    (FieldType::String, &[FieldType::String]),
    (FieldType::Bytes, &[FieldType::Bytes]),
    (FieldType::Buffer, &[FieldType::Buffer]),
];

/// Build a field declaring `member` with `raw` as its value.
fn field(name: &str, member: FieldType, raw: &str) -> ExternalField {
    ExternalField {
        name: name.to_string(),
        declared_member: member,
        raw_value: raw.to_string(),
    }
}

#[test]
fn field_type_exhaustive_closure() {
    // A field value arrives as text, so each integer width accepts a decimal or a
    // `0x`-prefixed hexadecimal literal and rejects one that overflows the width.
    for ft in FieldType::ALL {
        match ft {
            FieldType::U32 => {
                assert!(ft.parse_and_validate("42").is_ok());
                assert!(ft
                    .parse_and_validate("0xFFFF_FFFF".replace('_', "").as_str())
                    .is_ok());
                assert!(ft.parse_and_validate("4294967296").is_err());
            }
            FieldType::I32 => {
                assert!(ft.parse_and_validate("-42").is_ok());
                assert!(ft.parse_and_validate("2147483648").is_err());
            }
            FieldType::U64 => {
                assert!(ft.parse_and_validate("18446744073709551615").is_ok());
                assert!(ft.parse_and_validate("0x10").is_ok());
                assert!(ft.parse_and_validate("18446744073709551616").is_err());
            }
            FieldType::I64 => {
                assert!(ft.parse_and_validate("-9223372036854775808").is_ok());
                assert!(ft.parse_and_validate("9223372036854775808").is_err());
            }
            FieldType::F32 => {
                assert!(ft.parse_and_validate("1.5").is_ok());
                assert!(ft.parse_and_validate("inf").is_err());
            }
            FieldType::F64 => {
                assert!(ft.parse_and_validate("1.5e10").is_ok());
                assert!(ft.parse_and_validate("nan").is_err());
            }
            FieldType::Bool => {
                assert!(ft.parse_and_validate("false").is_ok());
                assert!(ft.parse_and_validate("1").is_err());
            }
            FieldType::String => assert!(ft.parse_and_validate("text").is_ok()),
            FieldType::Bytes => assert!(ft.parse_and_validate("data").is_ok()),
            FieldType::Buffer => assert!(ft.parse_and_validate("buf").is_ok()),
        }
    }
}

#[test]
fn schema_translation_error_exhaustive_closure() {
    let errors = [
        SchemaTranslationError::UnknownField {
            dialect: DIALECT,
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::DuplicateField {
            dialect: DIALECT,
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::MissingRequiredField {
            dialect: DIALECT,
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::IncompleteResourceRoster {
            dialect: DIALECT,
            node_op: "n".into(),
            expected_resource: "r",
            declared: vec!["r"],
        },
        SchemaTranslationError::IncompatibleIdentity {
            dialect: DIALECT,
            schema_id: "s".into(),
            required_dialect: "rd",
            found_version: 1,
            expected_version: 2,
        },
        SchemaTranslationError::UnmappedNode {
            dialect: DIALECT,
            node_op: "n".into(),
        },
        SchemaTranslationError::OverflowingField {
            dialect: DIALECT,
            node_op: "n".into(),
            field: "f".into(),
            field_type: FieldType::U32,
            value: "99999999999999".into(),
            reason: "overflow".into(),
        },
        SchemaTranslationError::OverflowingLayout {
            dialect: DIALECT,
            resource: "r".into(),
        },
        SchemaTranslationError::UnusedResource {
            schema_id: "s".into(),
            resource: "r".into(),
        },
        SchemaTranslationError::UnusedLayout {
            schema_id: "s".into(),
            resource: "r".into(),
        },
        SchemaTranslationError::IncompatibleLayout {
            dialect: DIALECT,
            resource: "r".into(),
            declared_type: DataType::F32,
            resource_type: DataType::U32,
        },
        SchemaTranslationError::IncompatibleFieldMember {
            dialect: DIALECT,
            node_op: "n".into(),
            field: "f".into(),
            contract_member: FieldType::U32,
            external_member: FieldType::U64,
        },
        SchemaTranslationError::UnknownMemberTag {
            dialect: DIALECT,
            tag: 4095,
        },
    ];

    for err in &errors {
        let msg = err.to_string();
        assert!(
            msg.contains("Fix:"),
            "every schema translation error must carry a Fix: action: {msg}"
        );
        match err {
            SchemaTranslationError::UnknownField { .. } => {}
            SchemaTranslationError::DuplicateField { .. } => {}
            SchemaTranslationError::MissingRequiredField { .. } => {}
            SchemaTranslationError::IncompleteResourceRoster { .. } => {}
            SchemaTranslationError::IncompatibleIdentity { .. } => {}
            SchemaTranslationError::UnmappedNode { .. } => {}
            SchemaTranslationError::OverflowingField { .. } => {}
            SchemaTranslationError::OverflowingLayout { .. } => {}
            SchemaTranslationError::UnusedResource { .. } => {}
            SchemaTranslationError::UnusedLayout { .. } => {}
            SchemaTranslationError::IncompatibleLayout { .. } => {}
            SchemaTranslationError::IncompatibleFieldMember { .. } => {}
            SchemaTranslationError::UnknownMemberTag { .. } => {}
        }
    }
}

/// Declared field contract the refusal fixtures translate against.
const DECLARED_FIELDS: &[FieldContract] = &[
    FieldContract {
        name: "stride",
        field_type: FieldType::U32,
        required: true,
    },
    FieldContract {
        name: "padding",
        field_type: FieldType::U32,
        required: false,
    },
];

#[test]
fn synthetic_fixtures_fail_closed_on_invalid_fields_and_resources() {
    assert!(validate_node_fields(
        DIALECT,
        "conv",
        &[
            field("stride", FieldType::U32, "2"),
            field("padding", FieldType::U32, "1"),
        ],
        DECLARED_FIELDS,
    )
    .is_ok());

    let unknown = validate_node_fields(
        DIALECT,
        "conv",
        &[
            field("stride", FieldType::U32, "2"),
            field("dilation", FieldType::U32, "1"),
        ],
        DECLARED_FIELDS,
    )
    .expect_err("unknown field must fail");
    assert_eq!(
        unknown.to_string(),
        "Unknown field `dilation` in external schema node `conv` for dialect `test::dialect`. \
         Fix: remove `dilation` or declare it in the dialect field contract."
    );

    let duplicate = validate_node_fields(
        DIALECT,
        "conv",
        &[
            field("stride", FieldType::U32, "2"),
            field("stride", FieldType::U32, "4"),
        ],
        DECLARED_FIELDS,
    )
    .expect_err("duplicate field must fail");
    assert_eq!(
        duplicate.to_string(),
        "Duplicate field `stride` in external schema node `conv` for dialect `test::dialect`. \
         Fix: deduplicate the field definition in the external schema payload."
    );

    let missing = validate_node_fields(
        DIALECT,
        "conv",
        &[field("padding", FieldType::U32, "1")],
        DECLARED_FIELDS,
    )
    .expect_err("missing required field must fail");
    assert_eq!(
        missing.to_string(),
        "Missing required field `stride` in external schema node `conv` for dialect \
         `test::dialect`. Fix: provide `stride` with valid data in the external schema payload."
    );

    let incompatible_member = validate_node_fields(
        DIALECT,
        "conv",
        &[field("stride", FieldType::U64, "2")],
        DECLARED_FIELDS,
    )
    .expect_err("a field declaring a wider member than its contract must fail");
    assert_eq!(
        incompatible_member.to_string(),
        "Incompatible member `u64` for field `stride` in external schema node `conv` for dialect \
         `test::dialect`: the field contract declares `u32`. Fix: declare `stride` as `u32` in \
         the external schema, or widen the dialect field contract."
    );

    let overflowing = validate_node_fields(
        DIALECT,
        "conv",
        &[field("stride", FieldType::U32, "999999999999999")],
        DECLARED_FIELDS,
    )
    .expect_err("overflowing u32 value must fail");
    assert_eq!(
        overflowing.to_string(),
        "Overflowing value `999999999999999` for field `stride` of type `U32` in node `conv` for \
         dialect `test::dialect`: invalid u32 value `999999999999999`: number too large to fit \
         in target type. Fix: ensure field values fit within declared type range."
    );

    static RES_BINDINGS: &[ResourceBinding] = &[
        ResourceBinding {
            name: "input",
            access: BufferAccess::ReadOnly,
            element_type: DataType::F32,
            alignment: 16,
            minimum_bytes: 64,
        },
        ResourceBinding {
            name: "weight",
            access: BufferAccess::ReadOnly,
            element_type: DataType::F32,
            alignment: 16,
            minimum_bytes: 64,
        },
    ];
    let abi = ResourceAbi {
        resources: RES_BINDINGS,
        layouts: &[],
    };

    assert!(
        validate_node_resources(DIALECT, "conv", &["input".into(), "weight".into()], &abi,).is_ok()
    );

    let roster = validate_node_resources(DIALECT, "conv", &["input".into()], &abi)
        .expect_err("incomplete resources must fail");
    assert_eq!(
        roster.to_string(),
        "Incomplete resource roster for external schema node `conv` in dialect `test::dialect`: \
         expected resource `weight` is not bound. Fix: bind all declared resources ([\"input\", \
         \"weight\"]) in the external resource ABI."
    );

    let identity = validate_schema_identity(DIALECT, "test::other_dialect", 1, DIALECT, 1)
        .expect_err("incompatible schema identity must fail");
    assert_eq!(
        identity.to_string(),
        "Incompatible schema identity `test::other_dialect` (version `1`) for dialect \
         `test::dialect` (requires `test::dialect`, schema version `1`). Fix: migrate external \
         schema to dialect `test::dialect` version `1`."
    );
}

#[test]
fn unused_declared_members_fail_closed() {
    let node = ExternalSchemaNode {
        op_name: "op_a".to_string(),
        fields: vec![],
        bound_resources: vec!["used_buf".to_string()],
    };
    let used = ExternalResourceDeclaration {
        name: "used_buf".to_string(),
        access: BufferAccess::ReadWrite,
        element_type: DataType::U32,
        alignment: 16,
        byte_capacity: 128,
    };

    let mut schema = ExternalSchema {
        schema_id: DIALECT.to_string(),
        version: 1,
        nodes: vec![node],
        declared_resources: vec![
            used.clone(),
            ExternalResourceDeclaration {
                name: "spare_buf".to_string(),
                ..used.clone()
            },
        ],
        declared_layouts: vec![],
    };

    let unused_resource = validate_external_schema(DIALECT, &schema, 1, |_| Ok(()))
        .expect_err("a declared resource no node binds must fail");
    assert_eq!(
        unused_resource.to_string(),
        "Unused declared resource `spare_buf` in external schema `test::dialect`. Fix: bind the \
         declared resource to a schema node or remove unused declaration."
    );

    schema.declared_resources = vec![used];
    schema.declared_layouts = vec![ExternalLayoutDeclaration {
        resource_name: "absent_buf".to_string(),
        element_type: DataType::U32,
        shape: vec![4],
        strides: vec![1],
        alignment: 16,
    }];
    let unused_layout = validate_external_schema(DIALECT, &schema, 1, |_| Ok(()))
        .expect_err("a layout for a resource the schema does not declare must fail");
    assert_eq!(
        unused_layout.to_string(),
        "Unused declared layout for resource `absent_buf` in external schema `test::dialect`. \
         Fix: bind the resource or remove unused layout declaration."
    );

    schema.declared_layouts = vec![ExternalLayoutDeclaration {
        resource_name: "used_buf".to_string(),
        element_type: DataType::F32,
        shape: vec![4],
        strides: vec![1],
        alignment: 16,
    }];
    let incompatible_layout = validate_external_schema(DIALECT, &schema, 1, |_| Ok(()))
        .expect_err("a layout element type the resource does not declare must fail");
    assert_eq!(
        incompatible_layout.to_string(),
        "Incompatible layout for resource `used_buf` in dialect `test::dialect`: element type \
         `F32` does not match resource element type `U32`. Fix: align layout element type with \
         resource declaration."
    );
}

/// Records every component the traversal hands it.
struct TestSchemaRecorder {
    /// Node operation names, in traversal order.
    visited_nodes: Vec<String>,
    /// Field name and declared member, in traversal order.
    visited_fields: Vec<(String, FieldType)>,
    /// Field name and decoded value, in traversal order.
    visited_values: Vec<(String, FieldValue)>,
    /// Bound resource names, in traversal order.
    visited_resources: Vec<String>,
    /// Declared resource names, in traversal order.
    visited_declarations: Vec<String>,
    /// Declared layout resource names, in traversal order.
    visited_layouts: Vec<String>,
}

impl TestSchemaRecorder {
    /// An empty recorder.
    fn new() -> Self {
        Self {
            visited_nodes: Vec::new(),
            visited_fields: Vec::new(),
            visited_values: Vec::new(),
            visited_resources: Vec::new(),
            visited_declarations: Vec::new(),
            visited_layouts: Vec::new(),
        }
    }
}

impl ExternalSchemaVisitor for TestSchemaRecorder {
    type Error = SchemaTranslationError;

    fn visit_schema(&mut self, _schema_id: &str, _version: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_node(&mut self, node: &ExternalSchemaNode) -> Result<(), Self::Error> {
        self.visited_nodes.push(node.op_name.clone());
        Ok(())
    }

    fn visit_field(&mut self, _node_op: &str, field: &ExternalField) -> Result<(), Self::Error> {
        self.visited_fields
            .push((field.name.clone(), field.declared_member));
        Ok(())
    }

    fn visit_field_value(
        &mut self,
        _node_op: &str,
        field_name: &str,
        value: &FieldValue,
    ) -> Result<(), Self::Error> {
        self.visited_values
            .push((field_name.to_string(), value.clone()));
        Ok(())
    }

    fn visit_resource_binding(
        &mut self,
        _node_op: &str,
        resource_name: &str,
    ) -> Result<(), Self::Error> {
        self.visited_resources.push(resource_name.to_string());
        Ok(())
    }

    fn visit_resource_declaration(
        &mut self,
        resource: &ExternalResourceDeclaration,
    ) -> Result<(), Self::Error> {
        self.visited_declarations.push(resource.name.clone());
        Ok(())
    }

    fn visit_layout_declaration(
        &mut self,
        layout: &ExternalLayoutDeclaration,
    ) -> Result<(), Self::Error> {
        self.visited_layouts.push(layout.resource_name.clone());
        Ok(())
    }
}

#[test]
fn external_schema_canonical_identity_and_visitor_closure() {
    let schema = ExternalSchema {
        schema_id: "vyre-test::generic".to_string(),
        version: 1,
        nodes: vec![
            ExternalSchemaNode {
                op_name: "vyre-test::generic::op_a".to_string(),
                fields: vec![field("rate", FieldType::U32, "10")],
                bound_resources: vec!["buf_in".to_string(), "buf_out".to_string()],
            },
            ExternalSchemaNode {
                op_name: "vyre-test::generic::op_b".to_string(),
                fields: vec![],
                bound_resources: vec!["buf_out".to_string()],
            },
        ],
        declared_resources: vec![
            ExternalResourceDeclaration {
                name: "buf_in".to_string(),
                access: BufferAccess::ReadOnly,
                element_type: DataType::U32,
                alignment: 16,
                byte_capacity: 128,
            },
            ExternalResourceDeclaration {
                name: "buf_out".to_string(),
                access: BufferAccess::WriteOnly,
                element_type: DataType::U32,
                alignment: 16,
                byte_capacity: 128,
            },
        ],
        declared_layouts: vec![
            ExternalLayoutDeclaration {
                resource_name: "buf_in".to_string(),
                element_type: DataType::U32,
                shape: vec![32],
                strides: vec![1],
                alignment: 16,
            },
            ExternalLayoutDeclaration {
                resource_name: "buf_out".to_string(),
                element_type: DataType::U32,
                shape: vec![32],
                strides: vec![1],
                alignment: 16,
            },
        ],
    };

    let mut recorder = TestSchemaRecorder::new();
    schema
        .accept("vyre-test::generic", &mut recorder)
        .expect("schema visitor traversal must succeed");

    assert_eq!(recorder.visited_nodes.len(), 2);
    assert_eq!(
        recorder.visited_fields,
        vec![("rate".to_string(), FieldType::U32)]
    );
    assert_eq!(
        recorder.visited_values,
        vec![("rate".to_string(), FieldValue::U32(10))]
    );
    assert_eq!(recorder.visited_resources.len(), 3);
    assert_eq!(recorder.visited_declarations, vec!["buf_in", "buf_out"]);
    assert_eq!(recorder.visited_layouts, vec!["buf_in", "buf_out"]);

    let id1 = schema.canonical_identity();
    let id2 = schema.canonical_identity();
    assert_eq!(id1, id2);
    assert_ne!(id1, [0; 32]);

    // The declared member is part of the identity, so the same text under a
    // different member is a different schema.
    let mut widened = schema.clone();
    widened.nodes[0].fields[0].declared_member = FieldType::U64;
    assert_ne!(widened.canonical_identity(), id1);

    let bound = validate_external_schema("vyre-test::generic", &schema, 1, |_node| Ok(()))
        .expect("validation must succeed on consistent schema");
    let mut expected_bound = BTreeSet::new();
    expected_bound.insert("buf_in".to_string());
    expected_bound.insert("buf_out".to_string());
    assert_eq!(bound, expected_bound);
}

#[test]
fn traversal_refuses_a_value_its_declared_member_cannot_carry() {
    let schema = ExternalSchema {
        schema_id: DIALECT.to_string(),
        version: 1,
        nodes: vec![ExternalSchemaNode {
            op_name: "op_a".to_string(),
            fields: vec![field("rate", FieldType::U32, "4294967296")],
            bound_resources: vec![],
        }],
        declared_resources: vec![],
        declared_layouts: vec![],
    };
    let mut recorder = TestSchemaRecorder::new();
    let err = schema
        .accept(DIALECT, &mut recorder)
        .expect_err("a raw value outside its declared member must stop the traversal");
    assert_eq!(
        err.to_string(),
        "Overflowing value `4294967296` for field `rate` of type `U32` in node `op_a` for dialect \
         `test::dialect`: invalid u32 value `4294967296`: number too large to fit in target type. \
         Fix: ensure field values fit within declared type range."
    );
    assert!(recorder.visited_values.is_empty());
}

#[test]
fn layout_contract_capacity_and_overflow_closure() {
    static SHAPE: &[u64] = &[8, 16];
    static STRIDES: &[u64] = &[16, 1];
    let layout = LayoutContract {
        name: "tensor_2d",
        element_type: DataType::F32,
        shape: SHAPE,
        strides: STRIDES,
        alignment: 16,
        contiguous: true,
    };

    let cap = layout
        .compute_capacity(DIALECT)
        .expect("valid layout capacity computation");
    assert_eq!(cap, 8 * 16 * 4);

    // The static contract and the decoded declaration describe one geometry and
    // must agree on the capacity it needs.
    let declaration = ExternalLayoutDeclaration {
        resource_name: "tensor_2d".to_string(),
        element_type: DataType::F32,
        shape: SHAPE.to_vec(),
        strides: STRIDES.to_vec(),
        alignment: 16,
    };
    assert_eq!(
        declaration
            .compute_extent_bytes(DIALECT)
            .expect("valid declaration capacity computation"),
        cap
    );

    static OVERFLOW_SHAPE: &[u64] = &[u64::MAX, u64::MAX];
    static OVERFLOW_STRIDES: &[u64] = &[u64::MAX, 1];
    let overflow_layout = LayoutContract {
        name: "overflow_tensor",
        element_type: DataType::F32,
        shape: OVERFLOW_SHAPE,
        strides: OVERFLOW_STRIDES,
        alignment: 16,
        contiguous: false,
    };
    let err = overflow_layout
        .compute_capacity(DIALECT)
        .expect_err("overflowing layout must fail");
    assert_eq!(
        err.to_string(),
        "Overflowing layout for resource `overflow_tensor` in dialect `test::dialect`: extent or \
         capacity computation overflows u64. Fix: check dimensions and stride parameters."
    );
}

#[test]
fn schema_member_wire_closure() {
    let pinned: BTreeMap<FieldType, u16> = PINNED_MEMBER_WIRE_TAGS.iter().copied().collect();
    assert_eq!(
        pinned.len(),
        PINNED_MEMBER_WIRE_TAGS.len(),
        "wire closure: a member is pinned to two tags"
    );

    let mut owners: BTreeMap<u16, FieldType> = BTreeMap::new();
    for member in FieldType::ALL {
        let Some(&tag) = pinned.get(&member) else {
            panic!(
                "wire closure: member `{member}` has no pinned wire tag; \
                 assign it a tag no member has ever carried"
            );
        };
        assert_eq!(
            member.wire_tag(),
            tag,
            "wire closure: member `{member}` moved off its pinned tag"
        );
        assert_eq!(
            FieldType::from_wire_tag(DIALECT, tag).expect("a pinned tag must resolve"),
            member,
            "wire closure: tag {tag} does not resolve back to `{member}`"
        );
        if let Some(other) = owners.insert(tag, member) {
            panic!("wire closure: members `{other}` and `{member}` share tag {tag}");
        }
    }

    let unknown = u16::MAX;
    assert!(
        !owners.contains_key(&unknown),
        "the unknown-tag fixture must name a tag no member carries"
    );
    let err = FieldType::from_wire_tag(DIALECT, unknown)
        .expect_err("a tag no declared member carries must fail");
    assert_eq!(
        err.to_string(),
        "Unknown schema member wire tag `65535` for dialect `test::dialect`. Fix: rebuild the \
         producer against a schema revision that declares tag `65535`, or re-encode the field \
         with a member this revision declares."
    );
}

#[test]
fn schema_member_visitor_closure() {
    let mut fields = Vec::new();
    for member in FieldType::ALL {
        let Some((_, raw, _)) = MEMBER_TRAVERSAL_SAMPLES
            .iter()
            .find(|(declared, _, _)| *declared == member)
        else {
            panic!(
                "visitor closure: member `{member}` has no recorded traversal sample; \
                 record the raw text it is declared with and the value it must decode to"
            );
        };
        fields.push(field(member.as_str(), member, raw));
    }

    let schema = ExternalSchema {
        schema_id: DIALECT.to_string(),
        version: 1,
        nodes: vec![ExternalSchemaNode {
            op_name: "op_all_members".to_string(),
            fields,
            bound_resources: vec![],
        }],
        declared_resources: vec![],
        declared_layouts: vec![],
    };

    let mut recorder = TestSchemaRecorder::new();
    schema
        .accept(DIALECT, &mut recorder)
        .expect("every recorded traversal sample must decode");

    for member in FieldType::ALL {
        let (_, _, shape) = MEMBER_TRAVERSAL_SAMPLES
            .iter()
            .find(|(declared, _, _)| *declared == member)
            .expect("the sample was found above");
        let name = member.as_str().to_string();
        assert!(
            recorder.visited_fields.contains(&(name.clone(), member)),
            "visitor closure: the traversal did not visit the declaration of `{member}`"
        );
        let (_, value) = recorder
            .visited_values
            .iter()
            .find(|(field_name, _)| *field_name == name)
            .unwrap_or_else(|| {
                panic!("visitor closure: the traversal decoded no value for `{member}`")
            });
        assert_eq!(
            value.field_type(),
            member,
            "visitor closure: `{member}` decoded to a value of another member kind"
        );
        assert_eq!(
            *value,
            shape.expected(),
            "visitor closure: `{member}` decoded to an unrecorded value"
        );
    }
    assert_eq!(recorder.visited_values.len(), FieldType::ALL.len());
}

#[test]
fn schema_member_validator_closure() {
    for member in FieldType::ALL {
        let Some((_, accepted, refused)) = MEMBER_VALIDATION_SAMPLES
            .iter()
            .find(|(declared, _, _)| *declared == member)
        else {
            panic!(
                "validator closure: member `{member}` has no recorded accept/reject sample; \
                 record one value it admits and one it refuses, or `None` when its text is \
                 unconstrained"
            );
        };

        let contract = [FieldContract {
            name: "member_under_test",
            field_type: member,
            required: true,
        }];
        validate_node_fields(
            DIALECT,
            "op_a",
            &[field("member_under_test", member, accepted)],
            &contract,
        )
        .unwrap_or_else(|err| {
            panic!("validator closure: `{member}` refused its accepted sample: {err}")
        });

        match refused {
            Some(refused) => {
                let err = validate_node_fields(
                    DIALECT,
                    "op_a",
                    &[field("member_under_test", member, refused)],
                    &contract,
                )
                .unwrap_err();
                assert!(
                    matches!(err, SchemaTranslationError::OverflowingField { .. }),
                    "validator closure: `{member}` refused `{refused}` with the wrong error: {err}"
                );
                assert!(err.to_string().contains("Fix:"));
            }
            None => {
                // A member whose text is unconstrained states that, and is held
                // to it: an arbitrary payload must translate.
                validate_node_fields(
                    DIALECT,
                    "op_a",
                    &[field("member_under_test", member, "\u{1}arbitrary\u{7f}")],
                    &contract,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "validator closure: `{member}` records no refusal yet refused a \
                         payload: {err}"
                    )
                });
            }
        }
    }
}

#[test]
fn schema_member_compatibility_closure() {
    for contract_member in FieldType::ALL {
        let Some((_, admitted)) = MEMBER_COMPATIBILITY
            .iter()
            .find(|(declared, _)| *declared == contract_member)
        else {
            panic!(
                "compatibility closure: member `{contract_member}` has no recorded compatibility \
                 row; record every external member a contract declaring it admits"
            );
        };
        assert!(
            admitted.contains(&contract_member),
            "compatibility closure: `{contract_member}` must admit itself"
        );

        for external_member in FieldType::ALL {
            let recorded = admitted.contains(&external_member);
            assert_eq!(
                contract_member.accepts(external_member),
                recorded,
                "compatibility closure: `{contract_member}` accepting `{external_member}` \
                 disagrees with the recorded row"
            );
            let outcome = validate_member_compatibility(
                DIALECT,
                "op_a",
                "member_under_test",
                contract_member,
                external_member,
            );
            assert_eq!(
                outcome.is_ok(),
                recorded,
                "compatibility closure: translation of `{external_member}` into \
                 `{contract_member}` disagrees with the recorded row"
            );
            if let Err(err) = outcome {
                assert!(matches!(
                    err,
                    SchemaTranslationError::IncompatibleFieldMember { .. }
                ));
            }
        }
    }
}
