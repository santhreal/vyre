//! Dialect external schema, field contract, resource ABI, layout, and translation error closure contracts.
//!
//! The contract requires versioned domain-neutral schema, field, resource, layout,
//! and translation contracts with exhaustive visitors and canonical identity, proving
//! unknown, duplicate, missing, incompatible, overflowing, unused, and unmapped members
//! fail closed before compilation.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_foundation::dialect::{
    validate_external_schema, validate_node_fields, validate_node_resources,
    validate_schema_identity, ExternalLayoutDeclaration, ExternalResourceDeclaration,
    ExternalSchema, ExternalSchemaNode, ExternalSchemaVisitor, FieldContract, FieldType,
    LayoutContract, ResourceAbi, ResourceBinding, SchemaTranslationError,
};
use vyre_foundation::ir::{BufferAccess, DataType};

#[test]
fn field_type_exhaustive_closure() {
    let types = [
        FieldType::U32,
        FieldType::I32,
        FieldType::U64,
        FieldType::I64,
        FieldType::F32,
        FieldType::F64,
        FieldType::Bool,
        FieldType::String,
        FieldType::Bytes,
        FieldType::Buffer,
    ];

    // A field value arrives as text, so each integer width accepts a decimal or a
    // `0x`-prefixed hexadecimal literal and rejects one that overflows the width.
    for ft in types {
        match ft {
            FieldType::U32 => {
                assert!(ft.parse_and_validate("123").is_ok());
                assert!(ft.parse_and_validate("0xFFFFFFFF").is_ok());
                assert!(ft.parse_and_validate("0Xff").is_ok());
                assert!(ft.parse_and_validate("0x100000000").is_err());
                assert!(ft.parse_and_validate("4294967296").is_err());
                assert!(ft.parse_and_validate("-1").is_err());
                assert!(ft.parse_and_validate("0xzz").is_err());
            }
            FieldType::I32 => {
                assert!(ft.parse_and_validate("-123").is_ok());
                assert!(ft.parse_and_validate("0x7FFFFFFF").is_ok());
                assert!(ft.parse_and_validate("0x80000000").is_err());
                assert!(ft.parse_and_validate("2147483648").is_err());
            }
            FieldType::U64 => {
                assert!(ft.parse_and_validate("1234567890123").is_ok());
                assert!(ft.parse_and_validate("0xFFFFFFFFFFFFFFFF").is_ok());
                assert!(ft.parse_and_validate("0x10000000000000000").is_err());
                assert!(ft.parse_and_validate("18446744073709551616").is_err());
            }
            FieldType::I64 => {
                assert!(ft.parse_and_validate("-1234567890123").is_ok());
                assert!(ft.parse_and_validate("0x7FFFFFFFFFFFFFFF").is_ok());
                assert!(ft.parse_and_validate("0x8000000000000000").is_err());
                assert!(ft.parse_and_validate("9223372036854775808").is_err());
            }
            FieldType::F32 => assert!(ft.parse_and_validate("3.14").is_ok()),
            FieldType::F64 => assert!(ft.parse_and_validate("3.1415926535").is_ok()),
            FieldType::Bool => assert!(ft.parse_and_validate("true").is_ok()),
            FieldType::String => assert!(ft.parse_and_validate("hello").is_ok()),
            FieldType::Bytes => assert!(ft.parse_and_validate("opaque").is_ok()),
            FieldType::Buffer => assert!(ft.parse_and_validate("buf0").is_ok()),
        }
    }
}

#[test]
fn schema_translation_error_exhaustive_closure() {
    let errors = [
        SchemaTranslationError::UnknownField {
            dialect: "d",
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::DuplicateField {
            dialect: "d",
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::MissingRequiredField {
            dialect: "d",
            node_op: "n".into(),
            field: "f".into(),
        },
        SchemaTranslationError::IncompleteResourceRoster {
            dialect: "d",
            node_op: "n".into(),
            expected_resource: "r",
            declared: vec!["r"],
        },
        SchemaTranslationError::IncompatibleIdentity {
            dialect: "d",
            schema_id: "s".into(),
            required_dialect: "rd",
            found_version: 1,
            expected_version: 2,
        },
        SchemaTranslationError::UnmappedNode {
            dialect: "d",
            node_op: "n".into(),
        },
        SchemaTranslationError::OverflowingField {
            dialect: "d",
            node_op: "n".into(),
            field: "f".into(),
            field_type: FieldType::U32,
            value: "99999999999999".into(),
            reason: "overflow".into(),
        },
        SchemaTranslationError::OverflowingLayout {
            dialect: "d",
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
            dialect: "d",
            resource: "r".into(),
            declared_type: DataType::F32,
            resource_type: DataType::U32,
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
        }
    }
}

#[test]
fn synthetic_fixtures_fail_closed_on_invalid_fields_and_resources() {
    let declared_fields = [
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

    // 1. Valid fields
    assert!(validate_node_fields(
        "test::dialect",
        "conv",
        &[
            ("stride".into(), "2".into()),
            ("padding".into(), "1".into())
        ],
        &declared_fields,
    )
    .is_ok());

    // 2. Unknown field fails
    let err_unknown = validate_node_fields(
        "test::dialect",
        "conv",
        &[
            ("stride".into(), "2".into()),
            ("dilation".into(), "1".into()),
        ],
        &declared_fields,
    )
    .expect_err("unknown field must fail");
    assert!(matches!(
        err_unknown,
        SchemaTranslationError::UnknownField { .. }
    ));

    // 3. Duplicate field fails
    let err_dup = validate_node_fields(
        "test::dialect",
        "conv",
        &[("stride".into(), "2".into()), ("stride".into(), "4".into())],
        &declared_fields,
    )
    .expect_err("duplicate field must fail");
    assert!(matches!(
        err_dup,
        SchemaTranslationError::DuplicateField { .. }
    ));

    // 4. Missing required field fails
    let err_missing = validate_node_fields(
        "test::dialect",
        "conv",
        &[("padding".into(), "1".into())],
        &declared_fields,
    )
    .expect_err("missing required field must fail");
    assert!(matches!(
        err_missing,
        SchemaTranslationError::MissingRequiredField { .. }
    ));

    // 5. Overflowing field value fails
    let err_overflow = validate_node_fields(
        "test::dialect",
        "conv",
        &[("stride".into(), "999999999999999".into())],
        &declared_fields,
    )
    .expect_err("overflowing u32 value must fail");
    assert!(matches!(
        err_overflow,
        SchemaTranslationError::OverflowingField { .. }
    ));

    // 6. Incomplete resource roster fails
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

    assert!(validate_node_resources(
        "test::dialect",
        "conv",
        &["input".into(), "weight".into()],
        &abi,
    )
    .is_ok());

    let err_res = validate_node_resources("test::dialect", "conv", &["input".into()], &abi)
        .expect_err("incomplete resources must fail");
    assert!(matches!(
        err_res,
        SchemaTranslationError::IncompleteResourceRoster { .. }
    ));

    // 7. Incompatible identity fails
    let err_id = validate_schema_identity(
        "test::dialect",
        "test::other_dialect",
        1,
        "test::dialect",
        1,
    )
    .expect_err("incompatible schema identity must fail");
    assert!(matches!(
        err_id,
        SchemaTranslationError::IncompatibleIdentity { .. }
    ));
}

struct TestSchemaRecorder {
    visited_nodes: Vec<String>,
    visited_fields: Vec<String>,
    visited_resources: Vec<String>,
    visited_declarations: Vec<String>,
    visited_layouts: Vec<String>,
}

impl ExternalSchemaVisitor for TestSchemaRecorder {
    type Error = ();

    fn visit_schema(&mut self, _schema_id: &str, _version: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_node(&mut self, node: &ExternalSchemaNode) -> Result<(), Self::Error> {
        self.visited_nodes.push(node.op_name.clone());
        Ok(())
    }

    fn visit_field(
        &mut self,
        _node_op: &str,
        field_name: &str,
        _field_value: &str,
    ) -> Result<(), Self::Error> {
        self.visited_fields.push(field_name.to_string());
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
                raw_fields: vec![("rate".to_string(), "10".to_string())],
                bound_resources: vec!["buf_in".to_string(), "buf_out".to_string()],
            },
            ExternalSchemaNode {
                op_name: "vyre-test::generic::op_b".to_string(),
                raw_fields: vec![],
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

    let mut recorder = TestSchemaRecorder {
        visited_nodes: Vec::new(),
        visited_fields: Vec::new(),
        visited_resources: Vec::new(),
        visited_declarations: Vec::new(),
        visited_layouts: Vec::new(),
    };

    schema
        .accept(&mut recorder)
        .expect("schema visitor traversal must succeed");

    assert_eq!(recorder.visited_nodes.len(), 2);
    assert_eq!(recorder.visited_fields, vec!["rate"]);
    assert_eq!(recorder.visited_resources.len(), 3);
    assert_eq!(recorder.visited_declarations, vec!["buf_in", "buf_out"]);
    assert_eq!(recorder.visited_layouts, vec!["buf_in", "buf_out"]);

    let id1 = schema.canonical_identity();
    let id2 = schema.canonical_identity();
    assert_eq!(id1, id2);
    assert_ne!(id1, [0; 32]);

    let bound = validate_external_schema("vyre-test::generic", &schema, 1, |_node| Ok(()))
        .expect("validation must succeed on consistent schema");
    let mut expected_bound = BTreeSet::new();
    expected_bound.insert("buf_in".to_string());
    expected_bound.insert("buf_out".to_string());
    assert_eq!(bound, expected_bound);
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
        .compute_capacity("test::dialect")
        .expect("valid layout capacity computation");
    assert_eq!(cap, 8 * 16 * 4);

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
        .compute_capacity("test::dialect")
        .expect_err("overflowing layout must fail");
    assert!(matches!(
        err,
        SchemaTranslationError::OverflowingLayout { .. }
    ));
}
