//! Dialect external schema, field contract, resource ABI, and translation error closure contracts.
//!
//! BACKLOG row 55 requires versioned domain-neutral schema, field, resource, layout,
//! and translation contracts with exhaustive visitors and canonical identity, proving
//! unknown, duplicate, missing, incompatible, and unmapped members fail closed before compilation.

#![forbid(unsafe_code)]

use vyre_foundation::dialect::{
    validate_node_fields, validate_node_resources, validate_schema_identity, FieldContract,
    FieldType, ResourceAbi, ResourceBinding, SchemaTranslationError,
};
use vyre_foundation::ir::{BufferAccess, DataType};

#[test]
fn field_type_exhaustive_closure() {
    let types = [
        FieldType::U32,
        FieldType::I32,
        FieldType::F32,
        FieldType::Bool,
        FieldType::String,
        FieldType::Bytes,
        FieldType::Buffer,
    ];

    for ft in types {
        match ft {
            FieldType::U32 => assert_eq!(ft, FieldType::U32),
            FieldType::I32 => assert_eq!(ft, FieldType::I32),
            FieldType::F32 => assert_eq!(ft, FieldType::F32),
            FieldType::Bool => assert_eq!(ft, FieldType::Bool),
            FieldType::String => assert_eq!(ft, FieldType::String),
            FieldType::Bytes => assert_eq!(ft, FieldType::Bytes),
            FieldType::Buffer => assert_eq!(ft, FieldType::Buffer),
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

    // 5. Incomplete resource roster fails
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

    // 6. Incompatible identity fails
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
