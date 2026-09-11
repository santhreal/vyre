//! Unit tests for declarative and versioned dialects.
//!
//! `define_dialect!` emits the public surface a dialect crate exports. Invoked
//! in a private test module that surface is unreachable, which the lint reports
//! once per generated item.
#![expect(
    unreachable_pub,
    reason = "the define_dialect! invocation below emits that surface, so removing the invocation from this file has to remove this line with it"
)]

use crate::dialect::descriptor::DialectRegistry;
use crate::dialect::member::FieldType;
use crate::dialect::schema::{
    validate_external_schema, validate_schema_identity, ExternalField, ExternalLayoutDeclaration,
    ExternalResourceDeclaration, ExternalSchema, ExternalSchemaNode, FieldContract, ResourceAbi,
    ResourceBinding, SchemaTranslationError,
};
use crate::dialect::traits::Dialect;
use crate::dialect::version::{validate_dialect_version, DialectVersionError};
use crate::dialect_lookup::{Signature, TypedParam};
use crate::ir::{BufferAccess, DataType, Expr};
use crate::operation::OperationTier;

const UNIT_SIG: Signature = Signature {
    inputs: &[TypedParam {
        name: "in_val",
        ty: "u32",
    }],
    outputs: &[TypedParam {
        name: "out_val",
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

static ALPHA_FIELDS: &[FieldContract] = &[
    FieldContract {
        name: "scale",
        field_type: FieldType::U32,
        required: true,
    },
    FieldContract {
        name: "tag",
        field_type: FieldType::String,
        required: false,
    },
];

static ALPHA_RESOURCES: &[ResourceBinding] = &[ResourceBinding {
    name: "scratch_buf",
    access: BufferAccess::ReadWrite,
    element_type: DataType::U32,
    alignment: 16,
    minimum_bytes: 64,
}];

static ALPHA_ABI: ResourceAbi = ResourceAbi {
    resources: ALPHA_RESOURCES,
    layouts: &[],
};

crate::define_dialect! {
    /// Unit test dialect.
    dialect: "vyre-unit::dialect",
    name: unit_dialect,
    visitor: UnitDialectVisitor,
    version: 3,
    min_supported_version: 2,
    tier: OperationTier::Library,
    category: "unit_test",
    summary: "Unit test dialect definition.",

    operations: [
        {
            op: Alpha,
            discriminant: 0,
            name: "alpha",
            id: "vyre-unit::dialect::alpha",
            version: 2,
            summary: "Alpha operation.",
            signature: UNIT_SIG,
            is_composable: true,
            decision: Some(crate::operation::AbsenceDecision::Uncharacterized),
            call_builder: call_alpha,
            fields: ALPHA_FIELDS,
            resource_abi: ALPHA_ABI,
        },
        {
            op: Beta,
            discriminant: 1,
            name: "beta",
            id: "vyre-unit::dialect::beta",
            version: 3,
            summary: "Beta operation.",
            signature: UNIT_SIG,
            is_composable: true,
            decision: Some(crate::operation::AbsenceDecision::Uncharacterized),
            call_builder: call_beta,
        },
    ]
}

#[test]
fn test_unit_dialect_metadata() {
    use unit_dialect::*;

    assert_eq!(SCHEMA_VERSION, 3);
    assert_eq!(MIN_SUPPORTED_VERSION, 2);
    assert_eq!(DIALECT_ID, "vyre-unit::dialect");

    let desc = DialectMarker::descriptor();
    assert_eq!(desc.id, "vyre-unit::dialect");
    assert_eq!(desc.version, 3);
    assert_eq!(desc.min_supported_version, 2);
    assert_eq!(desc.operations.len(), 2);
    assert_eq!(desc.operations[0].name, "alpha");
    assert_eq!(desc.operations[1].name, "beta");

    let reg = DialectRegistry::get("vyre-unit::dialect")
        .expect("Fix: unit dialect must be present in registry");
    assert_eq!(reg.id, "vyre-unit::dialect");
}

#[test]
fn test_unit_dialect_ops() {
    use unit_dialect::*;

    assert_eq!(Op::Alpha.op_id(), "vyre-unit::dialect::alpha");
    assert_eq!(Op::Alpha.op_name(), "alpha");
    assert_eq!(Op::Alpha.introduced_version(), 2);
    assert!(Op::Alpha.is_composable());

    assert_eq!(Op::Beta.op_id(), "vyre-unit::dialect::beta");
    assert_eq!(Op::Beta.op_name(), "beta");
    assert_eq!(Op::Beta.introduced_version(), 3);
    assert!(Op::Beta.is_composable());

    assert_eq!(match_op_id("vyre-unit::dialect::alpha"), Some(Op::Alpha));
    assert_eq!(match_op_id("vyre-unit::dialect::beta"), Some(Op::Beta));
    assert_eq!(match_op_id("vyre-unit::dialect::gamma"), None);
}

/// The taxonomy strings, the identifier roster and the call surface a dialect
/// generates are part of what a consumer reads, so each is asserted here.
///
/// WHY: they had no reader. A generated item nothing reads is one nothing holds
/// to a shape either, and a declaration order or an identifier could change
/// without a single case going red.
#[test]
fn test_unit_dialect_generated_surface() {
    use unit_dialect::{call_alpha, call_beta, match_call, ALL_OP_IDS, CATEGORY, SUMMARY};

    assert_eq!(CATEGORY, "unit_test");
    assert_eq!(SUMMARY, "Unit test dialect definition.");
    assert_eq!(
        ALL_OP_IDS,
        ["vyre-unit::dialect::alpha", "vyre-unit::dialect::beta"].as_slice(),
        "the roster is declaration order, which the wire discriminants follow"
    );

    let alpha = call_alpha(vec![Expr::u32(7)]);
    let (matched, args) = match_call(&alpha).expect("a call this dialect declares must match");
    assert_eq!(matched, unit_dialect::Op::Alpha);
    assert_eq!(args, [Expr::u32(7)].as_slice());

    let beta = call_beta(Vec::new());
    let (matched, args) = match_call(&beta).expect("the second builder must match too");
    assert_eq!(matched, unit_dialect::Op::Beta);
    assert!(args.is_empty());

    assert!(
        match_call(&Expr::u32(0)).is_none(),
        "an expression that is not a call matches nothing"
    );
    assert!(
        match_call(&Expr::call("vyre-unit::dialect::gamma", Vec::new())).is_none(),
        "a call this dialect does not declare matches nothing"
    );
}

#[test]
fn test_unit_dialect_version_validation() {
    let desc = unit_dialect::DESCRIPTOR;

    // Supported versions: 2, 3
    assert!(validate_dialect_version(&desc, 2).is_ok());
    assert!(validate_dialect_version(&desc, 3).is_ok());

    // Stale version: 1 < min_supported_version 2
    let stale = validate_dialect_version(&desc, 1).expect_err("version 1 is stale");
    assert!(matches!(stale, DialectVersionError::StaleVersion { .. }));
    assert!(stale.to_string().contains("Fix:"));

    // Unsupported future version: 4 > version 3
    let future = validate_dialect_version(&desc, 4).expect_err("version 4 is unsupported");
    assert!(matches!(
        future,
        DialectVersionError::UnsupportedVersion { .. }
    ));
    assert!(future.to_string().contains("Fix:"));
}

struct TestVisitor {
    visited_alpha: bool,
    visited_beta: bool,
}

impl unit_dialect::UnitDialectVisitor for TestVisitor {
    type Output = ();

    fn Alpha(&mut self, _args: &[Expr]) {
        self.visited_alpha = true;
    }

    fn Beta(&mut self, _args: &[Expr]) {
        self.visited_beta = true;
    }
}

#[test]
fn test_unit_dialect_visitor() {
    use unit_dialect::*;

    let mut visitor = TestVisitor {
        visited_alpha: false,
        visited_beta: false,
    };

    let args = [Expr::u32(100)];
    dispatch_visitor(Op::Alpha, &args, &mut visitor);
    assert!(visitor.visited_alpha);
    assert!(!visitor.visited_beta);

    dispatch_visitor(Op::Beta, &args, &mut visitor);
    assert!(visitor.visited_beta);
}

#[test]
fn test_unit_dialect_codec() {
    use unit_dialect::*;

    let mut buf = Vec::new();
    encode_op(Op::Alpha, &mut buf).expect("Fix: encode must succeed");
    encode_op(Op::Beta, &mut buf).expect("Fix: encode must succeed");

    let mut cursor = std::io::Cursor::new(&buf);
    let decoded_alpha = decode_op(&mut cursor)
        .expect("Fix: decode must succeed")
        .expect("Fix: valid discriminant");
    let decoded_beta = decode_op(&mut cursor)
        .expect("Fix: decode must succeed")
        .expect("Fix: valid discriminant");

    assert_eq!(decoded_alpha, Op::Alpha);
    assert_eq!(decoded_beta, Op::Beta);
}

#[test]
fn test_unit_dialect_validation() {
    use unit_dialect::*;

    let mut errors = Vec::new();
    let valid_args = [Expr::u32(42)];
    validate_call(Op::Beta, &valid_args, 3, &mut errors);
    assert!(errors.is_empty());

    // Arity mismatch
    let mut arity_errors = Vec::new();
    let bad_args = [Expr::u32(1), Expr::u32(2)];
    validate_call(Op::Beta, &bad_args, 3, &mut arity_errors);
    assert_eq!(arity_errors.len(), 1);
    assert_eq!(arity_errors[0].code().as_str(), "V020");

    // Version mismatch: beta introduced in v3, validated against target v2
    let mut version_errors = Vec::new();
    validate_call(Op::Beta, &valid_args, 2, &mut version_errors);
    assert_eq!(version_errors.len(), 1);
    assert_eq!(version_errors[0].code().as_str(), "V023");
}

#[test]
fn test_external_schema_adversarial_rejections() {
    use unit_dialect::*;

    // Valid external node
    let valid_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![
            ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "4".to_string() },
            ExternalField { name: "tag".to_string(), declared_member: FieldType::String, raw_value: "active".to_string() },
        ],
        bound_resources: vec!["scratch_buf".to_string()],
    };
    let mapped = validate_external_node(&valid_node).expect("valid node must map cleanly");
    assert_eq!(mapped, Op::Alpha);

    // Adversarial Case 1: Unknown field
    let unknown_field_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![
            ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "4".to_string() },
            ExternalField { name: "unrecognized_field".to_string(), declared_member: FieldType::String, raw_value: "val".to_string() },
        ],
        bound_resources: vec!["scratch_buf".to_string()],
    };
    let err =
        validate_external_node(&unknown_field_node).expect_err("unknown field must be rejected");
    assert!(matches!(err, SchemaTranslationError::UnknownField { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 2: Duplicate field
    let duplicate_field_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![
            ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "4".to_string() },
            ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "8".to_string() },
        ],
        bound_resources: vec!["scratch_buf".to_string()],
    };
    let err = validate_external_node(&duplicate_field_node)
        .expect_err("duplicate field must be rejected");
    assert!(matches!(err, SchemaTranslationError::DuplicateField { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 3: Missing required field
    let missing_field_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![ExternalField { name: "tag".to_string(), declared_member: FieldType::String, raw_value: "active".to_string() }],
        bound_resources: vec!["scratch_buf".to_string()],
    };
    let err = validate_external_node(&missing_field_node)
        .expect_err("missing required field must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::MissingRequiredField { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 4: Incomplete resource roster
    let incomplete_resource_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "4".to_string() }],
        bound_resources: vec![], // Missing scratch_buf
    };
    let err = validate_external_node(&incomplete_resource_node)
        .expect_err("incomplete resource roster must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::IncompleteResourceRoster { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 5: Unmapped external node (non-exhaustive mapping rejected)
    let unmapped_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::unknown_op".to_string(),
        fields: vec![],
        bound_resources: vec![],
    };
    let err = validate_external_node(&unmapped_node).expect_err("unmapped node must fail closed");
    assert!(matches!(err, SchemaTranslationError::UnmappedNode { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 6: Incompatible identity
    let id_err = validate_schema_identity(
        "vyre-unit::dialect",
        "wrong-dialect::name",
        3,
        "vyre-unit::dialect",
        3,
    )
    .expect_err("incompatible dialect identity must fail");
    assert!(matches!(
        id_err,
        SchemaTranslationError::IncompatibleIdentity { .. }
    ));
    assert!(id_err.to_string().contains("Fix:"));

    // Adversarial Case 7: Overflowing field value
    let overflowing_field_node = ExternalSchemaNode {
        op_name: "vyre-unit::dialect::alpha".to_string(),
        fields: vec![
            ExternalField { name: "scale".to_string(), declared_member: FieldType::U32, raw_value: "999999999999999999999999".to_string() },
            ExternalField { name: "tag".to_string(), declared_member: FieldType::String, raw_value: "active".to_string() },
        ],
        bound_resources: vec!["scratch_buf".to_string()],
    };
    let err = validate_external_node(&overflowing_field_node)
        .expect_err("overflowing field value must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::OverflowingField { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Complete external schema with unused declared resource
    let unused_res_schema = ExternalSchema {
        schema_id: "vyre-unit::dialect".to_string(),
        version: 3,
        nodes: vec![valid_node.clone()],
        declared_resources: vec![
            ExternalResourceDeclaration {
                name: "scratch_buf".to_string(),
                access: BufferAccess::ReadWrite,
                element_type: DataType::U32,
                alignment: 16,
                byte_capacity: 64,
            },
            ExternalResourceDeclaration {
                name: "unused_extra_buf".to_string(),
                access: BufferAccess::ReadOnly,
                element_type: DataType::U32,
                alignment: 16,
                byte_capacity: 128,
            },
        ],
        declared_layouts: vec![],
    };
    let err = validate_external_schema("vyre-unit::dialect", &unused_res_schema, 3, |node| {
        validate_external_node(node).map(|_| ())
    })
    .expect_err("unused declared resource must be rejected");
    assert!(matches!(err, SchemaTranslationError::UnusedResource { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Complete external schema with overflowing layout
    let overflowing_layout_schema = ExternalSchema {
        schema_id: "vyre-unit::dialect".to_string(),
        version: 3,
        nodes: vec![valid_node.clone()],
        declared_resources: vec![ExternalResourceDeclaration {
            name: "scratch_buf".to_string(),
            access: BufferAccess::ReadWrite,
            element_type: DataType::U32,
            alignment: 16,
            byte_capacity: 64,
        }],
        declared_layouts: vec![ExternalLayoutDeclaration {
            resource_name: "scratch_buf".to_string(),
            element_type: DataType::U32,
            shape: vec![u64::MAX, u64::MAX],
            strides: vec![u64::MAX, 1],
            alignment: 16,
        }],
    };
    let err = validate_external_schema(
        "vyre-unit::dialect",
        &overflowing_layout_schema,
        3,
        |node| validate_external_node(node).map(|_| ()),
    )
    .expect_err("overflowing layout must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::OverflowingLayout { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Complete external schema with incompatible layout element type
    let incompatible_layout_schema = ExternalSchema {
        schema_id: "vyre-unit::dialect".to_string(),
        version: 3,
        nodes: vec![valid_node.clone()],
        declared_resources: vec![ExternalResourceDeclaration {
            name: "scratch_buf".to_string(),
            access: BufferAccess::ReadWrite,
            element_type: DataType::U32,
            alignment: 16,
            byte_capacity: 64,
        }],
        declared_layouts: vec![ExternalLayoutDeclaration {
            resource_name: "scratch_buf".to_string(),
            element_type: DataType::F32,
            shape: vec![16],
            strides: vec![1],
            alignment: 16,
        }],
    };
    let err = validate_external_schema(
        "vyre-unit::dialect",
        &incompatible_layout_schema,
        3,
        |node| validate_external_node(node).map(|_| ()),
    )
    .expect_err("incompatible layout element type must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::IncompatibleLayout { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Valid complete external schema
    let valid_schema = ExternalSchema {
        schema_id: "vyre-unit::dialect".to_string(),
        version: 3,
        nodes: vec![valid_node],
        declared_resources: vec![ExternalResourceDeclaration {
            name: "scratch_buf".to_string(),
            access: BufferAccess::ReadWrite,
            element_type: DataType::U32,
            alignment: 16,
            byte_capacity: 64,
        }],
        declared_layouts: vec![ExternalLayoutDeclaration {
            resource_name: "scratch_buf".to_string(),
            element_type: DataType::U32,
            shape: vec![16],
            strides: vec![1],
            alignment: 16,
        }],
    };
    let bound = validate_external_schema("vyre-unit::dialect", &valid_schema, 3, |node| {
        validate_external_node(node).map(|_| ())
    })
    .expect("valid complete schema must pass validation");
    assert!(bound.contains("scratch_buf"));

    // Canonical identity is deterministic and non-zero
    let id1 = valid_schema.canonical_identity();
    let id2 = valid_schema.canonical_identity();
    assert_eq!(id1, id2);
    assert_ne!(id1, [0; 32]);
}
