//! Tests for declarative and versioned dialect infrastructure.
//!
//! Asserts the 9 required dialect generation stages:
//! 1. Typed builders
//! 2. Schema and metadata
//! 3. Visitor traversal
//! 4. Serialization / wire codec
//! 5. Semantic validation & error diagnostics with Fix:
//! 6. Rewrite and pattern matching
//! 7. Version compatibility & stale version rejection
//! 8. Closure rosters and exhaustive variant coverage
//! 9. External schema, field, and resource ABI translation validation

use vyre_foundation::define_dialect;
use vyre_foundation::dialect::{
    admit_descriptor_versions, admit_program_versions, admit_registered_versions,
    validate_dialect_version, validate_schema_identity, Dialect, DialectRegistry,
    DialectVersionError, ExternalSchemaNode, FieldContract, FieldType, ResourceAbi,
    ResourceBinding, SchemaTranslationError, SemanticVersionRejection,
};
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::{BufferAccess, DataType, Expr};
use vyre_foundation::operation::OperationTier;

const TEST_UNARY_SIG: Signature = vyre_test_support::u32_signature! {
    inputs: ["x"],
    output: "out",
};

const TEST_BINARY_SIG: Signature = vyre_test_support::u32_signature! {
    inputs: ["a", "b"],
    output: "out",
};

static INVERT_FIELDS: &[FieldContract] = &[FieldContract {
    name: "mask",
    field_type: FieldType::U32,
    required: false,
}];

static BLEND_RESOURCES: &[ResourceBinding] = &[
    ResourceBinding {
        name: "input_a",
        access: BufferAccess::ReadOnly,
        element_type: DataType::U32,
        alignment: 16,
        minimum_bytes: 32,
    },
    ResourceBinding {
        name: "input_b",
        access: BufferAccess::ReadOnly,
        element_type: DataType::U32,
        alignment: 16,
        minimum_bytes: 32,
    },
    ResourceBinding {
        name: "output_c",
        access: BufferAccess::WriteOnly,
        element_type: DataType::U32,
        alignment: 16,
        minimum_bytes: 32,
    },
];

static BLEND_ABI: ResourceAbi = ResourceAbi {
    resources: BLEND_RESOURCES,
};

define_dialect! {
    /// Sample declarative test dialect.
    dialect: "vyre-test::demo",
    name: test_demo_dialect,
    visitor: TestDemoVisitor,
    version: 2,
    min_supported_version: 1,
    tier: OperationTier::Library,
    category: "test_demo",
    summary: "Declarative test dialect for compiler stages.",

    operations: [
        {
            op: Invert,
            discriminant: 0,
            name: "invert",
            id: "vyre-test::demo::invert",
            version: 1,
            summary: "Bitwise invert.",
            signature: TEST_UNARY_SIG,
            is_composable: true,
            call_builder: call_invert,
            fields: INVERT_FIELDS,
        },
        {
            op: Blend,
            discriminant: 1,
            name: "blend",
            id: "vyre-test::demo::blend",
            version: 2,
            summary: "Weighted blend.",
            signature: TEST_BINARY_SIG,
            is_composable: true,
            call_builder: call_blend,
            resource_abi: BLEND_ABI,
        },
    ]
}

#[test]
fn dialect_schema_and_metadata_are_registered() {
    use test_demo_dialect::*;

    assert_eq!(SCHEMA_VERSION, 2);
    assert_eq!(MIN_SUPPORTED_VERSION, 1);
    assert_eq!(DIALECT_ID, "vyre-test::demo");
    assert_eq!(CATEGORY, "test_demo");

    let desc = DialectMarker::descriptor();
    assert_eq!(desc.id, "vyre-test::demo");
    assert_eq!(desc.version, 2);
    assert_eq!(desc.min_supported_version, 1);
    assert_eq!(desc.operations.len(), 2);

    assert_eq!(desc.operations[0].id, "vyre-test::demo::invert");
    assert_eq!(desc.operations[0].version, 1);
    assert!(desc.operations[0].is_composable);

    assert_eq!(desc.operations[1].id, "vyre-test::demo::blend");
    assert_eq!(desc.operations[1].version, 2);
    assert!(desc.operations[1].is_composable);

    // Global registry lookup
    let registered = DialectRegistry::get("vyre-test::demo")
        .expect("Fix: test dialect must be in the global registry");
    assert_eq!(registered.id, "vyre-test::demo");

    let found_by_op = DialectRegistry::find_by_op_id("vyre-test::demo::blend")
        .expect("Fix: dialect must be found by op id");
    assert_eq!(found_by_op.id, "vyre-test::demo");
}

#[test]
fn dialect_op_traits_and_builders() {
    use test_demo_dialect::*;

    assert_eq!(Op::Invert.op_id(), "vyre-test::demo::invert");
    assert_eq!(Op::Invert.op_name(), "invert");
    assert_eq!(Op::Invert.introduced_version(), 1);
    assert!(Op::Invert.is_composable());

    assert_eq!(Op::Blend.op_id(), "vyre-test::demo::blend");
    assert_eq!(Op::Blend.op_name(), "blend");
    assert_eq!(Op::Blend.introduced_version(), 2);
    assert!(Op::Blend.is_composable());

    // Typed call builders
    let inv_call = call_invert(vec![Expr::u32(42)]);
    if let Expr::Call { op_id, args } = &inv_call {
        assert_eq!(op_id.as_str(), "vyre-test::demo::invert");
        assert_eq!(args.len(), 1);
    } else {
        panic!("expected Expr::Call");
    }

    let blend_call = call_blend(vec![Expr::u32(10), Expr::u32(20)]);
    if let Expr::Call { op_id, args } = &blend_call {
        assert_eq!(op_id.as_str(), "vyre-test::demo::blend");
        assert_eq!(args.len(), 2);
    } else {
        panic!("expected Expr::Call");
    }
}

#[test]
fn dialect_pattern_matching() {
    use test_demo_dialect::*;

    assert_eq!(match_op_id("vyre-test::demo::invert"), Some(Op::Invert));
    assert_eq!(match_op_id("vyre-test::demo::blend"), Some(Op::Blend));
    assert_eq!(match_op_id("unknown::op"), None);

    let expr = call_blend(vec![Expr::u32(1), Expr::u32(2)]);
    let matched = match_call(&expr).expect("Fix: expression should match dialect");
    assert_eq!(matched.0, Op::Blend);
    assert_eq!(matched.1.len(), 2);

    let non_call = Expr::u32(42);
    assert!(match_call(&non_call).is_none());
}

struct CountingVisitor {
    inverts: usize,
    blends: usize,
}

impl test_demo_dialect::TestDemoVisitor for CountingVisitor {
    type Output = &'static str;

    fn Invert(&mut self, _args: &[Expr]) -> Self::Output {
        self.inverts += 1;
        "visited_invert"
    }

    fn Blend(&mut self, _args: &[Expr]) -> Self::Output {
        self.blends += 1;
        "visited_blend"
    }
}

#[test]
fn dialect_visitor_dispatch() {
    use test_demo_dialect::*;

    let mut visitor = CountingVisitor {
        inverts: 0,
        blends: 0,
    };
    let args1 = [Expr::u32(1)];
    let res1 = dispatch_visitor(Op::Invert, &args1, &mut visitor);
    assert_eq!(res1, "visited_invert");
    assert_eq!(visitor.inverts, 1);

    let args2 = [Expr::u32(1), Expr::u32(2)];
    let res2 = dispatch_visitor(Op::Blend, &args2, &mut visitor);
    assert_eq!(res2, "visited_blend");
    assert_eq!(visitor.blends, 1);
}

#[test]
fn dialect_wire_codec_roundtrip() {
    use test_demo_dialect::*;

    let mut buffer = Vec::new();
    encode_op(Op::Invert, &mut buffer).expect("Fix: encode must succeed");
    encode_op(Op::Blend, &mut buffer).expect("Fix: encode must succeed");

    let mut cursor = std::io::Cursor::new(&buffer);
    let decoded1 = decode_op(&mut cursor)
        .expect("Fix: decode must succeed")
        .expect("Fix: valid discriminant");
    let decoded2 = decode_op(&mut cursor)
        .expect("Fix: decode must succeed")
        .expect("Fix: valid discriminant");

    assert_eq!(decoded1, Op::Invert);
    assert_eq!(decoded2, Op::Blend);
}

#[test]
fn dialect_versioning_and_rejection() {
    let desc = test_demo_dialect::DESCRIPTOR;

    // Supported versions (1 and 2)
    assert!(validate_dialect_version(&desc, 1).is_ok());
    assert!(validate_dialect_version(&desc, 2).is_ok());

    // Stale version (0 < min_supported_version 1)
    let stale_err = validate_dialect_version(&desc, 0).expect_err("stale version must fail");
    match stale_err {
        DialectVersionError::StaleVersion {
            dialect,
            found,
            min_supported,
            current,
        } => {
            assert_eq!(dialect, "vyre-test::demo");
            assert_eq!(found, 0);
            assert_eq!(min_supported, 1);
            assert_eq!(current, 2);
            assert!(stale_err.to_string().contains("Fix:"));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Unsupported future version (3 > schema_version 2)
    let future_err = validate_dialect_version(&desc, 3).expect_err("future version must fail");
    match future_err {
        DialectVersionError::UnsupportedVersion {
            dialect,
            found,
            current,
        } => {
            assert_eq!(dialect, "vyre-test::demo");
            assert_eq!(found, 3);
            assert_eq!(current, 2);
            assert!(future_err.to_string().contains("Fix:"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn dialect_validation_hooks() {
    use test_demo_dialect::*;

    // Valid call at version 2
    let mut errors = Vec::new();
    let valid_args = [Expr::u32(1), Expr::u32(2)];
    validate_call(Op::Blend, &valid_args, 2, &mut errors);
    assert!(errors.is_empty());

    // Arity mismatch
    let mut arity_errors = Vec::new();
    let bad_args = [Expr::u32(1)];
    validate_call(Op::Blend, &bad_args, 2, &mut arity_errors);
    assert_eq!(arity_errors.len(), 1);
    assert_eq!(arity_errors[0].code().as_str(), "V020");

    // Version mismatch: blend was introduced in v2, validating at target version 1
    let mut version_errors = Vec::new();
    validate_call(Op::Blend, &valid_args, 1, &mut version_errors);
    assert_eq!(version_errors.len(), 1);
    assert_eq!(version_errors[0].code().as_str(), "V023");
}

#[test]
fn dialect_closure_roster() {
    use test_demo_dialect::*;

    assert_eq!(
        ALL_OP_IDS,
        &["vyre-test::demo::invert", "vyre-test::demo::blend"]
    );
    assert_eq!(OPERATIONS.len(), 2);
    assert_eq!(OPERATIONS[0].id, "vyre-test::demo::invert");
    assert_eq!(OPERATIONS[1].id, "vyre-test::demo::blend");
}

#[test]
fn dialect_external_schema_translation_and_adversarial_rejections() {
    use test_demo_dialect::*;

    // Valid external node mapping
    let valid_node = ExternalSchemaNode {
        op_name: "vyre-test::demo::blend".to_string(),
        raw_fields: vec![],
        bound_resources: vec![
            "input_a".to_string(),
            "input_b".to_string(),
            "output_c".to_string(),
        ],
    };
    let mapped = validate_external_node(&valid_node).expect("valid blend mapping");
    assert_eq!(mapped, Op::Blend);

    // Adversarial Case 1: Incomplete resource roster
    let missing_resource_node = ExternalSchemaNode {
        op_name: "vyre-test::demo::blend".to_string(),
        raw_fields: vec![],
        bound_resources: vec!["input_a".to_string(), "input_b".to_string()], // Missing output_c
    };
    let err = validate_external_node(&missing_resource_node)
        .expect_err("incomplete resource roster must be rejected");
    assert!(matches!(
        err,
        SchemaTranslationError::IncompleteResourceRoster { .. }
    ));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 2: Unknown field
    let unknown_field_node = ExternalSchemaNode {
        op_name: "vyre-test::demo::invert".to_string(),
        raw_fields: vec![("extraneous_field".to_string(), "1".to_string())],
        bound_resources: vec![],
    };
    let err =
        validate_external_node(&unknown_field_node).expect_err("unknown field must be rejected");
    assert!(matches!(err, SchemaTranslationError::UnknownField { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 3: Duplicate field
    let duplicate_field_node = ExternalSchemaNode {
        op_name: "vyre-test::demo::invert".to_string(),
        raw_fields: vec![
            ("mask".to_string(), "0xFF".to_string()),
            ("mask".to_string(), "0x00".to_string()),
        ],
        bound_resources: vec![],
    };
    let err = validate_external_node(&duplicate_field_node)
        .expect_err("duplicate field must be rejected");
    assert!(matches!(err, SchemaTranslationError::DuplicateField { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 4: Unmapped node (non-exhaustive mapping)
    let unmapped = ExternalSchemaNode {
        op_name: "vyre-test::demo::nonexistent".to_string(),
        raw_fields: vec![],
        bound_resources: vec![],
    };
    let err = validate_external_node(&unmapped).expect_err("unmapped node must fail closed");
    assert!(matches!(err, SchemaTranslationError::UnmappedNode { .. }));
    assert!(err.to_string().contains("Fix:"));

    // Adversarial Case 5: Incompatible schema identity
    let id_err =
        validate_schema_identity("vyre-test::demo", "wrong::dialect", 2, "vyre-test::demo", 2)
            .expect_err("incompatible schema identity must fail");
    assert!(matches!(
        id_err,
        SchemaTranslationError::IncompatibleIdentity { .. }
    ));
    assert!(id_err.to_string().contains("Fix:"));
}

/// WHY: a dialect whose supported floor is above its schema version admits no
/// declarable version, and an operation declaring a version its dialect has not
/// reached is unreachable at every one. Deriving the roster from the live
/// registry turns a newly registered dialect red rather than leaving it
/// unchecked.
#[test]
fn every_registered_dialect_admits_its_own_version_declarations() {
    admit_registered_versions().expect("every registered dialect must admit its own declarations");
    let registry = DialectRegistry::global();
    assert!(
        !registry.is_empty(),
        "the dialect registry must register at least one dialect for this contract to prove anything"
    );
    for (id, descriptor) in registry {
        admit_descriptor_versions(descriptor).unwrap_or_else(|error| {
            panic!("dialect `{id}` declares an inadmissible version: {error}")
        });
        assert!(
            descriptor.min_supported_version <= descriptor.version,
            "dialect `{id}` supports a floor of {} above its schema version {}",
            descriptor.min_supported_version,
            descriptor.version
        );
        for op in descriptor.operations {
            assert!(
                op.version <= descriptor.version,
                "operation `{}` declares version {} above dialect `{id}` schema version {}",
                op.id,
                op.version,
                descriptor.version
            );
        }
    }
}

/// WHY: a declared schema version has to decide something. Every registered
/// operation introduced above its dialect's floor is rejected one version below
/// its introduction and admitted at it, so a version declaration that authorized
/// nothing would fail here.
#[test]
fn a_declared_version_admits_an_operation_only_from_its_introduction() {
    let mut proven = 0_usize;
    for (id, descriptor) in DialectRegistry::global() {
        for op in descriptor.operations {
            let admitted = op.version.max(descriptor.min_supported_version);
            let declared = std::collections::BTreeMap::from([((*id).to_owned(), admitted)]);
            admit_program_versions(&declared, &[op.id]).unwrap_or_else(|error| {
                panic!(
                    "operation `{}` must be admitted at version {admitted}: {error}",
                    op.id
                )
            });
            if op.version <= descriptor.min_supported_version {
                continue;
            }
            let stale = std::collections::BTreeMap::from([((*id).to_owned(), op.version - 1)]);
            let rejection = admit_program_versions(&stale, &[op.id])
                .expect_err("an operation introduced after the declared version must be rejected");
            assert_eq!(
                rejection,
                SemanticVersionRejection::Version(DialectVersionError::OperationVersionMismatch {
                    op_id: op.id,
                    introduced_in: op.version,
                    target_version: op.version - 1,
                }),
                "dialect `{id}` operation `{}` produced the wrong rejection",
                op.id
            );
            proven += 1;
        }
    }
    assert!(
        proven > 0,
        "no registered operation is introduced above its dialect floor, so this contract proves nothing"
    );
}

/// WHY: a stale declaration must be refused rather than read as current, and a
/// dialect nothing registers must be refused rather than ignored.
#[test]
fn a_stale_or_unregistered_declaration_is_refused() {
    let (id, descriptor) = DialectRegistry::global()
        .iter()
        .find(|(_, descriptor)| descriptor.min_supported_version > 0)
        .expect("at least one registered dialect must state a supported floor above zero");
    let stale = std::collections::BTreeMap::from([(
        (*id).to_owned(),
        descriptor.min_supported_version - 1,
    )]);
    assert_eq!(
        admit_program_versions(&stale, &[]),
        Err(SemanticVersionRejection::Version(
            DialectVersionError::StaleVersion {
                dialect: descriptor.id,
                found: descriptor.min_supported_version - 1,
                min_supported: descriptor.min_supported_version,
                current: descriptor.version,
            }
        ))
    );
    let future = std::collections::BTreeMap::from([((*id).to_owned(), descriptor.version + 1)]);
    assert_eq!(
        admit_program_versions(&future, &[]),
        Err(SemanticVersionRejection::Version(
            DialectVersionError::UnsupportedVersion {
                dialect: descriptor.id,
                found: descriptor.version + 1,
                current: descriptor.version,
            }
        ))
    );
    let unregistered =
        std::collections::BTreeMap::from([("vyre::no-such-dialect".to_owned(), 1_u32)]);
    assert_eq!(
        admit_program_versions(&unregistered, &[]),
        Err(SemanticVersionRejection::UnregisteredDialect {
            dialect: "vyre::no-such-dialect".to_owned(),
        })
    );
}
