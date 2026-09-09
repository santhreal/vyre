//! Integration tests for closed declarative extension schemas in foundation:
//! collision rejection by name, missing proof field rejection, canonical decode/re-encode equality,
//! stable content interning, full-identity hashing, and reconciled effect/divergence walks.

use std::sync::Arc;
use vyre_foundation::extension::{
    ExtensionCatalogBundle, ExtensionCatalogError, OpaqueExprResolver, OpaqueNodeResolver,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, ExprNode, Node, NodeExtension, Program,
};
use vyre_foundation::optimizer::expr_arena::ExprArena;
use vyre_spec::{
    ExtensionIdentity, ExtensionNamespace, ExtensionNumericalContract, ExtensionProofFields,
    ExtensionResourceBounds, ExtensionSchema, ExtensionSemVer, SideEffectClass,
};

#[derive(Debug)]
struct CanonicalTestExpr {
    payload: Vec<u8>,
    pure: bool,
}

impl ExprNode for CanonicalTestExpr {
    fn extension_kind(&self) -> &'static str {
        "test.canonical.expr"
    }

    fn debug_identity(&self) -> &str {
        "canonical-test-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        self.pure
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }
}

#[derive(Debug)]
struct CanonicalTestNode {
    payload: Vec<u8>,
    pure: bool,
    divergent: bool,
}

impl NodeExtension for CanonicalTestNode {
    fn extension_kind(&self) -> &'static str {
        "test.canonical.node"
    }

    fn debug_identity(&self) -> &str {
        "canonical-test-node"
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }

    fn is_pure(&self) -> bool {
        self.pure
    }

    fn is_divergent(&self) -> bool {
        self.divergent
    }
}

fn deserialize_canonical_expr(bytes: &[u8]) -> Result<Arc<dyn ExprNode>, String> {
    // Deliberate bad decode: if payload starts with 0xFF, corrupt the decoded payload so it fails roundtrip
    if bytes.first() == Some(&0xFF) {
        return Ok(Arc::new(CanonicalTestExpr {
            payload: vec![0x00],
            pure: true,
        }));
    }
    Ok(Arc::new(CanonicalTestExpr {
        payload: bytes.to_vec(),
        pure: true,
    }))
}

fn deserialize_canonical_node(bytes: &[u8]) -> Result<Arc<dyn NodeExtension>, String> {
    if bytes.first() == Some(&0xFF) {
        return Ok(Arc::new(CanonicalTestNode {
            payload: vec![0x00],
            pure: false,
            divergent: true,
        }));
    }
    Ok(Arc::new(CanonicalTestNode {
        payload: bytes.to_vec(),
        pure: false,
        divergent: true,
    }))
}

inventory::submit! {
    OpaqueExprResolver {
        kind: "test.canonical.expr",
        deserialize: deserialize_canonical_expr,
    }
}

inventory::submit! {
    OpaqueNodeResolver {
        kind: "test.canonical.node",
        deserialize: deserialize_canonical_node,
    }
}

#[test]
fn catalog_bundle_refuses_duplicate_identities_and_version_collisions_by_name() {
    let mut bundle = ExtensionCatalogBundle::new("proof_bundle");
    let ns = ExtensionNamespace::new("test.collision.refusal").unwrap();
    let ver = ExtensionSemVer::new(1, 0, 0);

    let proof_a = ExtensionProofFields {
        host_shareable: true,
        is_pure: true,
        cse_eligible: true,
        is_divergent: false,
        may_alias: false,
        terminates: true,
        target_capability: "cuda_sm90".into(),
    };
    let digest_a = ExtensionSchema::compute_digest(ns.as_str(), &ver, &[], &[], &[], &proof_a);
    let id_a = ExtensionIdentity::new(ns.clone(), ver, digest_a);

    let schema_a = ExtensionSchema {
        identity: id_a.clone(),
        display_name: "Original Extension".into(),
        description: "First registration".into(),
        fields: Vec::new(),
        operands: Vec::new(),
        result_types: Vec::new(),
        side_effects: SideEffectClass::Pure,
        shape_rules: Vec::new(),
        numerical_contract: ExtensionNumericalContract::default(),
        laws: Vec::new(),
        resource_bounds: ExtensionResourceBounds::default(),
        proof_fields: proof_a.clone(),
    };

    bundle
        .register(schema_a.clone())
        .expect("initial registration succeeds");

    // 1. Exact duplicate identity refusal
    let duplicate_err = bundle
        .register(schema_a)
        .expect_err("Exact duplicate identity must be refused");
    match duplicate_err {
        ExtensionCatalogError::DuplicateIdentity(colliding_id) => {
            assert_eq!(colliding_id, id_a);
        }
        other => panic!("Unexpected error: {other:?}"),
    }

    // 2. Conflicting version for same namespace refusal
    let mut proof_b = proof_a;
    proof_b.target_capability = "cuda_sm80".into();
    let digest_b = ExtensionSchema::compute_digest(ns.as_str(), &ver, &[], &[], &[], &proof_b);
    let id_b = ExtensionIdentity::new(ns.clone(), ver, digest_b);

    let schema_b = ExtensionSchema {
        identity: id_b.clone(),
        display_name: "Conflicting Extension".into(),
        description: "Same namespace and version, different digest".into(),
        fields: Vec::new(),
        operands: Vec::new(),
        result_types: Vec::new(),
        side_effects: SideEffectClass::Pure,
        shape_rules: Vec::new(),
        numerical_contract: ExtensionNumericalContract::default(),
        laws: Vec::new(),
        resource_bounds: ExtensionResourceBounds::default(),
        proof_fields: proof_b,
    };

    let version_collision_err = bundle
        .register(schema_b)
        .expect_err("Conflicting definition on same namespace and version must be refused");
    match version_collision_err {
        ExtensionCatalogError::DuplicateNamespaceVersion {
            namespace,
            version,
            first_id,
            second_id,
        } => {
            assert_eq!(namespace, ns);
            assert_eq!(version, ver);
            assert_eq!(first_id, id_a);
            assert_eq!(second_id, id_b);
        }
        other => panic!("Unexpected error: {other:?}"),
    }
}

#[test]
fn catalog_bundle_refuses_empty_required_proof_fields() {
    let mut bundle = ExtensionCatalogBundle::new("proof_check_bundle");
    let ns = ExtensionNamespace::new("test.proof.missing").unwrap();
    let ver = ExtensionSemVer::new(1, 0, 0);

    let empty_proof = ExtensionProofFields {
        host_shareable: true,
        is_pure: true,
        cse_eligible: true,
        is_divergent: false,
        may_alias: false,
        terminates: true,
        target_capability: "   ".into(), // Blank/whitespace target capability
    };
    let digest = ExtensionSchema::compute_digest(ns.as_str(), &ver, &[], &[], &[], &empty_proof);
    let id = ExtensionIdentity::new(ns, ver, digest);

    let schema = ExtensionSchema {
        identity: id,
        display_name: "Invalid Proof Extension".into(),
        description: "Omitted proof field".into(),
        fields: Vec::new(),
        operands: Vec::new(),
        result_types: Vec::new(),
        side_effects: SideEffectClass::Pure,
        shape_rules: Vec::new(),
        numerical_contract: ExtensionNumericalContract::default(),
        laws: Vec::new(),
        resource_bounds: ExtensionResourceBounds::default(),
        proof_fields: empty_proof,
    };

    let err = bundle
        .register(schema)
        .expect_err("Empty target_capability must be refused at registration");
    assert!(matches!(err, ExtensionCatalogError::MissingProofField(_)));
}

#[test]
fn opaque_expr_interning_keys_on_stable_content_not_pointer_identity() {
    let node_a = Arc::new(CanonicalTestExpr {
        payload: vec![1, 2, 3, 4],
        pure: true,
    });
    let node_b = Arc::new(CanonicalTestExpr {
        payload: vec![1, 2, 3, 4],
        pure: true,
    });

    assert!(
        !Arc::ptr_eq(&node_a, &node_b),
        "Two distinct allocations must have different pointers"
    );

    let mut arena = ExprArena::default();
    let id_a = arena.intern(&Expr::Opaque(node_a));
    let id_b = arena.intern(&Expr::Opaque(node_b));

    assert_eq!(
        id_a, id_b,
        "Interning must key on stable content fingerprint and kind, producing identical ExprIds"
    );
    assert_eq!(arena.len(), 1);

    // Rebuild produces an expression equal in value
    let rebuilt = arena.rebuild(id_a);
    match rebuilt {
        Expr::Opaque(ext) => {
            assert_eq!(ext.extension_kind(), "test.canonical.expr");
            assert_eq!(ext.wire_payload(), vec![1, 2, 3, 4]);
        }
        _ => panic!("Rebuilt expression must be Expr::Opaque"),
    }
}

#[test]
fn program_hashing_records_full_opaque_identity() {
    let prog1 = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Opaque(Arc::new(CanonicalTestExpr {
                payload: vec![10, 20, 30],
                pure: true,
            })),
        )],
    );

    let prog2 = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Opaque(Arc::new(CanonicalTestExpr {
                payload: vec![10, 20, 31], // different payload byte
                pure: true,
            })),
        )],
    );

    assert_ne!(
        prog1.canonical_wire_hash().expect("first program hashes"),
        prog2.canonical_wire_hash().expect("second program hashes"),
        "Program hash must differ when opaque payload/fingerprint differs"
    );
}

#[test]
fn wire_decode_rejects_payloads_that_do_not_round_trip_canonically() {
    // 1. Valid payload round-trips byte-for-byte
    let valid_prog = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Opaque(Arc::new(CanonicalTestExpr {
                payload: vec![0x42, 0x43, 0x44],
                pure: true,
            })),
        )],
    );

    let valid_wire = valid_prog.to_wire().expect("valid program encodes");
    let decoded_prog = Program::from_wire(&valid_wire).expect("valid program decodes");
    assert_eq!(decoded_prog, valid_prog);

    // 2. Corrupted payload (starts with 0xFF) is rejected during decode
    let invalid_prog = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Opaque(Arc::new(CanonicalTestExpr {
                payload: vec![0xFF, 0x01, 0x02],
                pure: true,
            })),
        )],
    );

    let invalid_wire = invalid_prog.to_wire().expect("encodes wire bytes");
    let decode_err = Program::from_wire(&invalid_wire).expect_err(
        "Decoder must reject extension whose deserialization fails byte equality round-trip",
    );
    assert!(
        decode_err
            .to_string()
            .contains("Canonical decode/re-encode mismatch"),
        "decode failure must name the canonical round-trip mismatch, got: {decode_err}"
    );
}
