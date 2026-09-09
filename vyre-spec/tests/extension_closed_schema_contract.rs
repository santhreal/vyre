//! Proof tests for closed declarative extension schemas, collision-resistant IDs,
//! runtime-derived proof fields, and canonical round-trip verification.

use vyre_spec::{
    ExtensionDataTypeId, ExtensionIdentity, ExtensionNamespace, ExtensionNumericalContract,
    ExtensionProofFieldKind, ExtensionProofFields, ExtensionResourceBounds, ExtensionSchema,
    ExtensionSemVer, SideEffectClass,
};

#[test]
fn distinct_extension_names_produce_distinct_ids_without_fnv1a_collision() {
    // Known legacy 31-bit FNV-1a collision pair
    let name_a = "d5pj";
    let name_b = "x.ta";
    let id_a = ExtensionDataTypeId::from_name(name_a);
    let id_b = ExtensionDataTypeId::from_name(name_b);

    assert_ne!(
        id_a, id_b,
        "Collision-resistant full digest must produce distinct IDs for legacy FNV-1a colliding names"
    );
    assert!(id_a.is_extension());
    assert!(id_b.is_extension());

    // Additional diverse distinct names
    let names = [
        "tensor.gather",
        "tensor.scatter",
        "tensor.matmul_fp8",
        "custom.fp16_gemm",
        "vendor.speculate",
        "dialect.alpha",
        "dialect.beta",
    ];
    let mut ids = std::collections::BTreeSet::new();
    for name in names {
        let id = ExtensionDataTypeId::from_name(name);
        assert!(id.is_extension(), "Extension ID must have high bit set");
        assert!(
            ids.insert(id.as_u32()),
            "Distinct extension name `{name}` produced colliding ID {:#010x}",
            id.as_u32()
        );
    }
}

#[test]
fn runtime_derived_proof_field_set_is_exhaustive() {
    // Closes the class: deriving the proof field set at runtime ensures that
    // adding a new proof field without a decision turns the test red.
    let derived_fields = ExtensionProofFieldKind::ALL;

    // Verify all 7 required proof fields are enumerated
    let expected_names = [
        "host_shareable",
        "is_pure",
        "cse_eligible",
        "is_divergent",
        "may_alias",
        "terminates",
        "target_capability",
    ];

    assert_eq!(
        derived_fields.len(),
        expected_names.len(),
        "Proof field count mismatch: expected exactly {} proof fields",
        expected_names.len()
    );

    for (field_kind, &expected_name) in derived_fields.iter().zip(expected_names.iter()) {
        assert_eq!(field_kind.as_str(), expected_name);

        // Exhaustive match without wildcard to ensure compile-time closure
        match field_kind {
            ExtensionProofFieldKind::HostShareability => {}
            ExtensionProofFieldKind::Purity => {}
            ExtensionProofFieldKind::CseEligibility => {}
            ExtensionProofFieldKind::Divergence => {}
            ExtensionProofFieldKind::Aliasing => {}
            ExtensionProofFieldKind::Termination => {}
            ExtensionProofFieldKind::TargetCapability => {}
        }
    }
}

#[test]
fn canonical_schema_digest_and_identity_round_trip() {
    let namespace = ExtensionNamespace::new("org.vyre.test.extension").expect("valid namespace");
    let version = ExtensionSemVer::new(1, 2, 3);
    let proof_fields = ExtensionProofFields {
        host_shareable: true,
        is_pure: true,
        cse_eligible: true,
        is_divergent: false,
        may_alias: false,
        terminates: true,
        target_capability: "sm_90a".into(),
    };

    let digest =
        ExtensionSchema::compute_digest(namespace.as_str(), &version, &[], &[], &[], &proof_fields);

    let identity = ExtensionIdentity::new(namespace.clone(), version, digest);
    let canonical_str = identity.to_canonical_string();
    assert!(canonical_str.starts_with("org.vyre.test.extension@1.2.3#"));
    assert_eq!(canonical_str, format!("{identity}"));

    let schema = ExtensionSchema {
        identity: identity.clone(),
        display_name: "Test Extension".into(),
        description: "Declarative test schema".into(),
        fields: Vec::new(),
        operands: Vec::new(),
        result_types: Vec::new(),
        side_effects: SideEffectClass::Pure,
        shape_rules: Vec::new(),
        numerical_contract: ExtensionNumericalContract::default(),
        laws: Vec::new(),
        resource_bounds: ExtensionResourceBounds::default(),
        proof_fields: proof_fields.clone(),
    };

    assert!(schema.is_host_shareable());
    assert!(schema.is_pure());
    assert!(schema.cse_eligible());
    assert!(!schema.is_divergent());
    assert!(!schema.may_alias());
    assert!(schema.terminates());
    assert_eq!(schema.target_capability(), "sm_90a");

    // Changing any single proof field strictly changes the computed digest
    let mut modified_proof = proof_fields.clone();
    modified_proof.is_pure = false;
    let modified_digest = ExtensionSchema::compute_digest(
        namespace.as_str(),
        &version,
        &[],
        &[],
        &[],
        &modified_proof,
    );
    assert_ne!(
        digest, modified_digest,
        "Digest must differ when purity changes"
    );

    let mut modified_proof2 = proof_fields.clone();
    modified_proof2.target_capability = "sm_80".into();
    let modified_digest2 = ExtensionSchema::compute_digest(
        namespace.as_str(),
        &version,
        &[],
        &[],
        &[],
        &modified_proof2,
    );
    assert_ne!(
        digest, modified_digest2,
        "Digest must differ when target capability changes"
    );
}
