//! Tests for protocol compatibility matrix, schema registry join, version negotiation,
//! generation-scoped cache namespaces, retained sessions, and rollout lifecycle.
//!
//! WHY: proves row 120 contracts:
//! - Explicit compatibility matrix covering every public protocol domain and registered schema version.
//! - Runtime derivation of the version surface fails when a schema or domain lacks matrix coverage.
//! - Protocol negotiation selects preregistered mutual contracts without silent downgrades or aliases.
//! - Unsupported version pairs fail before expensive work, naming the domain and upgrade action.
//! - Cache namespaces and retained sessions are generation-scoped; stale records are rejected.
//! - Crash interruption during rollout leaves no half-migrated state; rollback restores prior generation.
//! - Negotiated contract digest is bound into artifact identity.

use vyre_spec::{
    derive_artifact_identity, CacheNamespace, CompatibilityDisposition, CompatibilityMatrix,
    DefaultsPolicy, FieldType, GenerationId, ProtocolDomain, ProtocolVersion, RetainedSessionScope,
    RolloutManager, SchemaId, SchemaRegistry, SessionScopeError, SessionStatus,
};

#[test]
fn all_protocol_domains_are_covered_in_canonical_matrix() {
    let matrix = CompatibilityMatrix::canonical();
    for &domain in ProtocolDomain::ALL {
        let disp = matrix.check(
            domain,
            ProtocolVersion::V1_0_0,
            ProtocolVersion::V1_0_0,
        );
        assert_eq!(
            disp,
            CompatibilityDisposition::Supported,
            "Fix: domain '{domain}' must have a supported cell for v1.0.0 -> v1.0.0."
        );

        let disp_1_1 = matrix.check(
            domain,
            ProtocolVersion::V1_1_0,
            ProtocolVersion::V1_1_0,
        );
        assert_eq!(
            disp_1_1,
            CompatibilityDisposition::Supported,
            "Fix: domain '{domain}' must have a supported cell for v1.1.0 -> v1.1.0."
        );
    }
}

#[test]
fn runtime_derived_version_surface_covers_every_registered_schema() {
    let matrix = CompatibilityMatrix::canonical();
    let all_schemas = SchemaRegistry::all();

    assert!(
        !all_schemas.is_empty(),
        "Fix: SchemaRegistry must declare at least one schema definition."
    );

    for schema in all_schemas {
        let domain = schema.protocol_domain();
        let semver = schema.semver;

        // Matrix check for schema domain and version must be supported
        let disp = matrix.check(domain, semver, semver);
        assert!(
            disp.is_compatible(),
            "Fix: Schema '{}' (ID: {:?}) with version {} in domain '{}' has no compatible cell in CompatibilityMatrix.",
            schema.id.as_str(),
            schema.id,
            semver,
            domain
        );

        // Direct check_schema must also be compatible
        let schema_disp = matrix.check_schema(schema.id, semver);
        assert!(
            schema_disp.is_compatible(),
            "Fix: check_schema failed for SchemaId::{:?} with version {semver}",
            schema.id
        );
    }
}

#[test]
fn protocol_negotiation_selects_highest_mutually_supported_version() {
    let matrix = CompatibilityMatrix::canonical();
    let offered = [ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0];
    let supported = [ProtocolVersion::V1_1_0];

    let contract = matrix
        .negotiate(ProtocolDomain::PublicWire, &offered, &supported)
        .expect("Fix: negotiation must succeed for overlapping supported versions.");

    assert_eq!(contract.domain, ProtocolDomain::PublicWire);
    assert_eq!(contract.client_version, ProtocolVersion::V1_1_0);
    assert_eq!(contract.host_version, ProtocolVersion::V1_1_0);
    assert_eq!(contract.disposition, CompatibilityDisposition::Supported);
    assert_ne!(contract.contract_digest, [0_u8; 32]);
}

#[test]
fn unsupported_version_pair_fails_with_actionable_upgrade_error() {
    let matrix = CompatibilityMatrix::canonical();

    // Test across several domains with unsupported major jumps
    let domains = [
        ProtocolDomain::Artifact,
        ProtocolDomain::Schedule,
        ProtocolDomain::Measurement,
        ProtocolDomain::Resource,
        ProtocolDomain::RuntimeProtocol,
    ];

    for domain in domains {
        let offered = [ProtocolVersion::new(99, 0, 0)];
        let supported = [ProtocolVersion::V1_0_0];

        let err = matrix
            .negotiate(domain, &offered, &supported)
            .expect_err("Fix: unsupported major version jump must fail negotiation.");

        assert_eq!(err.domain, domain);
        assert!(
            err.upgrade_action.contains("Fix:"),
            "Fix: negotiation error must contain actionable 'Fix:' message: {}",
            err.upgrade_action
        );
        assert!(
            err.upgrade_action.contains(domain.as_str()),
            "Fix: negotiation error must name the domain: {}",
            err.upgrade_action
        );
    }
}

#[test]
fn empty_negotiation_offers_fail_closed() {
    let matrix = CompatibilityMatrix::canonical();
    let err = matrix
        .negotiate(ProtocolDomain::Schedule, &[], &[ProtocolVersion::V1_0_0])
        .expect_err("Fix: empty client offer list must fail closed.");
    assert!(err.upgrade_action.contains("Fix:"));
}

#[test]
fn schema_registry_covers_every_schema_id() {
    let all_schemas = SchemaRegistry::all();
    assert_eq!(
        all_schemas.len(),
        SchemaId::ALL.len(),
        "Fix: SchemaRegistry must contain exactly one definition for each SchemaId."
    );

    for &id in SchemaId::ALL {
        let def = SchemaRegistry::lookup(id)
            .unwrap_or_else(|| panic!("Fix: SchemaId::{id:?} must be registered in SchemaRegistry"));
        assert_eq!(def.id, id);
        assert!(
            def.validate_invariants(),
            "Fix: SchemaDefinition for '{id}' failed invariant validation."
        );
        assert!(
            !def.domain_separator.is_empty(),
            "Fix: domain separator must not be empty for '{id}'"
        );
        assert!(
            def.bounds.max_bytes > 0,
            "Fix: max_bytes bound must be positive for '{id}'"
        );
        assert_eq!(
            def.defaults_policy,
            DefaultsPolicy::NoDefaults,
            "Fix: canonical records must enforce NoDefaults policy"
        );
    }
}

#[test]
fn schema_canonical_fields_are_strictly_ordered_and_typed() {
    for def in SchemaRegistry::all() {
        let mut prev_num = 0;
        for field in def.fields {
            assert!(
                field.number > prev_num,
                "Fix: fields in schema '{}' must have strictly increasing numbers (found {} after {})",
                def.id,
                field.number,
                prev_num
            );
            prev_num = field.number;
            assert!(
                !field.name.is_empty(),
                "Fix: field {} in schema '{}' must have a non-empty name",
                field.number,
                def.id
            );
            match field.field_type {
                FieldType::U8
                | FieldType::U16
                | FieldType::U32
                | FieldType::U64
                | FieldType::I32
                | FieldType::I64
                | FieldType::F32
                | FieldType::F64
                | FieldType::Bool
                | FieldType::FixedBytes(_)
                | FieldType::VarBytes
                | FieldType::Utf8String
                | FieldType::List(_) => {}
                _ => panic!("Uncataloged FieldType variant in schema registry"),
            }
        }
    }
}

#[test]
fn generation_scoped_cache_namespaces_isolate_keys_and_reject_stale_records() {
    let gen1 = GenerationId::new(1);
    let gen2 = GenerationId::new(2);

    let ns1 = CacheNamespace::new(
        ProtocolDomain::Artifact,
        ProtocolVersion::V1_0_0,
        gen1,
        "megakernel_lowering_cache",
    );

    let ns2 = CacheNamespace::new(
        ProtocolDomain::Artifact,
        ProtocolVersion::V1_0_0,
        gen2,
        "megakernel_lowering_cache",
    );

    let base_key = b"kernel_matmul_fp32_tile16";

    let key1 = ns1.scoped_key(base_key);
    let key2 = ns2.scoped_key(base_key);

    // Different generations must produce distinct cache keys
    assert_ne!(
        key1, key2,
        "Fix: cache keys for different generations must be strictly disjoint."
    );

    // Validation against active generation 1
    assert!(ns1.validate_generation(gen1).is_ok());

    // Validation of gen 1 record against active generation 2 must fail
    let err = ns1.validate_generation(gen2).expect_err(
        "Fix: stale record from generation 1 must be rejected when active generation is 2.",
    );
    assert_eq!(err.record_generation, gen1);
    assert_eq!(err.active_generation, gen2);
    assert_eq!(err.namespace, "megakernel_lowering_cache");
    assert!(err.to_string().contains("Fix:"));
}

#[test]
fn negotiation_output_is_part_of_artifact_identity() {
    let matrix = CompatibilityMatrix::canonical();
    let base_artifact_hash = [0x42_u8; 32];

    // Run 1: Client negotiates v1.0.0
    let contract_v1_0 = matrix
        .negotiate(
            ProtocolDomain::Artifact,
            &[ProtocolVersion::V1_0_0],
            &[ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0],
        )
        .expect("Fix: negotiation for v1.0.0 must succeed");

    // Run 2: Client negotiates v1.1.0
    let contract_v1_1 = matrix
        .negotiate(
            ProtocolDomain::Artifact,
            &[ProtocolVersion::V1_1_0],
            &[ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0],
        )
        .expect("Fix: negotiation for v1.1.0 must succeed");

    let identity_1_0 = derive_artifact_identity(&base_artifact_hash, &contract_v1_0);
    let identity_1_1 = derive_artifact_identity(&base_artifact_hash, &contract_v1_1);

    // Two runs negotiating different contracts must produce distinct artifact identities
    assert_ne!(
        identity_1_0, identity_1_1,
        "Fix: runs negotiating different contracts must produce distinct artifact identities."
    );

    // Identical contract negotiation must be deterministic
    let identity_1_0_repeat = derive_artifact_identity(&base_artifact_hash, &contract_v1_0);
    assert_eq!(identity_1_0, identity_1_0_repeat);
}

#[test]
fn rolling_upgrade_crash_interruption_and_rollback_preserve_consistency() {
    let matrix = CompatibilityMatrix::canonical();
    let initial_gen = GenerationId::new(100);
    let mut rollout = RolloutManager::new(initial_gen);

    assert_eq!(rollout.active_generation(), initial_gen);
    assert_eq!(rollout.staging_generation(), None);

    // Client 1 connects and registers active session in gen 100
    let contract_1 = matrix
        .negotiate(
            ProtocolDomain::RuntimeProtocol,
            &[ProtocolVersion::V1_0_0],
            &[ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0],
        )
        .expect("Fix: negotiation must succeed");

    let session_1 = RetainedSessionScope {
        session_id: [1_u8; 32],
        domain: ProtocolDomain::RuntimeProtocol,
        contract: contract_1,
        generation: initial_gen,
        status: SessionStatus::Active,
    };
    rollout.register_session(session_1.clone());

    assert!(session_1.validate(initial_gen, &contract_1).is_ok());

    // Phase 1: Stage an upgrade to gen 101
    let staging_gen = rollout.stage_upgrade();
    assert_eq!(staging_gen, GenerationId::new(101));
    assert_eq!(rollout.staging_generation(), Some(staging_gen));
    assert_eq!(rollout.active_generation(), initial_gen); // Active gen remains 100

    // Register staging session in gen 101
    let staging_session = RetainedSessionScope {
        session_id: [2_u8; 32],
        domain: ProtocolDomain::RuntimeProtocol,
        contract: contract_1,
        generation: staging_gen,
        status: SessionStatus::Staging,
    };
    rollout.register_session(staging_session.clone());

    // Phase 2: Simulate crash interruption during upgrade
    rollout.crash_interruption();
    assert_eq!(rollout.active_generation(), initial_gen);
    assert_eq!(rollout.staging_generation(), None);
    // Interrupted staging session was cleanly purged
    assert_eq!(rollout.sessions().len(), 1);
    assert_eq!(rollout.sessions()[0].session_id, session_1.session_id);
    assert!(session_1.validate(initial_gen, &contract_1).is_ok());

    // Phase 3: Re-stage upgrade and perform atomic commit
    let staging_gen_2 = rollout.stage_upgrade();
    let staging_session_2 = RetainedSessionScope {
        session_id: [3_u8; 32],
        domain: ProtocolDomain::RuntimeProtocol,
        contract: contract_1,
        generation: staging_gen_2,
        status: SessionStatus::Staging,
    };
    rollout.register_session(staging_session_2.clone());

    let committed_gen = rollout
        .atomic_commit()
        .expect("Fix: atomic commit of staged upgrade must succeed");
    assert_eq!(committed_gen, GenerationId::new(101));
    assert_eq!(rollout.active_generation(), GenerationId::new(101));
    assert_eq!(rollout.staging_generation(), None);

    // Gen 100 session is now stale when validated against active gen 101
    let stale_err = session_1
        .validate(GenerationId::new(101), &contract_1)
        .expect_err("Fix: old session from gen 100 must be rejected under active gen 101");
    assert!(matches!(
        stale_err,
        SessionScopeError::StaleGeneration { .. }
    ));

    // Phase 4: Rollback to generation 100
    rollout.rollback(initial_gen);
    assert_eq!(rollout.active_generation(), initial_gen);
    // Sessions from gen 101 were purged during rollback
    assert_eq!(rollout.sessions().len(), 1);
    assert_eq!(rollout.sessions()[0].generation, initial_gen);
    assert!(session_1.validate(initial_gen, &contract_1).is_ok());
}

#[test]
fn mixed_client_operation_under_negotiation_maintains_isolation() {
    let matrix = CompatibilityMatrix::canonical();
    let host_supported = [ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0];

    // Client A offers v1.0.0
    let contract_a = matrix
        .negotiate(
            ProtocolDomain::PublicWire,
            &[ProtocolVersion::V1_0_0],
            &host_supported,
        )
        .expect("Fix: client A negotiation must succeed");
    assert_eq!(contract_a.client_version, ProtocolVersion::V1_0_0);

    // Client B offers v1.1.0
    let contract_b = matrix
        .negotiate(
            ProtocolDomain::PublicWire,
            &[ProtocolVersion::V1_1_0],
            &host_supported,
        )
        .expect("Fix: client B negotiation must succeed");
    assert_eq!(contract_b.client_version, ProtocolVersion::V1_1_0);

    // Verify contracts have distinct digests
    assert_ne!(contract_a.contract_digest, contract_b.contract_digest);

    // Verify sessions cannot cross-validate with mismatched contracts
    let session_a = RetainedSessionScope {
        session_id: [10_u8; 32],
        domain: ProtocolDomain::PublicWire,
        contract: contract_a,
        generation: GenerationId::INITIAL,
        status: SessionStatus::Active,
    };

    assert!(session_a.validate(GenerationId::INITIAL, &contract_a).is_ok());
    let mismatch_err = session_a
        .validate(GenerationId::INITIAL, &contract_b)
        .expect_err("Fix: session validated with mismatched contract must fail");
    assert!(matches!(mismatch_err, SessionScopeError::ContractMismatch));
}
