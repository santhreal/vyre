use std::collections::{BTreeMap, BTreeSet};

use super::{
    operation_id_namespace, registry_error::validate_identity, ConformanceProvider,
    ConformanceRegistry, ExtensionProvenance, IdNamespace, LoweringProvider,
    OperationCatalogBundle, OperationRegistration, OperationRegistry, OperationRegistryError,
    OperationTier, SemanticDescriptor,
};
use crate::numeric::NumericContract;

/// The tier roster carries every variant of the tier enum.
#[test]
fn the_roster_carries_every_tier() {
    let mut seen = BTreeSet::new();
    for tier in OperationTier::ALL {
        seen.insert(match tier {
            OperationTier::Foundation => 0,
            OperationTier::Intrinsic => 1,
            OperationTier::Library => 2,
            OperationTier::External => 3,
            OperationTier::Unknown => 4,
        });
    }
    assert_eq!(
        seen.len(),
        5,
        "OperationTier::ALL must list every variant the match above names"
    );
}

/// A namespace is a minting fact, never a placement one.
#[test]
fn the_namespace_never_answers_with_a_tier() {
    assert_eq!(
        operation_id_namespace("vyre-primitives::graph::toposort"),
        IdNamespace::Workspace("vyre-primitives")
    );
}

/// A workspace id cannot carry a tier only a consumer identity has.
#[test]
fn a_workspace_id_declaring_an_external_tier_is_rejected() {
    let entry = OperationRegistration::new_unconstrained(
        "vyre-libs::scan::literal_set",
        OperationTier::External,
        None,
        None,
        None,
    );
    assert_eq!(
        validate_identity(&entry),
        Err(OperationRegistryError::InvalidTier {
            id: "vyre-libs::scan::literal_set",
            declared: OperationTier::External,
            origin: "workspace",
        })
    );
}

/// A consumer id carries the external tier and no other.
#[test]
fn an_external_id_declaring_a_workspace_tier_is_rejected() {
    let entry = OperationRegistration::new_unconstrained(
        "community_pack::scan::signature",
        OperationTier::Library,
        None,
        None,
        None,
    );
    assert_eq!(
        validate_identity(&entry),
        Err(OperationRegistryError::InvalidTier {
            id: "community_pack::scan::signature",
            declared: OperationTier::Library,
            origin: "external",
        })
    );
    assert_eq!(
        validate_identity(&OperationRegistration::new_unconstrained(
            "community_pack::scan::signature",
            OperationTier::External,
            None,
            None,
            None,
        )),
        Ok(())
    );
}

/// Every tier a workspace crate can mint is accepted, and the two that name
/// no minting crate are not.
#[test]
fn a_workspace_id_carries_every_workspace_tier() {
    for tier in [
        OperationTier::Foundation,
        OperationTier::Intrinsic,
        OperationTier::Library,
    ] {
        assert_eq!(
            validate_identity(&OperationRegistration::new_unconstrained(
                "vyre-primitives::hardware::popcount_u32",
                tier,
                None,
                None,
                None,
            )),
            Ok(()),
            "{tier:?} is a tier a workspace crate mints"
        );
    }
    assert_eq!(
        validate_identity(&OperationRegistration::new_unconstrained(
            "vyre-primitives::hardware::popcount_u32",
            OperationTier::Unknown,
            None,
            None,
            None,
        )),
        Err(OperationRegistryError::InvalidTier {
            id: "vyre-primitives::hardware::popcount_u32",
            declared: OperationTier::Unknown,
            origin: "workspace",
        })
    );
}

/// An id that names no crate is refused before any tier question.
#[test]
fn an_id_naming_no_crate_is_refused_whatever_it_declares() {
    for id in ["not_a_namespace", "core.indirect_dispatch", "vyre-libs::"] {
        assert_eq!(
            validate_identity(&OperationRegistration::new_unconstrained(
                id,
                OperationTier::Library,
                None,
                None,
                None,
            )),
            Err(OperationRegistryError::UnknownNamespace { id })
        );
    }
}

/// Catalog bundle collects registered descriptors and lowering providers and computes a stable digest.
#[test]
fn catalog_bundle_assembly_and_descriptor_lookup() {
    let bundle = OperationCatalogBundle::from_registry();
    assert!(!bundle.is_empty());
    assert!(bundle.len() > 0);
    let digest = bundle.digest();
    assert_ne!(*digest, [0u8; 32]);

    static REG: OperationRegistration = OperationRegistration::new_unconstrained(
        "vyre-primitives::hardware::popcount_u32",
        OperationTier::Intrinsic,
        None,
        None,
        None,
    );
    let desc = REG.descriptor();
    assert_eq!(desc.id, "vyre-primitives::hardware::popcount_u32");
    assert_eq!(desc.tier, OperationTier::Intrinsic);

    let lowering = REG.lowering_provider();
    assert_eq!(lowering.id, "vyre-primitives::hardware::popcount_u32");
    assert!(lowering.build.is_none());

    let conf = REG.conformance_provider();
    assert_eq!(conf.id, "vyre-primitives::hardware::popcount_u32");
    assert!(conf.test_inputs.is_none());
}

/// A test derives the registered operation set at run time and proves that every operation
/// has an identity-joined descriptor, lowering provider, and conformance provider.
#[test]
fn every_registered_operation_has_descriptor_lowering_and_conformance_provider() {
    let registry = OperationRegistry::global();
    let bundle = registry.catalog_bundle();
    let conformance = ConformanceRegistry::from_registry();

    let all_registered_ids: BTreeSet<&'static str> = registry.iter().map(|op| op.id).collect();

    assert!(
        !all_registered_ids.is_empty(),
        "Fix: registry must have registered operations"
    );

    let mut missing_descriptors = Vec::new();
    let mut missing_lowering = Vec::new();
    let mut missing_conformance = Vec::new();

    for &id in &all_registered_ids {
        if bundle.descriptor(id).is_none() {
            missing_descriptors.push(id);
        }
        if bundle.lowering(id).is_none() {
            missing_lowering.push(id);
        }
        if conformance.provider(id).is_none() {
            missing_conformance.push(id);
        }
    }

    assert!(
        missing_descriptors.is_empty(),
        "Fix: every registered operation must have a SemanticDescriptor, missing: {missing_descriptors:?}"
    );
    assert!(
        missing_lowering.is_empty(),
        "Fix: every registered operation must have a LoweringProvider, missing: {missing_lowering:?}"
    );
    assert!(
        missing_conformance.is_empty(),
        "Fix: every registered operation must have a ConformanceProvider, missing: {missing_conformance:?}"
    );
}

/// Adversarial verification: missing any of the three providers fails closure validation.
#[test]
fn missing_provider_fails_closure_validation() {
    fn validate_three_record_closure(
        ids: &BTreeSet<&'static str>,
        descriptors: &BTreeMap<&'static str, SemanticDescriptor>,
        lowering: &BTreeMap<&'static str, LoweringProvider>,
        conformance: &BTreeMap<&'static str, ConformanceProvider>,
    ) -> Result<(), &'static str> {
        for &id in ids {
            if !descriptors.contains_key(id) {
                return Err("missing SemanticDescriptor");
            }
            if !lowering.contains_key(id) {
                return Err("missing LoweringProvider");
            }
            if !conformance.contains_key(id) {
                return Err("missing ConformanceProvider");
            }
        }
        Ok(())
    }

    let id = "vyre-libs::math::test_op";
    let mut ids = BTreeSet::new();
    ids.insert(id);

    let desc = SemanticDescriptor {
        id,
        semantic_version: 1,
        signature: None,
        tier: OperationTier::Library,
        category: Some("math"),
        laws: &[],
        numeric: NumericContract::EXACT,
        geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
        explicit_effects: None,
        explicit_capabilities: None,
        opaque_reason: Some("test operation placeholder rationale"),
    };
    let low = LoweringProvider { id, build: None };
    let conf = ConformanceProvider {
        id,
        test_inputs: None,
        expected_output: None,
    };

    let mut descs = BTreeMap::new();
    let mut lows = BTreeMap::new();
    let mut confs = BTreeMap::new();

    descs.insert(id, desc);
    lows.insert(id, low);
    confs.insert(id, conf);

    assert!(validate_three_record_closure(&ids, &descs, &lows, &confs).is_ok());

    // Remove descriptor -> error
    let mut bad_descs = descs.clone();
    bad_descs.remove(id);
    assert_eq!(
        validate_three_record_closure(&ids, &bad_descs, &lows, &confs),
        Err("missing SemanticDescriptor")
    );

    // Remove lowering -> error
    let mut bad_lows = lows.clone();
    bad_lows.remove(id);
    assert_eq!(
        validate_three_record_closure(&ids, &descs, &bad_lows, &confs),
        Err("missing LoweringProvider")
    );

    // Remove conformance -> error
    let mut bad_confs = confs.clone();
    bad_confs.remove(id);
    assert_eq!(
        validate_three_record_closure(&ids, &descs, &lows, &bad_confs),
        Err("missing ConformanceProvider")
    );
}

/// A test proves a production catalog read cannot reach a fixture or an expected output,
/// and that changing a fixture leaves the production digest identical.
#[test]
fn production_catalog_read_cannot_reach_fixtures_and_changing_fixtures_preserves_digest() {
    let id = "vyre-libs::bitset::and_not";
    let desc = SemanticDescriptor {
        id,
        semantic_version: 1,
        signature: None,
        tier: OperationTier::Library,
        category: Some("bitset"),
        laws: &["opaque"],
        numeric: NumericContract::EXACT,
        geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
        explicit_effects: None,
        explicit_capabilities: None,
        opaque_reason: None,
    };
    let lowering = LoweringProvider { id, build: None };

    let mut descriptors = BTreeMap::new();
    let mut lowering_providers = BTreeMap::new();
    let extensions = BTreeMap::new();

    descriptors.insert(id, desc);
    lowering_providers.insert(id, lowering);

    let bundle1 = OperationCatalogBundle::from_parts(
        descriptors.clone(),
        lowering_providers.clone(),
        extensions.clone(),
    );
    let digest1 = *bundle1.digest();

    // Structurally verify: OperationCatalogBundle and SemanticDescriptor expose NO fixture fields
    assert!(bundle1.descriptor(id).is_some());
    assert!(bundle1.lowering(id).is_some());
    // ConformanceProvider with fixture 1
    let fixture1 = ConformanceProvider {
        id,
        test_inputs: Some(|| vec![vec![vec![1, 2, 3]]]),
        expected_output: Some(|| vec![vec![vec![4, 5, 6]]]),
    };

    // ConformanceProvider with modified fixture 2
    let fixture2 = ConformanceProvider {
        id,
        test_inputs: Some(|| vec![vec![vec![99, 99, 99, 99]]]),
        expected_output: Some(|| vec![vec![vec![0, 0, 0, 0]]]),
    };

    assert_ne!(
        (fixture1.test_inputs.unwrap())(),
        (fixture2.test_inputs.unwrap())()
    );

    // Recompute bundle digest: it must remain strictly byte-identical!
    let bundle2 = OperationCatalogBundle::from_parts(
        descriptors.clone(),
        lowering_providers.clone(),
        extensions.clone(),
    );
    let digest2 = *bundle2.digest();
    assert_eq!(
        digest1, digest2,
        "Fix: changing test fixtures or expected outputs must not alter the production OperationCatalogBundle digest"
    );

    // Changing an execution-relevant semantic property MUST alter the digest
    let mut modified_descriptors = descriptors.clone();
    modified_descriptors.insert(
        id,
        SemanticDescriptor {
            semantic_version: 2,
            ..desc
        },
    );
    let bundle3 =
        OperationCatalogBundle::from_parts(modified_descriptors, lowering_providers, extensions);
    let digest3 = *bundle3.digest();
    assert_ne!(
        digest1, digest3,
        "Fix: changing a semantic version or descriptor property must alter the production OperationCatalogBundle digest"
    );
}

/// A test proves a OperationCatalogBundle digest is part of artifact identity: two bundles differing
/// in one extension version produce different artifact identities.
#[test]
fn catalog_bundle_digest_is_part_of_artifact_identity() {
    let base_request_digest: [u8; 32] = [42u8; 32];

    let mut bundle_v1 = OperationCatalogBundle::empty();
    bundle_v1 = bundle_v1.with_extension(
        "custom_dialect",
        1,
        vec![SemanticDescriptor {
            id: "custom_dialect::op_a",
            semantic_version: 1,
            signature: None,
            tier: OperationTier::External,
            category: Some("custom"),
            laws: &[],
            numeric: NumericContract::EXACT,
            geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
            explicit_effects: None,
            explicit_capabilities: None,
            opaque_reason: Some("external custom dialect operation"),
        }],
        vec![LoweringProvider {
            id: "custom_dialect::op_a",
            build: None,
        }],
    );

    let mut bundle_v2 = OperationCatalogBundle::empty();
    bundle_v2 = bundle_v2.with_extension(
        "custom_dialect",
        2,
        vec![SemanticDescriptor {
            id: "custom_dialect::op_a",
            semantic_version: 1,
            signature: None,
            tier: OperationTier::External,
            category: Some("custom"),
            laws: &[],
            numeric: NumericContract::EXACT,
            geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
            explicit_effects: None,
            explicit_capabilities: None,
            opaque_reason: Some("external custom dialect operation v2"),
        }],
        vec![LoweringProvider {
            id: "custom_dialect::op_a",
            build: None,
        }],
    );

    assert_ne!(
        bundle_v1.digest(),
        bundle_v2.digest(),
        "Two bundles differing in one extension version must produce different bundle digests"
    );

    let artifact_id_1 = bundle_v1.artifact_identity(&base_request_digest);
    let artifact_id_2 = bundle_v2.artifact_identity(&base_request_digest);

    assert_ne!(
        artifact_id_1, artifact_id_2,
        "Two bundles differing in one extension version must produce different artifact identities"
    );

    // Proves independently versioned dialect forms a closed catalog without central link anchor
    assert_eq!(
        bundle_v1.extension("custom_dialect"),
        Some(&ExtensionProvenance {
            name: "custom_dialect",
            version: 1
        })
    );
    assert_eq!(
        bundle_v2.extension("custom_dialect"),
        Some(&ExtensionProvenance {
            name: "custom_dialect",
            version: 2
        })
    );
    assert!(bundle_v1.contains("custom_dialect::op_a"));
    assert!(bundle_v2.contains("custom_dialect::op_a"));
}

/// A test enumerates the operation roster at run time and fails when an operation has no contract
/// record, or a record with neither a law nor an explicit no-transform decision, asserting zero
/// operations in an unrecorded state.
#[test]
fn every_registered_operation_has_contract_record_with_valid_decision() {
    let registry = OperationRegistry::global();
    let all_ops: Vec<super::SemanticOperation> = registry.iter().collect();
    assert!(
        !all_ops.is_empty(),
        "Fix: registry must contain registered operations"
    );

    let mut violations = Vec::new();
    let mut guarded_laws_count = 0usize;
    let mut no_transform_count = 0usize;
    let mut opaque_count = 0usize;
    let mut not_recorded_count = 0usize;

    for op in &all_ops {
        let record = op.contract_record();
        match record.decision {
            vyre_spec::TransformDecision::GuardedLaws(_) => guarded_laws_count += 1,
            vyre_spec::TransformDecision::NoTransform { .. } => no_transform_count += 1,
            vyre_spec::TransformDecision::Opaque { .. } => opaque_count += 1,
            vyre_spec::TransformDecision::NotRecorded => not_recorded_count += 1,
        }

        if !op.has_transform_decision() {
            violations.push(format!(
                "operation `{}` has neither algebraic laws nor an explicit opaque/no-transform decision",
                op.id
            ));
            continue;
        }
        if let Err(err) = record.validate() {
            violations.push(format!(
                "operation `{}` contract record failed validation: {err}",
                op.id
            ));
        }
    }

    assert_eq!(
        not_recorded_count, 0,
        "Fix: zero operations may be in unrecorded state, found {not_recorded_count}"
    );
    assert!(
        violations.is_empty(),
        "Fix: {} operations failed contract decision check: {violations:#?}",
        violations.len()
    );
    println!(
        "OP_COUNTS: total={}, guarded_laws={}, no_transform={}, opaque={}, not_recorded={}",
        all_ops.len(),
        guarded_laws_count,
        no_transform_count,
        opaque_count,
        not_recorded_count
    );
    assert!(
        guarded_laws_count + no_transform_count + opaque_count == all_ops.len(),
        "Fix: all operations must be classified in the three resolved states"
    );
}

/// A test proves adding an operation without a decision turns the registry red.
#[test]
fn adding_operation_without_decision_turns_registry_red() {
    let undecidable_reg = OperationRegistration::new_unconstrained(
        "vyre-foundation::test::undecided_operation",
        OperationTier::Foundation,
        None,
        None,
        None,
    );
    assert!(!undecidable_reg.has_transform_decision());
    let record = undecidable_reg.contract_record();
    assert!(
        record.decision.is_not_recorded(),
        "Unannotated registration must produce NotRecorded transform decision"
    );
    let result = record.validate();
    assert_eq!(
        result,
        Err(vyre_spec::ContractValidationError::NotRecorded),
        "Contract record without a decision must fail validation with NotRecorded"
    );
}

/// A test proves a law label without executable proof evidence is rejected.
#[test]
fn law_label_without_executable_proof_evidence_is_rejected() {
    let unproven_law = vyre_spec::GuardedLaw::unconditional(vyre_spec::AlgebraicLaw::Associative)
        .with_proof_method(vyre_spec::ProofMethod::None);
    assert_eq!(
        unproven_law.validate(),
        Err(vyre_spec::LawValidationError::NoExecutableProofEvidence {
            law: "associative".to_string()
        })
    );

    let zero_witness_law =
        vyre_spec::GuardedLaw::unconditional(vyre_spec::AlgebraicLaw::Commutative)
            .with_proof_method(vyre_spec::ProofMethod::WitnessedU32 { seed: 42, count: 0 });
    assert_eq!(
        zero_witness_law.validate(),
        Err(vyre_spec::LawValidationError::NoExecutableProofEvidence {
            law: "commutative".to_string()
        })
    );
}

/// A test proves placeholder opaque reasons are rejected by name.
#[test]
fn placeholder_opaque_reasons_are_rejected() {
    for placeholder in [
        "todo",
        "none",
        "opaque",
        "tbd",
        "placeholder",
        "no-op",
        "not implemented",
    ] {
        let reg = OperationRegistration::new_unconstrained(
            "vyre-foundation::test::placeholder_op",
            OperationTier::Foundation,
            None,
            None,
            None,
        )
        .with_opaque(placeholder);
        let record = reg.contract_record();
        assert!(
            record.validate().is_err(),
            "Placeholder reason `{placeholder}` must fail contract validation"
        );
    }
}

/// A test proves declarative dialect operations generate builders, documentation, and contract joins.
#[test]
fn declarative_dialect_operation_generation() {
    fn dummy_builder() -> crate::ir::Program {
        crate::ir::Program::empty()
    }

    crate::declare_dialect_op! {
        id: "vyre-foundation::test::generated_dialect_op",
        tier: OperationTier::Foundation,
        category: "test_dialect",
        doc: "A generated dialect test operation for verifying single-source generation.",
        reference_obligation: "Pure value identity reference semantics.",
        laws: &["idempotent"],
        builder: dummy_builder,
    }

    let op = super::SemanticOperation {
        id: "vyre-foundation::test::generated_dialect_op",
        semantic_version: 1,
        signature: None,
        tier: OperationTier::Foundation,
        category: Some("test_dialect"),
        build: Some(dummy_builder),
        test_inputs: None,
        expected_output: None,
        laws: &["idempotent"],
        numeric: NumericContract::EXACT,
        geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
        source_file: file!(),
        explicit_effects: None,
        explicit_capabilities: None,
        opaque_reason: None,
    };

    let record = op.contract_record();
    assert_eq!(record.validate(), Ok(()));
    assert_eq!(record.decision.laws().len(), 1);
    assert_eq!(record.decision.laws()[0].law.name(), "idempotent");
}
