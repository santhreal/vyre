//! Contracts the `application-runnable` gate is written against.
//!
//! Two classes are closed here rather than sampled. The receipt's canonical
//! identity fields are enumerated from the serialized document at run time, so
//! a digest field added to the schema and left out of validation turns this
//! suite red. The product-family vocabulary is enumerated from
//! [`PROHIBITED_DOMAIN_TERMS`], so a term added to the roster is proven refused
//! without anyone listing it twice.
//!
//! The route contract is proven against a route the compiler really produced:
//! every registered generic frontend dialect is staged, compiled and lowered,
//! and the record that comes back is the subject of the assertions. A record
//! built by hand would prove the validator and not the compiler.

use super::*;

const SECRET: &[u8] = b"application-runnable-test-secret";

/// A receipt whose every field states a fact the validator admits.
fn neutral_receipt() -> ApplicationEvidenceReceipt {
    let mut receipt = ApplicationEvidenceReceipt {
        receipt_version: APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        schema_digest: [0x01; 32],
        graph_digest: [0x02; 32],
        resource_digest: [0x03; 32],
        artifact_digest: [0x04; 32],
        payload_digest: [0x05; 32],
        target_format: "primary-binary".to_string(),
        execution_passed: true,
        output_digest: [0x06; 32],
        benchmark_metrics: BenchmarkEvidenceMetrics {
            compile_time_ns: 4_100_000,
            load_time_ns: 900_000,
            p50_latency_ns: 61_000,
            p99_latency_ns: 88_000,
            throughput_items_per_sec: 16_384.0,
            peak_resident_bytes: 33_554_432,
            schedule_features: vec![
                "topology:sequential".to_string(),
                "groups:2".to_string(),
                "fused".to_string(),
            ],
        },
        auth_tag: String::new(),
    };
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    receipt
}

/// The receipt as a JSON document, for the run-time field enumeration.
fn as_document(receipt: &ApplicationEvidenceReceipt) -> serde_json::Value {
    serde_json::to_value(receipt).expect("the receipt serializes")
}

#[test]
fn a_complete_authenticated_neutral_receipt_is_admitted() {
    assert_eq!(
        validate_evidence_receipt(&neutral_receipt(), SECRET),
        Ok(())
    );
}

/// Every canonical identity field the schema carries is refused when it is zero.
///
/// The roster is the receipt's own serialized document, so a digest field added
/// to the schema is judged by the field it becomes. A field the validator does
/// not read fails here with the name it was given.
#[test]
fn every_canonical_digest_field_is_refused_when_it_is_zero() {
    let receipt = neutral_receipt();
    let document = as_document(&receipt);
    let fields: Vec<String> = document
        .as_object()
        .expect("the receipt serializes as an object")
        .keys()
        .filter(|name| name.ends_with("_digest"))
        .cloned()
        .collect();
    assert!(
        fields.len() >= 6,
        "the receipt states fewer canonical identities than the schema requires: {fields:?}"
    );

    for field in fields {
        let mut zeroed = document.clone();
        zeroed[&field] = serde_json::Value::Array(vec![serde_json::json!(0); 32]);
        let zeroed: ApplicationEvidenceReceipt =
            serde_json::from_value(zeroed).expect("zeroing a digest keeps the receipt decodable");
        let error = validate_evidence_receipt(&zeroed, SECRET)
            .expect_err("a zero canonical identity is refused");
        assert_eq!(
            error,
            ReceiptValidationError::ZeroDigest {
                field: match field.as_str() {
                    "schema_digest" => "schema_digest",
                    "graph_digest" => "graph_digest",
                    "resource_digest" => "resource_digest",
                    "artifact_digest" => "artifact_digest",
                    "payload_digest" => "payload_digest",
                    "output_digest" => "output_digest",
                    other => panic!(
                        "the schema carries canonical identity `{other}` that validation does not read"
                    ),
                }
            },
            "zeroing `{field}` was not refused as a zero canonical identity"
        );
    }
}

#[test]
fn an_unsupported_schema_version_is_refused() {
    let mut receipt = neutral_receipt();
    receipt.receipt_version = APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION + 1;
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    assert_eq!(
        validate_evidence_receipt(&receipt, SECRET),
        Err(ReceiptValidationError::UnsupportedVersion {
            found: APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION + 1,
            expected: APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        })
    );
}

#[test]
fn an_empty_target_format_is_refused() {
    let mut receipt = neutral_receipt();
    receipt.target_format = "   ".to_string();
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    assert_eq!(
        validate_evidence_receipt(&receipt, SECRET),
        Err(ReceiptValidationError::EmptyTargetFormat)
    );
}

#[test]
fn a_receipt_recording_a_failed_execution_is_refused() {
    let mut receipt = neutral_receipt();
    receipt.execution_passed = false;
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    assert_eq!(
        validate_evidence_receipt(&receipt, SECRET),
        Err(ReceiptValidationError::ExecutionFailed)
    );
}

#[test]
fn a_zero_or_inverted_latency_is_refused() {
    for (p50, p99) in [(0, 88_000), (61_000, 0), (91_000, 88_000)] {
        let mut receipt = neutral_receipt();
        receipt.benchmark_metrics.p50_latency_ns = p50;
        receipt.benchmark_metrics.p99_latency_ns = p99;
        receipt.auth_tag = receipt.compute_auth_tag(SECRET);
        assert_eq!(
            validate_evidence_receipt(&receipt, SECRET),
            Err(ReceiptValidationError::InvalidLatency { p50, p99 }),
            "p50={p50} p99={p99} was not refused"
        );
    }
}

#[test]
fn a_receipt_naming_no_schedule_feature_is_refused() {
    let mut receipt = neutral_receipt();
    receipt.benchmark_metrics.schedule_features.clear();
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    assert_eq!(
        validate_evidence_receipt(&receipt, SECRET),
        Err(ReceiptValidationError::EmptyScheduleFeatures)
    );
}

#[test]
fn an_unauthenticated_or_tampered_receipt_is_refused() {
    let mut missing = neutral_receipt();
    missing.auth_tag = String::new();
    assert_eq!(
        validate_evidence_receipt(&missing, SECRET),
        Err(ReceiptValidationError::UnauthenticatedReceipt)
    );

    let mut wrong_secret = neutral_receipt();
    wrong_secret.auth_tag = wrong_secret.compute_auth_tag(b"another-secret");
    assert_eq!(
        validate_evidence_receipt(&wrong_secret, SECRET),
        Err(ReceiptValidationError::UnauthenticatedReceipt)
    );

    let mut tampered = neutral_receipt();
    tampered.benchmark_metrics.p50_latency_ns += 1;
    assert_eq!(
        validate_evidence_receipt(&tampered, SECRET),
        Err(ReceiptValidationError::UnauthenticatedReceipt),
        "a metric edited after signing must not authenticate"
    );
}

/// Every field name and every string value the schema carries is domain-neutral.
///
/// The scan is the same walk validation performs, so the schema cannot admit a
/// field whose own name is a product family.
#[test]
fn the_receipt_schema_carries_no_product_family_name() {
    let document = as_document(&neutral_receipt());
    let found = prohibited_names_in(&document);
    assert!(
        found.is_empty(),
        "the receipt schema carries product-family names: {found:?}"
    );
}

/// Every term in the prohibited vocabulary is refused when a value carries it.
#[test]
fn every_prohibited_domain_term_is_refused_in_a_receipt_value() {
    assert!(!PROHIBITED_DOMAIN_TERMS.is_empty());
    for term in PROHIBITED_DOMAIN_TERMS {
        let mut receipt = neutral_receipt();
        receipt.target_format = format!("primary-binary-{term}");
        receipt.auth_tag = receipt.compute_auth_tag(SECRET);
        let error = validate_evidence_receipt(&receipt, SECRET)
            .expect_err("a product-family name is refused");
        assert_eq!(
            error,
            ReceiptValidationError::ProhibitedDomainName {
                field: "/target_format".to_string(),
                value: format!("primary-binary-{term}"),
                term,
            },
            "`{term}` was not refused"
        );
    }
}

/// Every term in the prohibited vocabulary is refused when a nested value carries it.
#[test]
fn a_prohibited_domain_term_nested_in_a_metric_value_is_refused() {
    let mut receipt = neutral_receipt();
    receipt
        .benchmark_metrics
        .schedule_features
        .push("resnet-stage".to_string());
    receipt.auth_tag = receipt.compute_auth_tag(SECRET);
    assert_eq!(
        validate_evidence_receipt(&receipt, SECRET),
        Err(ReceiptValidationError::ProhibitedDomainName {
            field: "/benchmark_metrics/schedule_features/3".to_string(),
            value: "resnet-stage".to_string(),
            term: "resnet",
        })
    );
}

/// Every registered execution topology has its own tag.
///
/// `variant_index` is exhaustive with no catch-all, so a topology variant added
/// to the compiler stops this suite compiling until a sample and a tag are
/// stated for it.
#[test]
fn every_execution_topology_variant_has_its_own_tag() {
    fn variant_index(topology: ExecutionTopology) -> usize {
        match topology {
            ExecutionTopology::Sequential => 0,
            ExecutionTopology::ConcurrentQueue { .. } => 1,
            ExecutionTopology::ResidentPartition { .. } => 2,
        }
    }
    fn mode_index(mode: ResidentPartitionMode) -> usize {
        match mode {
            ResidentPartitionMode::FixedSpatialMask => 0,
            ResidentPartitionMode::BoundedWorkQueue => 1,
        }
    }

    let modes = [
        ResidentPartitionMode::FixedSpatialMask,
        ResidentPartitionMode::BoundedWorkQueue,
    ];
    let mut samples = vec![
        ExecutionTopology::Sequential,
        ExecutionTopology::ConcurrentQueue { queues: 4 },
    ];
    samples.extend(
        modes
            .iter()
            .map(|mode| ExecutionTopology::ResidentPartition {
                partitions: 8,
                mode: *mode,
            }),
    );

    let covered_topologies: BTreeSet<usize> = samples.iter().copied().map(variant_index).collect();
    assert_eq!(
        covered_topologies,
        BTreeSet::from([0, 1, 2]),
        "the samples do not cover every execution topology variant"
    );
    let covered_modes: BTreeSet<usize> = modes.iter().copied().map(mode_index).collect();
    assert_eq!(
        covered_modes,
        BTreeSet::from([0, 1]),
        "the samples do not cover every resident partition mode"
    );

    let tags: BTreeSet<String> = samples.iter().copied().map(topology_tag).collect();
    assert_eq!(
        tags.len(),
        samples.len(),
        "two topologies share one tag: {tags:?}"
    );
}

#[test]
fn schedule_feature_coverage_refuses_a_disagreement_in_either_direction() {
    let artifact = vec!["topology:sequential".to_string(), "fused".to_string()];
    let target = vec!["topology:sequential".to_string(), "fused".to_string()];
    assert_eq!(
        validate_schedule_feature_coverage("app", "backend", &artifact, &target),
        Ok(())
    );

    assert_eq!(
        validate_schedule_feature_coverage("app", "backend", &artifact, &artifact[..1]),
        Err(ScheduleFeatureError::FeatureAbsentFromTarget {
            feature: "fused".to_string(),
            target: "backend".to_string(),
        })
    );
    assert_eq!(
        validate_schedule_feature_coverage("app", "backend", &artifact[..1], &target),
        Err(ScheduleFeatureError::FeatureAbsentFromArtifact {
            feature: "fused".to_string(),
            artifact: "app".to_string(),
        })
    );
}

/// Every derived route defect is refused with the error that names it.
///
/// The base record is a route the compiler produced, so each case removes one
/// real fact rather than asserting against a shape invented here.
#[test]
fn every_route_defect_is_refused_with_the_error_that_names_it() {
    let base = compiled_route();
    assert_eq!(validate_production_route(&base), Ok(()));

    let mut single_node = base.clone();
    single_node.graph_node_count = 1;
    assert_eq!(
        validate_production_route(&single_node),
        Err(ApplicationRouteError::IsolatedKernel {
            application_id: base.application_id.clone(),
            node_count: 1,
        })
    );

    let mut disconnected = base.clone();
    disconnected.graph_internal_edge_count = 0;
    assert_eq!(
        validate_production_route(&disconnected),
        Err(ApplicationRouteError::DisconnectedGraph {
            application_id: base.application_id.clone(),
        })
    );

    let mut unclosed = base.clone();
    unclosed.graph_closure_error = Some("value 2 has no producer".to_string());
    assert_eq!(
        validate_production_route(&unclosed),
        Err(ApplicationRouteError::GraphClosureFailed {
            application_id: base.application_id.clone(),
            reason: "value 2 has no producer".to_string(),
        })
    );

    let mut refused_request = base.clone();
    refused_request.compile_request_error = Some("budget is empty".to_string());
    assert_eq!(
        validate_production_route(&refused_request),
        Err(ApplicationRouteError::NotCompiled {
            application_id: base.application_id.clone(),
            reason: "budget is empty".to_string(),
        })
    );

    let mut no_artifact = base.clone();
    no_artifact.artifact_schema_version = None;
    assert_eq!(
        validate_production_route(&no_artifact),
        Err(ApplicationRouteError::NotCompiled {
            application_id: base.application_id.clone(),
            reason: "no artifact schema version was recorded".to_string(),
        })
    );

    let mut unbound = base.clone();
    unbound.unbound_graph_values = vec!["s1_b".to_string()];
    assert_eq!(
        validate_production_route(&unbound),
        Err(ApplicationRouteError::IncompleteResourceRoster {
            application_id: base.application_id.clone(),
            detail: "s1_b".to_string(),
        })
    );

    let mut unresolved = base.clone();
    unresolved.unresolved_abi_bindings = vec!["node 1 buffer `s1_b`".to_string()];
    assert_eq!(
        validate_production_route(&unresolved),
        Err(ApplicationRouteError::IncompleteResourceRoster {
            application_id: base.application_id.clone(),
            detail: "node 1 buffer `s1_b`".to_string(),
        })
    );

    let mut empty_roster = base.clone();
    empty_roster.resource_roster.clear();
    assert_eq!(
        validate_production_route(&empty_roster),
        Err(ApplicationRouteError::IncompleteResourceRoster {
            application_id: base.application_id.clone(),
            detail: "the roster is empty".to_string(),
        })
    );

    let mut unstable = base.clone();
    unstable.artifact_identity_stable = false;
    assert_eq!(
        validate_production_route(&unstable),
        Err(ApplicationRouteError::UnstableArtifactIdentity {
            application_id: base.application_id.clone(),
        })
    );

    let mut refused_target = base.clone();
    refused_target.target_errors = vec!["`spirv`: unsupported operation".to_string()];
    assert_eq!(
        validate_production_route(&refused_target),
        Err(ApplicationRouteError::TargetLoweringRefused {
            application_id: base.application_id.clone(),
            reason: "`spirv`: unsupported operation".to_string(),
        })
    );

    let mut no_payload = base.clone();
    no_payload.targets.clear();
    assert_eq!(
        validate_production_route(&no_payload),
        Err(ApplicationRouteError::NoTargetPayload {
            application_id: base.application_id.clone(),
        })
    );

    let mut refused_envelope = base.clone();
    refused_envelope.targets[0].envelope_admitted = false;
    let backend_id = refused_envelope.targets[0].backend_id.clone();
    assert_eq!(
        validate_production_route(&refused_envelope),
        Err(ApplicationRouteError::EnvelopeRefused {
            application_id: base.application_id.clone(),
            backend_id,
        })
    );

    let mut uncertified = base.clone();
    uncertified.uncertified_stages = vec![base.stages[0].clone()];
    assert_eq!(
        validate_production_route(&uncertified),
        Err(ApplicationRouteError::ReferenceOnlyEvidence {
            application_id: base.application_id.clone(),
            stages: vec![base.stages[0].clone()],
        })
    );

    let mut host_only = base.clone();
    host_only.device_certified_by.clear();
    assert_eq!(
        validate_production_route(&host_only),
        Err(ApplicationRouteError::ReferenceOnlyEvidence {
            application_id: base.application_id.clone(),
            stages: base.stages.clone(),
        }),
        "a route with only host-side evidence must not read as certified"
    );
}

/// One route the compiler produced, with device certification stated.
///
/// The route is derived and lowered for real. Only the certificate roster is
/// supplied here, because a device certificate is recorded evidence and not
/// something a unit test can measure.
fn compiled_route() -> ApplicationRouteRecord {
    let mut routes = derived_routes();
    let mut record = routes.remove(0);
    record.device_certified_by = vec!["stated-device-certificate.json".to_string()];
    record.uncertified_stages.clear();
    record
}

/// Every registered generic frontend dialect, derived and lowered.
fn derived_routes() -> Vec<ApplicationRouteRecord> {
    let dialects = derive_frontend_capabilities();
    assert!(
        !dialects.is_empty(),
        "no generic frontend dialect is registered, so no application can be derived"
    );
    dialects
        .iter()
        .map(|dialect| {
            let operations: Vec<&str> =
                dialect.operations.iter().map(|op| op.id.as_str()).collect();
            let derived = derive_application(&dialect.dialect_id, &operations);
            route_of(&derived, &[])
        })
        .collect()
}

/// Every registered generic frontend dialect declares operations the compiler can stage.
#[test]
fn every_registered_frontend_operation_carries_a_program_builder() {
    let dialects = derive_frontend_capabilities();
    assert!(
        !dialects.is_empty(),
        "no generic frontend dialect is registered"
    );
    for dialect in &dialects {
        assert!(
            !dialect.operations.is_empty(),
            "dialect `{}` declares no operation",
            dialect.dialect_id
        );
        for op in &dialect.operations {
            assert!(
                op.has_registered_builder,
                "frontend operation `{}` registers no program builder",
                op.id
            );
            assert!(
                op.signature_outputs > 0,
                "frontend operation `{}` declares no signature output",
                op.id
            );
        }
    }
}

/// Every registered production backend registers both facets a route needs.
#[test]
fn every_registered_production_backend_states_a_payload_format_and_both_facets() {
    let modules = derive_artifact_modules().expect("the backend registry starts");
    assert!(
        !modules.is_empty(),
        "no backend registers an artifact module"
    );
    let production: Vec<_> = modules
        .iter()
        .filter(|module| !module.reference_oracle)
        .collect();
    assert!(
        !production.is_empty(),
        "every registered backend is a conformance oracle, so nothing lowers a production artifact"
    );
    for module in production {
        assert!(
            module.target_format.is_some(),
            "production backend `{}` registers no payload format",
            module.backend_id
        );
        assert!(
            module.has_target_compiler,
            "production backend `{}` registers no target compiler",
            module.backend_id
        );
        assert!(
            module.has_materializer,
            "production backend `{}` registers no materializer",
            module.backend_id
        );
        assert!(
            module.semantic_operation_count > 0,
            "production backend `{}` claims no semantic operation",
            module.backend_id
        );
    }
}

/// A derived application reaches a lowered target payload through the production route.
///
/// This is the fact the gate's name states, and it is asserted against the real
/// compiler rather than a recorded number: the graph is staged from the live
/// registry, analyzed, compiled, and lowered through every registered
/// production backend, and the artifact and target records are compared feature
/// by feature.
#[test]
fn every_derived_application_reaches_a_lowered_target_payload() {
    for record in derived_routes() {
        assert!(
            record.graph_node_count >= 2,
            "application `{}` is a {}-node graph, which is an isolated kernel",
            record.application_id,
            record.graph_node_count
        );
        assert!(
            record.graph_internal_edge_count >= 1,
            "application `{}` has no value produced by one node and read by another",
            record.application_id
        );
        assert_eq!(
            record.graph_closure_error, None,
            "application `{}` did not close",
            record.application_id
        );
        assert_eq!(record.compile_request_error, None);
        assert_eq!(record.compile_error, None);
        assert!(record.artifact_schema_version.is_some());
        assert!(
            record.artifact_identity_stable,
            "application `{}` selected a different artifact identity on recompilation",
            record.application_id
        );
        assert!(
            record.unbound_graph_values.is_empty(),
            "application `{}` leaves graph values unbound: {:?}",
            record.application_id,
            record.unbound_graph_values
        );
        assert!(
            record.unresolved_abi_bindings.is_empty(),
            "application `{}` leaves ABI bindings unresolved: {:?}",
            record.application_id,
            record.unresolved_abi_bindings
        );
        assert!(
            record.target_errors.is_empty(),
            "application `{}` target lowering was refused: {:?}",
            record.application_id,
            record.target_errors
        );
        assert!(
            !record.targets.is_empty(),
            "application `{}` reached no target payload",
            record.application_id
        );

        for target in &record.targets {
            assert!(
                target.envelope_admitted,
                "application `{}` payload from `{}` was refused as an envelope",
                record.application_id, target.backend_id
            );
            assert!(
                target.entry_point_count > 0,
                "application `{}` payload from `{}` carries no entry point",
                record.application_id,
                target.backend_id
            );
            assert_eq!(
                target.arm_assignment_count,
                target.module_count,
                "application `{}` lowered {} module(s) through `{}` with {} arm assignment(s)",
                record.application_id,
                target.module_count,
                target.backend_id,
                target.arm_assignment_count
            );
            assert_eq!(
                Some(target.bundle_topology.clone()),
                record.selected_topology,
                "application `{}` lowered a topology through `{}` the artifact did not select",
                record.application_id,
                target.backend_id
            );
            assert_eq!(
                validate_schedule_feature_coverage(
                    &record.application_id,
                    &target.backend_id,
                    &record.artifact_features,
                    &target.target_features,
                ),
                Ok(()),
                "application `{}` and backend `{}` disagree on the selected schedule features",
                record.application_id,
                target.backend_id
            );
        }

        assert_eq!(
            validate_production_route(&record),
            Err(ApplicationRouteError::ReferenceOnlyEvidence {
                application_id: record.application_id.clone(),
                stages: record.stages.clone(),
            }),
            "a real route with no device certificate must be refused as reference-only"
        );
    }
}
