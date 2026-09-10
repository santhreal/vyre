//! Whole-application release evidence contracts.
//!
//! Release evidence covers complete representative applications from three
//! unrelated domains: dense numerical work, irregular stateful work, and
//! latency-sensitive interactive work. A record states parity, compile and load
//! time, p50 and p99 latency, throughput, peak and resident bytes, cold and
//! warm state, the selected schedule identity, and the comparison against a
//! version-pinned native baseline when one is measured. A proxy or an isolated
//! kernel is refused.
//!
//! The contracts here hold on any host. A record states a device figure only
//! when a device produced it, so the tests that dispatch are behind the
//! `device-tests` feature and the honesty contracts are checked against
//! constructed records and against the producer source.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ValueContract,
    ValueLifetime,
};
#[cfg(feature = "device-tests")]
use vyre_bench::workloads::{
    dense_numerical_pipeline, interactive_event_pipeline, irregular_stateful_traversal,
};
use vyre_bench::workloads::{
    all_whole_application_workloads, whole_application_native_baselines, ApplicationDomain,
    NativeComparisonConditions, RequiredWholeApplicationField, WholeAppNativeBaselineUnmeasured,
    WholeAppNativeComparisonRecord, WholeAppParityRecord, WholeAppProductionRouteRecord,
    WholeAppStateMetrics, WholeAppThroughputRecord, WholeApplicationRecord,
    WholeApplicationRecordField, WholeApplicationRefusal, WholeApplicationWorkload,
    MIN_MEASURED_SAMPLES, WHOLE_APPLICATION_RECORD_SCHEMA_V1, WHOLE_APPLICATION_RECORD_SCHEMA_V2,
};

/// Absolute path of one file inside this repository.
fn repository_file(relative: &str) -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root().join(relative)
}

/// Read one repository file as source text.
fn repository_source(relative: &str) -> String {
    let path = repository_file(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// Every `.rs` file under one directory, discovered at run time.
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        {
            let path = entry.expect("read directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Whether `source` carries a literal `YYYY-MM-DDTHH:MM:SSZ` timestamp.
fn contains_literal_timestamp(source: &str) -> bool {
    source.as_bytes().windows(20).any(|window| {
        window[4] == b'-'
            && window[7] == b'-'
            && window[10] == b'T'
            && window[13] == b':'
            && window[16] == b':'
            && window[19] == b'Z'
            && [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
                .iter()
                .all(|index| window[*index].is_ascii_digit())
    })
}

/// The 14 comparison conditions, every dimension set.
fn complete_conditions() -> NativeComparisonConditions {
    NativeComparisonConditions {
        semantics: Some("exact_u32".to_string()),
        dtype: Some("u32".to_string()),
        shapes: Some("[1024]".to_string()),
        raggedness: Some("uniform_contiguous".to_string()),
        initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
        target: Some("test_target".to_string()),
        stream: Some("stream_0".to_string()),
        toolchain_and_flags: Some("test_toolchain".to_string()),
        clock_and_power_state: Some("as_configured".to_string()),
        warmup: Some("1_cold_submission_discarded".to_string()),
        interleaving: Some("sequential".to_string()),
        repetitions: Some("30_measured_samples".to_string()),
        cache_state: Some("as_left_by_previous_submission".to_string()),
        objective: Some("minimize_p50_latency".to_string()),
    }
}

/// A complete record whose every field is stated, for validation contracts.
///
/// This record is constructed, never measured. It exercises the validation
/// rules without acquiring a device, so a defect in those rules is caught on
/// every host.
fn constructed_record() -> WholeApplicationRecord {
    WholeApplicationRecord {
        schema_version: WHOLE_APPLICATION_RECORD_SCHEMA_V2.to_string(),
        workload_id: "workload.whole_app.constructed".to_string(),
        workload_name: "Constructed Record".to_string(),
        domain: ApplicationDomain::DenseNumerical,
        graph_node_count: 3,
        graph_edge_count: 2,
        compile_request_validated: true,
        production_route: WholeAppProductionRouteRecord {
            backend_id: "test-backend".to_string(),
            device_id: "test-device".to_string(),
            device_generation: 1,
            artifact_digest: "a".repeat(64),
            target_payload_digest: "b".repeat(64),
            bound_resource_count: 4,
            completed_submissions: 31,
        },
        parity_result: WholeAppParityRecord {
            compared_values: 1,
            is_exact_match: true,
            max_ulp_distance: 0,
            reference_digest: "c".repeat(64),
            candidate_digest: "c".repeat(64),
            parity_status: "matched_exact_bytes".to_string(),
        },
        compile_time_ns: 4_000_000,
        load_time_ns: 900_000,
        measured_samples: 30,
        p50_latency_ns: 120_000,
        p99_latency_ns: 180_000,
        device_p50_latency_ns: Some(90_000),
        throughput: WholeAppThroughputRecord {
            gflops: None,
            gb_per_sec: Some(0.17),
            items_per_sec: Some(8_500_000.0),
        },
        peak_bytes: Some(20_480),
        resident_bytes: Some(0),
        cold_state: WholeAppStateMetrics {
            latency_ns: 400_000,
            memory_bytes: None,
            is_cold_start: true,
            warmup_ratio: Some(3.33),
        },
        warm_state: WholeAppStateMetrics {
            latency_ns: 120_000,
            memory_bytes: None,
            is_cold_start: false,
            warmup_ratio: Some(3.33),
        },
        selected_schedule_id: format!("schedule.{}", "d".repeat(64)),
        input_digest: "e".repeat(64),
        native_baseline_comparison: None,
        native_baseline_unmeasured: Some(WholeAppNativeBaselineUnmeasured {
            baseline_id: "native.test.baseline".to_string(),
            baseline_name: "Test Native Baseline".to_string(),
            reason: "no version-pinned native baseline is vendored on this host".to_string(),
        }),
        host_environment: "test-arch-test-os / test-backend / test-device".to_string(),
        recorded_at_utc: "1970-01-01T00:00:00Z".to_string(),
    }
}

/// Every application domain class has a registered whole-application workload
/// with a connected multi-node graph, and the required field space is closed.
///
/// The domain list and the required field list are read from the library at run
/// time, so adding either without a workload or a decision turns this red.
#[test]
fn test_whole_application_domain_class_completeness_and_runtime_closure() {
    let workloads = all_whole_application_workloads();
    assert!(
        workloads.len() >= 3,
        "Fix: whole-application registry must declare at least 3 canonical applications, got {}",
        workloads.len()
    );

    let mut covered_domains = BTreeSet::new();
    for workload in &workloads {
        covered_domains.insert(workload.domain);
        let (graph, inputs) = (workload.build_graph_and_inputs)();
        workload
            .validate_topology(&graph)
            .expect("Workload topology must be a connected multi-node graph");
        assert!(
            graph.nodes().len() >= 2,
            "workload `{}` declares {} nodes",
            workload.id,
            graph.nodes().len()
        );
        assert!(
            WholeApplicationWorkload::count_internal_edges(&graph) >= 1,
            "workload `{}` declares no internal edge",
            workload.id
        );
        for value in graph.values() {
            if value.producer.is_none() {
                assert!(
                    inputs.contains_key(&value.name),
                    "workload `{}` supplies no input bytes for external value `{}`",
                    workload.id,
                    value.name
                );
            }
        }
    }

    for domain in ApplicationDomain::ALL {
        assert!(
            covered_domains.contains(&domain),
            "Fix: ApplicationDomain `{:?}` ({}) has no registered whole-application workload",
            domain,
            domain.display_name()
        );
    }

    let required: BTreeSet<&str> = RequiredWholeApplicationField::ALL
        .iter()
        .map(|field| field.as_str())
        .collect();
    assert_eq!(
        required.len(),
        RequiredWholeApplicationField::ALL.len(),
        "Fix: two required fields share one serialized name"
    );
}

/// No isolated kernel and no disconnected pair satisfies the whole-application class.
#[test]
fn test_isolated_kernel_cannot_satisfy_whole_application_class() {
    let probe_workload = WholeApplicationWorkload {
        id: "workload.topology_probe",
        name: "Topology Probe",
        domain: ApplicationDomain::DenseNumerical,
        description: "Graph shapes the whole-application class refuses",
        pinned_native_baseline_id: "native.topology_probe",
        pinned_native_baseline_name: "Topology Probe Native",
        default_conditions: complete_conditions(),
        build_graph_and_inputs: || (ProgramGraph::new(), BTreeMap::new()),
    };

    let mut single_node_graph = ProgramGraph::new();
    single_node_graph
        .add_external_value(
            "out",
            ValueContract::dense_1d(
                DataType::U32,
                16,
                BufferAccess::WriteOnly,
                ValueLifetime::Output,
            ),
        )
        .unwrap();
    single_node_graph
        .add_node(
            "isolated_kernel",
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(16)],
                [16, 1, 1],
                vec![Node::store("out", Expr::gid_x(), Expr::u32(42))],
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

    match probe_workload
        .validate_topology(&single_node_graph)
        .unwrap_err()
    {
        WholeApplicationRefusal::SingleNodeIsolatedKernel { node_count, .. } => {
            assert_eq!(node_count, 1);
        }
        other => panic!("Expected SingleNodeIsolatedKernel refusal, got: {other:?}"),
    }

    let mut disconnected_graph = ProgramGraph::new();
    for name in ["out1", "out2"] {
        disconnected_graph
            .add_external_value(
                name,
                ValueContract::dense_1d(
                    DataType::U32,
                    16,
                    BufferAccess::WriteOnly,
                    ValueLifetime::Output,
                ),
            )
            .unwrap();
    }
    for (node, buffer, value) in [("node1", "out1", 1u32), ("node2", "out2", 2u32)] {
        disconnected_graph
            .add_node(
                node,
                Program::wrapped(
                    vec![BufferDecl::output(buffer, 0, DataType::U32).with_count(16)],
                    [16, 1, 1],
                    vec![Node::store(buffer, Expr::gid_x(), Expr::u32(value))],
                ),
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
    }

    match probe_workload
        .validate_topology(&disconnected_graph)
        .unwrap_err()
    {
        WholeApplicationRefusal::DisconnectedGraph { edge_count, .. } => {
            assert_eq!(edge_count, 0);
        }
        other => panic!("Expected DisconnectedGraph refusal, got: {other:?}"),
    }
}

/// A stale schema version and every partial field fail closed.
///
/// The prior schema version is refused by name, because a v1 record states a
/// device latency taken from the host reference evaluator.
#[test]
fn test_stale_or_partial_record_fails_closed() {
    let valid = constructed_record();
    valid
        .validate_required_fields()
        .expect("constructed record must state every field it claims");
    valid
        .fail_closed_if_stale_or_partial()
        .expect("constructed record must not fail closed");

    let mut superseded = valid.clone();
    superseded.schema_version = WHOLE_APPLICATION_RECORD_SCHEMA_V1.to_string();
    let refusal = superseded
        .fail_closed_if_stale_or_partial()
        .expect_err("a v1 record must fail closed");
    match refusal {
        WholeApplicationRefusal::StaleSchemaVersion {
            version, expected, ..
        } => {
            assert_eq!(version, WHOLE_APPLICATION_RECORD_SCHEMA_V1);
            assert_eq!(expected, WHOLE_APPLICATION_RECORD_SCHEMA_V2);
        }
        other => panic!("Expected StaleSchemaVersion refusal, got: {other:?}"),
    }

    let mutations: Vec<(&str, fn(&mut WholeApplicationRecord))> = vec![
        ("schema_version", |record| {
            record.schema_version = "vyre.whole-application-record.v0".to_string();
        }),
        ("graph_node_count", |record| record.graph_node_count = 1),
        ("graph_edge_count", |record| record.graph_edge_count = 0),
        ("compile_request_validated", |record| {
            record.compile_request_validated = false;
        }),
        ("production_route.backend_id", |record| {
            record.production_route.backend_id = String::new();
        }),
        ("production_route.device_id", |record| {
            record.production_route.device_id = String::new();
        }),
        ("production_route.artifact_digest", |record| {
            record.production_route.artifact_digest = String::new();
        }),
        ("production_route.target_payload_digest", |record| {
            record.production_route.target_payload_digest = String::new();
        }),
        ("production_route.bound_resource_count", |record| {
            record.production_route.bound_resource_count = 0;
        }),
        ("production_route.completed_submissions", |record| {
            record.production_route.completed_submissions = 0;
        }),
        ("parity_result.compared_values", |record| {
            record.parity_result.compared_values = 0;
        }),
        ("parity_result.is_exact_match", |record| {
            record.parity_result.is_exact_match = false;
        }),
        ("parity_result.candidate_digest", |record| {
            record.parity_result.candidate_digest = String::new();
        }),
        ("compile_time_ns", |record| record.compile_time_ns = 0),
        ("load_time_ns", |record| record.load_time_ns = 0),
        ("measured_samples", |record| {
            record.measured_samples = MIN_MEASURED_SAMPLES - 1;
        }),
        ("p50_latency_ns", |record| record.p50_latency_ns = 0),
        ("p99_latency_ns", |record| {
            record.p99_latency_ns = record.p50_latency_ns - 1;
        }),
        ("throughput", |record| {
            record.throughput = WholeAppThroughputRecord::default();
        }),
        ("peak_bytes_absent", |record| record.peak_bytes = None),
        ("peak_bytes_zero", |record| record.peak_bytes = Some(0)),
        ("resident_bytes", |record| record.resident_bytes = None),
        ("cold_state.latency_ns", |record| {
            record.cold_state.latency_ns = 0;
        }),
        ("warm_state.latency_ns", |record| {
            record.warm_state.latency_ns = 0;
        }),
        ("cold_and_warm_state_collapsed", |record| {
            record.warm_state.is_cold_start = true;
        }),
        ("selected_schedule_id", |record| {
            record.selected_schedule_id = String::new();
        }),
        ("input_digest", |record| record.input_digest = String::new()),
        ("native_baseline_absent_without_reason", |record| {
            record.native_baseline_unmeasured = None;
        }),
        ("native_baseline_unmeasured.reason", |record| {
            if let Some(unmeasured) = record.native_baseline_unmeasured.as_mut() {
                unmeasured.reason = "   ".to_string();
            }
        }),
        ("host_environment", |record| {
            record.host_environment = String::new();
        }),
        ("recorded_at_utc", |record| {
            record.recorded_at_utc = String::new();
        }),
    ];

    for (name, mutate) in mutations {
        let mut record = valid.clone();
        mutate(&mut record);
        assert!(
            record.validate_required_fields().is_err(),
            "Fix: validation accepted a record with `{name}` unstated or out of range"
        );
    }
}

/// A record cannot state both a comparison and a reason for its absence.
#[test]
fn test_native_baseline_comparison_and_absence_are_exclusive() {
    let mut record = constructed_record();
    record.native_baseline_comparison = Some(WholeAppNativeComparisonRecord {
        baseline_id: "native.test.baseline".to_string(),
        baseline_name: "Test Native Baseline".to_string(),
        baseline_pinned_version: "1.0.0".to_string(),
        identical_input_digest: record.input_digest.clone(),
        native_p50_latency_ns: 200_000,
        speedup_ratio: 1.6666,
        verdict: "win".to_string(),
        equality_conditions: complete_conditions(),
    });
    let error = record
        .validate_required_fields()
        .expect_err("a record stating both a comparison and its absence must fail");
    assert!(
        error
            .missing_fields
            .iter()
            .any(|field| field.contains("both present")),
        "Fix: validation must name the mutually exclusive baseline fields, got {:?}",
        error.missing_fields
    );

    record.native_baseline_unmeasured = None;
    record
        .validate_required_fields()
        .expect("a record with only a measured comparison must validate");
    assert!(
        record.readiness_gaps().is_empty(),
        "a record with a measured comparison has no readiness gap"
    );

    let comparison = record.native_baseline_comparison.as_mut().unwrap();
    comparison.identical_input_digest = "f".repeat(64);
    let error = record
        .validate_required_fields()
        .expect_err("a comparison on other inputs must fail");
    assert!(
        error
            .missing_fields
            .iter()
            .any(|field| field.contains("identical_input_digest")),
        "Fix: validation must reject a comparison whose inputs are not the measured inputs, got {:?}",
        error.missing_fields
    );
}

/// An unmeasured pinned baseline produces no comparison and a stated gap.
///
/// No whole-application native kernel is vendored and measured, so the catalog
/// refuses every pinned identifier and the record states that instead of a
/// derived ratio.
#[test]
fn test_unmeasured_native_baseline_produces_no_comparison() {
    let catalog = whole_application_native_baselines();
    for workload in all_whole_application_workloads() {
        let error = catalog
            .measured(workload.pinned_native_baseline_id)
            .expect_err("no whole-application native baseline is measured on this host");
        assert!(
            error.contains(workload.pinned_native_baseline_id),
            "Fix: the refusal must name the baseline it could not measure, got `{error}`"
        );
        assert!(
            error.contains("Fix:"),
            "Fix: the refusal must state the corrective action, got `{error}`"
        );
    }

    let record = constructed_record();
    let gaps: BTreeSet<&str> = record
        .readiness_gaps()
        .iter()
        .map(|field| field.as_str())
        .collect();
    assert!(
        gaps.contains(RequiredWholeApplicationField::NativeBaselineComparison.as_str()),
        "Fix: a record with no comparison must state the comparison as a readiness gap, got {gaps:?}"
    );
    assert!(
        gaps.contains(RequiredWholeApplicationField::EqualityConditions.as_str()),
        "Fix: a record with no comparison states no equality conditions, so that is a gap too"
    );
}

/// A comparison is refused when the pinned baseline carries no measurement.
#[test]
fn test_comparison_from_unmeasured_baseline_is_refused() {
    let baseline = vyre_bench::workloads::VersionPinnedNativeBaseline {
        baseline_id: "native.test.unmeasured".to_string(),
        name: "Unmeasured Native Baseline".to_string(),
        vendor_or_library: "Test Vendor".to_string(),
        pinned_version: "1.0.0".to_string(),
        upstream_url: "https://example.invalid/baseline".to_string(),
        commit_or_tag: "v1.0.0".to_string(),
        source_sha256: "0".repeat(64),
        toolchain: "test_toolchain".to_string(),
        compilation_flags: vec!["-O3".to_string()],
        equality_conditions: complete_conditions(),
        measurement: None,
    };
    let error = WholeAppNativeComparisonRecord::from_measured_baseline(
        "workload.whole_app.constructed",
        &baseline,
        &complete_conditions(),
        &"e".repeat(64),
        120_000,
    )
    .expect_err("a baseline with no measurement must not produce a comparison");
    assert!(
        error.contains("no recorded measurement"),
        "Fix: the refusal must state that the baseline was never measured, got `{error}`"
    );
}

/// Parity is derived from compared bytes, never asserted.
#[test]
fn test_parity_is_derived_from_compared_bytes() {
    let reference = BTreeMap::from([("out".to_string(), vec![1u8, 0, 0, 0, 2, 0, 0, 0])]);
    let identical = reference.clone();
    let matched = WholeAppParityRecord::compare(&reference, &identical)
        .expect("identical outputs must compare");
    assert!(matched.is_exact_match);
    assert_eq!(matched.compared_values, 1);
    assert_eq!(matched.max_ulp_distance, 0);
    assert_eq!(matched.parity_status, "matched_exact_bytes");
    assert_eq!(matched.reference_digest, matched.candidate_digest);

    let differing = BTreeMap::from([("out".to_string(), vec![1u8, 0, 0, 0, 9, 0, 0, 0])]);
    let mismatched = WholeAppParityRecord::compare(&reference, &differing)
        .expect("differing outputs of equal length must compare");
    assert!(
        !mismatched.is_exact_match,
        "Fix: parity must report differing bytes as a mismatch"
    );
    assert_eq!(mismatched.max_ulp_distance, 7);
    assert_eq!(
        mismatched.parity_status,
        "mismatched_max_lane_difference_7"
    );
    assert_ne!(mismatched.reference_digest, mismatched.candidate_digest);

    let renamed = BTreeMap::from([("other".to_string(), vec![1u8, 0, 0, 0, 2, 0, 0, 0])]);
    let error = WholeAppParityRecord::compare(&reference, &renamed)
        .expect_err("comparing different output names must fail");
    assert!(
        error.contains("output name sets differ"),
        "Fix: parity must refuse a comparison over a different output set, got `{error}`"
    );

    let empty = BTreeMap::new();
    assert!(
        WholeAppParityRecord::compare(&empty, &empty).is_err(),
        "Fix: parity over no outputs is not a parity result"
    );

    let truncated = BTreeMap::from([("out".to_string(), vec![1u8, 0, 0, 0])]);
    assert!(
        WholeAppParityRecord::compare(&reference, &truncated).is_err(),
        "Fix: parity must refuse outputs of differing byte length"
    );
}

/// Every serialized record field has one recorded source.
///
/// The field set is derived from the struct in the producer source and from the
/// serialized JSON keys. A field added to the record without an entry in
/// `WholeApplicationRecordField` turns this red, as does an entry naming a
/// field the record does not carry.
#[test]
fn test_every_record_field_has_a_recorded_source() {
    let source = repository_source("vyre-bench/src/workloads/whole_app.rs");
    let file = syn::parse_file(&source).expect("parse whole_app.rs");
    let record_struct = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == "WholeApplicationRecord" => Some(item),
            _ => None,
        })
        .expect("whole_app.rs must define WholeApplicationRecord");
    let declared: BTreeSet<String> = record_struct
        .fields
        .iter()
        .map(|field| {
            field
                .ident
                .as_ref()
                .expect("WholeApplicationRecord has named fields")
                .to_string()
        })
        .collect();

    let decided: BTreeSet<String> = WholeApplicationRecordField::ALL
        .iter()
        .map(|field| field.as_str().to_string())
        .collect();
    assert_eq!(
        decided.len(),
        WholeApplicationRecordField::ALL.len(),
        "Fix: two record field entries share one serialized name"
    );
    assert_eq!(
        declared, decided,
        "Fix: every WholeApplicationRecord field needs one WholeApplicationRecordField entry naming where its value is read from"
    );

    let mut with_comparison = constructed_record();
    with_comparison.native_baseline_unmeasured = None;
    with_comparison.native_baseline_comparison = Some(WholeAppNativeComparisonRecord {
        baseline_id: "native.test.baseline".to_string(),
        baseline_name: "Test Native Baseline".to_string(),
        baseline_pinned_version: "1.0.0".to_string(),
        identical_input_digest: with_comparison.input_digest.clone(),
        native_p50_latency_ns: 200_000,
        speedup_ratio: 1.6666,
        verdict: "win".to_string(),
        equality_conditions: complete_conditions(),
    });

    let mut serialized_keys = BTreeSet::new();
    for record in [constructed_record(), with_comparison] {
        let value = serde_json::to_value(&record).expect("serialize record");
        let object = value.as_object().expect("a record serializes as an object");
        for key in object.keys() {
            serialized_keys.insert(key.clone());
        }
        let restored: WholeApplicationRecord =
            serde_json::from_value(value).expect("a serialized record decodes back");
        assert_eq!(restored, record, "record serialization must round trip");
    }
    assert_eq!(
        serialized_keys, decided,
        "Fix: the serialized record keys and the recorded field decisions must be the same set"
    );

    for field in WholeApplicationRecordField::ALL {
        assert!(
            !field.source().as_str().is_empty(),
            "Fix: record field `{}` states no source",
            field.as_str()
        );
    }
}

/// The producer states no constant host, timestamp, schedule, or device claim.
///
/// Each of these was a published constant. The scan closes the class: a new
/// literal of the same shape in the producer turns this red before it reaches an
/// evidence artifact.
#[test]
fn test_producer_carries_no_measurement_constants() {
    for relative in [
        "vyre-bench/src/workloads/whole_app.rs",
        "vyre-bench/src/workloads/provenance.rs",
        "vyre-bench/src/workloads/native_baseline.rs",
    ] {
        let source = repository_source(relative);
        assert!(
            !contains_literal_timestamp(&source),
            "Fix: `{relative}` carries a literal UTC timestamp. A recorded instant comes from the clock at record time."
        );
        assert!(
            !source.contains("x86_64-linux-gnu /"),
            "Fix: `{relative}` carries a literal host environment string. A recorded host comes from the acquired device."
        );
        assert!(
            !source.contains("RTX "),
            "Fix: `{relative}` names a device model as a literal. A recorded device comes from the device identity."
        );
        assert!(
            !source.contains("executed_via_production_route: true"),
            "Fix: `{relative}` asserts the production route instead of recording the identities it passed through."
        );
        assert!(
            !source.contains("schedule.fused_megakernel_persistent_v1"),
            "Fix: `{relative}` carries a literal schedule identity. A recorded schedule identity is the selected plan's."
        );
    }

    let producer = repository_source("vyre-bench/src/workloads/whole_app.rs");
    for line in producer.lines() {
        let trimmed = line.trim();
        if !trimmed.contains("native_p50_latency_ns:") || trimmed.starts_with("pub ") {
            continue;
        }
        assert!(
            trimmed.contains("measurement.p50_latency_ns"),
            "Fix: a native baseline latency is read from the pinned baseline's own measurement, not derived. Offending assignment: `{trimmed}`"
        );
    }
    assert!(
        producer.contains("measurement.p50_latency_ns"),
        "Fix: the producer must read the native latency from the pinned baseline measurement"
    );
}

/// No test computes the tracked evidence directory to write evidence into.
///
/// The tracked whole-application artifacts have one producer, the
/// `whole-app-evidence` command, and a test writes to a temporary directory.
/// The file list is read from the test directory at run time.
#[test]
fn test_no_test_writes_into_tracked_evidence_paths() {
    let tests_dir = repository_file("vyre-bench/tests");
    let files = rust_files(&tests_dir);
    assert!(
        files.len() > 1,
        "Fix: the test directory scan found {} files",
        files.len()
    );
    for path in files {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let lines: Vec<&str> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if !line.contains("write_whole_application_evidence_artifacts(") {
                continue;
            }
            let end = (index + 5).min(lines.len());
            let call_site = lines[index..end].join("\n");
            for tracked in ["vyre_workspace_root", "release/evidence"] {
                assert!(
                    !call_site.contains(tracked),
                    "Fix: `{}` line {} writes whole-application evidence into a repository path resolved by `{tracked}`. A test writes into a temporary directory.",
                    path.display(),
                    index + 1
                );
            }
        }
    }
}

/// A record whose submissions do not cover its samples did not execute the route.
#[test]
fn test_production_route_is_derived_from_completed_submissions() {
    let mut record = constructed_record();
    assert!(record.executed_via_production_route());

    record.production_route.completed_submissions = record.measured_samples - 1;
    assert!(
        !record.executed_via_production_route(),
        "Fix: a record with fewer completed submissions than measured samples did not execute every sample on the device"
    );

    record.production_route.completed_submissions = record.measured_samples;
    record.production_route.device_id = String::new();
    assert!(
        !record.executed_via_production_route(),
        "Fix: a record with no device identity did not execute on a device"
    );
    assert_eq!(
        record.production_route.missing_identities(),
        vec!["production_route.device_id"]
    );
}

/// Fewer than the minimum measured samples is refused before any device work.
#[cfg(feature = "device-tests")]
#[test]
fn test_sample_floor_is_enforced() {
    let device = vyre_bench::workloads::WholeApplicationDevice::probe(None)
        .expect("this host must acquire a dispatch device for device-tests");
    let error = dense_numerical_pipeline()
        .execute_and_measure(&device, MIN_MEASURED_SAMPLES - 1)
        .expect_err("a sample count below the floor must be refused");
    assert!(
        error.contains(&MIN_MEASURED_SAMPLES.to_string()),
        "Fix: the refusal must name the sample floor, got `{error}`"
    );
}

/// Every domain executes on the acquired device and matches the reference.
#[cfg(feature = "device-tests")]
#[test]
fn test_end_to_end_whole_application_device_execution_and_parity() {
    let device = vyre_bench::workloads::WholeApplicationDevice::probe(None)
        .expect("this host must acquire a dispatch device for device-tests");

    for (workload, domain) in [
        (dense_numerical_pipeline(), ApplicationDomain::DenseNumerical),
        (
            irregular_stateful_traversal(),
            ApplicationDomain::IrregularStateful,
        ),
        (
            interactive_event_pipeline(),
            ApplicationDomain::LatencySensitiveInteractive,
        ),
    ] {
        let record = workload
            .execute_and_measure(&device, MIN_MEASURED_SAMPLES)
            .expect("whole-application execution on the acquired device must succeed");
        record
            .validate_required_fields()
            .expect("a measured record must state every field it claims");
        record
            .fail_closed_if_stale_or_partial()
            .expect("a fresh measured record must not fail closed");

        assert_eq!(record.domain, domain);
        assert_eq!(record.graph_node_count, 3);
        assert!(record.graph_edge_count >= 2);
        assert!(record.parity_result.is_exact_match);
        assert!(record.parity_result.compared_values >= 1);
        assert!(record.executed_via_production_route());
        assert_eq!(
            record.production_route.backend_id,
            device.backend_id(),
            "the record names the backend that executed it"
        );
        assert_eq!(record.production_route.artifact_digest.len(), 64);
        assert_eq!(record.production_route.target_payload_digest.len(), 64);
        assert_eq!(
            record.production_route.completed_submissions,
            MIN_MEASURED_SAMPLES + 1
        );
        assert!(record.p50_latency_ns > 0);
        assert!(record.p99_latency_ns >= record.p50_latency_ns);
        assert!(record.compile_time_ns > 0);
        assert!(record.load_time_ns > 0);
        assert!(record.peak_bytes.is_some_and(|bytes| bytes > 0));
        assert!(record.resident_bytes.is_some());
        assert!(record.throughput.gflops.is_none());
        assert!(record.throughput.gb_per_sec.is_some());
        assert!(record.selected_schedule_id.starts_with("schedule."));
        assert_eq!(record.selected_schedule_id.len(), "schedule.".len() + 64);
        assert!(record.input_digest.len() == 64);
        assert!(
            record.host_environment.contains(device.backend_id()),
            "the recorded host names the acquired backend, got `{}`",
            record.host_environment
        );
        assert!(
            record.recorded_at_utc.ends_with('Z') && record.recorded_at_utc.len() == 20,
            "the recorded instant is an RFC 3339 UTC timestamp, got `{}`",
            record.recorded_at_utc
        );
        assert!(
            record.native_baseline_comparison.is_none()
                && record.native_baseline_unmeasured.is_some(),
            "no whole-application native baseline is measured, so the record states the reason"
        );
    }
}

/// The producer writes the evidence artifacts into the directory it is given.
#[cfg(feature = "device-tests")]
#[test]
fn test_whole_application_evidence_artifacts_are_written_where_directed() {
    let suite = vyre_bench::workloads::generate_whole_application_evidence_suite(
        None,
        MIN_MEASURED_SAMPLES,
    )
    .expect("whole-application evidence suite generation must succeed on a device host");
    assert_eq!(suite.schema_version, WHOLE_APPLICATION_RECORD_SCHEMA_V2);
    assert_eq!(suite.total_workloads, 3);
    assert_eq!(suite.covered_domains, 3);
    assert_eq!(suite.status, "measured_with_unstated_fields");
    assert!(suite
        .readiness_gaps
        .contains(&RequiredWholeApplicationField::NativeBaselineComparison
            .as_str()
            .to_string()));

    let directory = tempfile::tempdir().expect("create a temporary directory");
    let written = vyre_bench::workloads::write_whole_application_evidence_artifacts(
        directory.path(),
        None,
        MIN_MEASURED_SAMPLES,
    )
    .expect("writing evidence artifacts must succeed");
    assert_eq!(
        written.len(),
        4,
        "one domain matrix and one record per domain"
    );
    for path in &written {
        assert!(
            path.starts_with(directory.path()),
            "the producer writes only into the directory it is given, got {}",
            path.display()
        );
        let text = fs::read_to_string(path).expect("read a written artifact");
        assert!(text.ends_with('\n'));
    }
}
