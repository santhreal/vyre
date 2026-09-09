//! Application domain release evidence contracts test suite.
//!
//! BACKLOG row 57 requires release evidence to include complete representative applications
//! from at least three unrelated domains, including dense numerical work, irregular stateful work,
//! and latency-sensitive interactive work. It records parity, compile/load time, p50/p99, throughput,
//! peak/resident bytes, cold/warm state, selected schedule, and comparison with the best available
//! native baseline on identical inputs. No proxy or isolated kernel can satisfy whole-application
//! readiness.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ValueContract, ValueLifetime};
use vyre_bench::workloads::{
    all_whole_application_workloads, dense_numerical_pipeline,
    generate_whole_application_evidence_suite, interactive_event_pipeline,
    irregular_stateful_traversal, ApplicationDomain, RequiredWholeApplicationField,
    WholeApplicationRecord, WholeApplicationRefusal, WholeApplicationWorkload,
    WHOLE_APPLICATION_RECORD_SCHEMA_V1,
};

/// 1. Closure test asserting that every registered application domain class is covered
/// by a whole-application workload and that every record carries every required field.
///
/// Derived at run time from `ApplicationDomain::ALL` and `RequiredWholeApplicationField::ALL`.
/// Adding a domain class or a required field must turn the suite RED until a decision is recorded.
#[test]
fn test_whole_application_domain_class_completeness_and_runtime_closure() {
    let workloads = all_whole_application_workloads();
    assert!(
        workloads.len() >= 3,
        "Fix: whole-application registry must declare at least 3 canonical applications, got {}",
        workloads.len()
    );

    let mut covered_domains = BTreeSet::new();

    for wl in &workloads {
        covered_domains.insert(wl.domain);

        let (graph, _inputs) = (wl.build_graph_and_inputs)();
        wl.validate_topology(&graph)
            .expect("Workload topology must be valid multi-node connected graph");

        // Execute and measure
        let record = wl
            .execute_and_measure(30)
            .expect("Whole-application execution and measurement must succeed");

        // Validate all required fields
        record
            .validate_required_fields()
            .expect("Whole-application record must carry all required fields");
        record
            .fail_closed_if_stale_or_partial()
            .expect("Fresh record must not fail closed");

        // Validate individual required field properties
        assert_eq!(
            record.schema_version, WHOLE_APPLICATION_RECORD_SCHEMA_V1,
            "Record schema version must be current"
        );
        assert!(
            record.graph_node_count >= 2,
            "Whole application must have at least 2 connected nodes"
        );
        assert!(
            record.graph_edge_count >= 1,
            "Whole application must have at least 1 internal edge"
        );
        assert!(
            record.compile_request_validated,
            "Must be compiled through validated CompileRequest"
        );
        assert!(
            record.executed_via_production_route,
            "Must execute through production route"
        );
        assert!(
            record.parity_result.is_exact_match,
            "Parity against reference must be exact"
        );
        assert!(record.compile_time_ns > 0, "Compile time must be recorded in ns");
        assert!(record.load_time_ns > 0, "Load time must be recorded in ns");
        assert!(record.p50_latency_ns > 0, "p50 latency must be recorded in ns");
        assert!(
            record.p99_latency_ns >= record.p50_latency_ns,
            "p99 latency must be >= p50 latency"
        );
        assert!(
            record.throughput.gflops.is_some()
                || record.throughput.gb_per_sec.is_some()
                || record.throughput.items_per_sec.is_some(),
            "Throughput must be recorded with explicit units"
        );
        assert!(record.peak_bytes > 0, "Peak bytes must be non-zero");
        assert!(
            record.cold_state.latency_ns > 0,
            "Cold state latency must be recorded"
        );
        assert!(
            record.warm_state.latency_ns > 0,
            "Warm state latency must be recorded"
        );
        assert!(
            !record.selected_schedule_id.is_empty(),
            "Selected schedule identity must be recorded"
        );
        assert!(
            !record.native_baseline_comparison.baseline_id.is_empty(),
            "Native baseline comparison must be recorded"
        );
        assert!(
            !record.native_baseline_comparison.identical_input_digest.is_empty(),
            "Native baseline must evaluate identical input digest"
        );
        assert!(
            record.native_baseline_comparison.native_p50_latency_ns > 0,
            "Native baseline latency must be non-zero"
        );
        assert!(
            record.native_baseline_comparison.speedup_ratio > 0.0,
            "Speedup ratio must be positive"
        );
    }

    // Programmatic check: every domain variant in ApplicationDomain::ALL must be covered
    for domain in ApplicationDomain::ALL {
        assert!(
            covered_domains.contains(&domain),
            "Fix: ApplicationDomain `{:?}` ({}) has no registered whole-application workload",
            domain,
            domain.display_name()
        );
    }

    // Check runtime derivation of all 16 required fields
    assert_eq!(
        RequiredWholeApplicationField::ALL.len(),
        16,
        "Fix: RequiredWholeApplicationField::ALL must contain all 16 canonical required fields"
    );
}

/// 2. Checked property contract: no isolated kernel or proxy can satisfy the whole-application class.
#[test]
fn test_isolated_kernel_cannot_satisfy_whole_application_class() {
    let dummy_wl = WholeApplicationWorkload {
        id: "workload.dummy_test",
        name: "Dummy Test Workload",
        domain: ApplicationDomain::DenseNumerical,
        description: "Test",
        pinned_native_baseline_id: "native.dummy",
        pinned_native_baseline_name: "Dummy Native",
        default_conditions: vyre_bench::workloads::NativeComparisonConditions {
            semantics: Some("exact".to_string()),
            dtype: Some("u32".to_string()),
            shapes: Some("[1024]".to_string()),
            raggedness: Some("uniform_contiguous".to_string()),
            initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
            target: Some("sm_90a".to_string()),
            stream: Some("stream_0".to_string()),
            toolchain_and_flags: Some("nvcc".to_string()),
            clock_and_power_state: Some("locked".to_string()),
            warmup: Some("300".to_string()),
            interleaving: Some("ab_ba".to_string()),
            repetitions: Some("30".to_string()),
            cache_state: Some("flushed".to_string()),
            objective: Some("minimize_latency".to_string()),
        },
        build_graph_and_inputs: || (ProgramGraph::new(), BTreeMap::new()),
    };

    // Construct a single-node graph (isolated kernel wrapped in a graph)
    let mut single_node_graph = ProgramGraph::new();
    single_node_graph
        .add_external_value(
            "out",
            ValueContract::dense_1d(DataType::U32, 16, BufferAccess::WriteOnly, ValueLifetime::Output),
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

    let refusal = dummy_wl.validate_topology(&single_node_graph).unwrap_err();
    match refusal {
        WholeApplicationRefusal::SingleNodeIsolatedKernel { node_count, .. } => {
            assert_eq!(node_count, 1);
        }
        other => panic!("Expected SingleNodeIsolatedKernel refusal, got: {other:?}"),
    }

    // Construct a 2-node graph with NO internal connecting edge (disconnected graph)
    let mut disconnected_graph = ProgramGraph::new();
    disconnected_graph
        .add_external_value(
            "out1",
            ValueContract::dense_1d(DataType::U32, 16, BufferAccess::WriteOnly, ValueLifetime::Output),
        )
        .unwrap();
    disconnected_graph
        .add_external_value(
            "out2",
            ValueContract::dense_1d(DataType::U32, 16, BufferAccess::WriteOnly, ValueLifetime::Output),
        )
        .unwrap();
    disconnected_graph
        .add_node(
            "node1",
            Program::wrapped(
                vec![BufferDecl::output("out1", 0, DataType::U32).with_count(16)],
                [16, 1, 1],
                vec![Node::store("out1", Expr::gid_x(), Expr::u32(1))],
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    disconnected_graph
        .add_node(
            "node2",
            Program::wrapped(
                vec![BufferDecl::output("out2", 0, DataType::U32).with_count(16)],
                [16, 1, 1],
                vec![Node::store("out2", Expr::gid_x(), Expr::u32(2))],
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

    let refusal_disc = dummy_wl.validate_topology(&disconnected_graph).unwrap_err();
    match refusal_disc {
        WholeApplicationRefusal::DisconnectedGraph { edge_count, .. } => {
            assert_eq!(edge_count, 0);
        }
        other => panic!("Expected DisconnectedGraph refusal, got: {other:?}"),
    }
}

/// 3. Fail-closed contract: stale schema versions and partial records must fail closed.
#[test]
fn test_stale_or_partial_record_fails_closed() {
    let valid_wl = dense_numerical_pipeline();
    let valid_record = valid_wl.execute_and_measure(30).unwrap();

    // 1. Stale schema version
    let mut stale_version = valid_record.clone();
    stale_version.schema_version = "vyre.whole-application-record.v0".to_string();
    assert!(stale_version.fail_closed_if_stale_or_partial().is_err());

    // 2. Missing compile time
    let mut missing_compile = valid_record.clone();
    missing_compile.compile_time_ns = 0;
    assert!(missing_compile.validate_required_fields().is_err());

    // 3. Missing load time
    let mut missing_load = valid_record.clone();
    missing_load.load_time_ns = 0;
    assert!(missing_load.validate_required_fields().is_err());

    // 4. Missing p50 latency
    let mut missing_p50 = valid_record.clone();
    missing_p50.p50_latency_ns = 0;
    assert!(missing_p50.validate_required_fields().is_err());

    // 5. Missing peak bytes
    let mut missing_peak = valid_record.clone();
    missing_peak.peak_bytes = 0;
    assert!(missing_peak.validate_required_fields().is_err());

    // 6. Missing selected schedule identity
    let mut missing_schedule = valid_record.clone();
    missing_schedule.selected_schedule_id = String::new();
    assert!(missing_schedule.validate_required_fields().is_err());

    // 7. Missing native baseline comparison
    let mut missing_baseline = valid_record.clone();
    missing_baseline.native_baseline_comparison.baseline_id = String::new();
    assert!(missing_baseline.validate_required_fields().is_err());

    // 8. Missing parity
    let mut missing_parity = valid_record.clone();
    missing_parity.parity_result.is_exact_match = false;
    assert!(missing_parity.validate_required_fields().is_err());
}

/// 4. End-to-end execution, reference parity, and native baseline comparison across all 3 domains.
#[test]
fn test_end_to_end_whole_application_reference_parity_and_execution() {
    // Dense Numerical
    let dense = dense_numerical_pipeline();
    let dense_record = dense.execute_and_measure(30).unwrap();
    assert_eq!(dense_record.domain, ApplicationDomain::DenseNumerical);
    assert_eq!(dense_record.graph_node_count, 3);
    assert!(dense_record.graph_edge_count >= 2);
    assert!(dense_record.parity_result.is_exact_match);
    assert!(dense_record.native_baseline_comparison.speedup_ratio > 0.0);

    // Irregular Stateful
    let irregular = irregular_stateful_traversal();
    let irregular_record = irregular.execute_and_measure(30).unwrap();
    assert_eq!(irregular_record.domain, ApplicationDomain::IrregularStateful);
    assert_eq!(irregular_record.graph_node_count, 3);
    assert!(irregular_record.graph_edge_count >= 2);
    assert!(irregular_record.parity_result.is_exact_match);
    assert!(irregular_record.native_baseline_comparison.speedup_ratio > 0.0);

    // Latency-Sensitive Interactive
    let interactive = interactive_event_pipeline();
    let interactive_record = interactive.execute_and_measure(30).unwrap();
    assert_eq!(interactive_record.domain, ApplicationDomain::LatencySensitiveInteractive);
    assert_eq!(interactive_record.graph_node_count, 3);
    assert!(interactive_record.graph_edge_count >= 2);
    assert!(interactive_record.parity_result.is_exact_match);
    assert!(interactive_record.native_baseline_comparison.speedup_ratio > 0.0);
}

/// 5. Release evidence suite generation and artifact serialization.
#[test]
fn test_whole_application_evidence_artifact_generation() {
    let suite = generate_whole_application_evidence_suite(30)
        .expect("Whole-application evidence suite generation must succeed");

    assert_eq!(suite.schema_version, WHOLE_APPLICATION_RECORD_SCHEMA_V1);
    assert_eq!(suite.total_workloads, 3);
    assert_eq!(suite.covered_domains, 3);
    assert_eq!(suite.status, "passed");

    for record in &suite.records {
        let serialized = serde_json::to_string_pretty(record).unwrap();
        let deserialized: WholeApplicationRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.schema_version, record.schema_version);
        assert_eq!(deserialized.workload_id, record.workload_id);
        assert_eq!(deserialized.workload_name, record.workload_name);
        assert_eq!(deserialized.domain, record.domain);
        assert_eq!(deserialized.graph_node_count, record.graph_node_count);
        assert_eq!(deserialized.graph_edge_count, record.graph_edge_count);
        assert_eq!(deserialized.compile_time_ns, record.compile_time_ns);
        assert_eq!(deserialized.load_time_ns, record.load_time_ns);
        assert_eq!(deserialized.p50_latency_ns, record.p50_latency_ns);
        assert_eq!(deserialized.p99_latency_ns, record.p99_latency_ns);
        assert_eq!(deserialized.peak_bytes, record.peak_bytes);
        assert_eq!(deserialized.resident_bytes, record.resident_bytes);
        assert_eq!(deserialized.selected_schedule_id, record.selected_schedule_id);
        assert_eq!(deserialized.parity_result, record.parity_result);
        assert_eq!(deserialized.equality_conditions, record.equality_conditions);
        deserialized.validate_required_fields().unwrap();
    }

    let target_dir = vyre_test_support::monorepo::vyre_workspace_root().join("release/evidence/benchmarks");
    let written = vyre_bench::workloads::write_whole_application_evidence_artifacts(
        &target_dir,
        30,
    )
    .expect("Writing evidence artifacts must succeed");
    assert_eq!(written.len(), 4, "Must write 4 evidence artifacts (1 matrix + 3 records)");
}
