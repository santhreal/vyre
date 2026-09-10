//! Acceptance and regression contracts for BACKLOG row 47:
//! End-to-end performance floor against expert-written native kernels.
//!
//! Acceptance criteria verified:
//! 1. The harness refuses to record a measurement whose payload did not come from IR
//!    and schedule search, and refuses one where a required equality field is unset.
//! 2. The verdict is per cell: construct a result set whose mean improves while one
//!    cell regresses, and assert the verdict is a regression.
//! 3. Statistically indistinguishable is reported as its own verdict rather than a win,
//!    using the recorded estimator and uncertainty.
//! 4. The recorded field set is derived from the row's required list at run time and
//!    fails when a field is missing.
//! 5. Representative complete graphs and adversarial kernel-sized regions are covered
//!    and mapped to pinned external native baselines.

use std::collections::BTreeSet;

use vyre_bench::workloads::{
    evaluate_cell_comparison, generate_floor_comparison_report, validate_equality_conditions,
    validate_payload_provenance, AggregateVerdict, ArtifactBehavior, CaseMeasurementRecord,
    CellVerdict, EqualityDimension, MeasurementCell, MemoryMetrics, NativeComparisonConditions,
    PayloadProvenance, RequiredMeasurementField, SharedMemoryMetrics, SpillMetrics,
    StatisticalEstimator, ThroughputMetrics, VersionPinnedNativeBaseline, WorkloadSpecification,
    WorkspaceTraffic,
};

/// Construct a valid mock measurement record satisfying all 15 row 47 requirements.
fn valid_mock_measurement(
    case_id: &str,
    p50_latency_ns: u64,
    provenance: PayloadProvenance,
    conditions: NativeComparisonConditions,
) -> CaseMeasurementRecord {
    let samples = vec![
        p50_latency_ns - 100,
        p50_latency_ns - 50,
        p50_latency_ns,
        p50_latency_ns + 50,
        p50_latency_ns + 100,
    ];
    let estimator = StatisticalEstimator::from_samples(&samples, 0.95);

    CaseMeasurementRecord {
        case_id: case_id.to_string(),
        provenance,
        compile_time_ns: 25_000_000,
        candidate_count: 64,
        prediction_error: 0.035,
        register_count: 32,
        spill_metrics: SpillMetrics::zero(),
        shared_memory_metrics: SharedMemoryMetrics::new(2048, 1024),
        raw_device_samples: samples,
        estimator_and_uncertainty: estimator,
        device_time_ns: p50_latency_ns,
        p50_latency_ns,
        p99_latency_ns: p50_latency_ns + 120,
        throughput: ThroughputMetrics {
            throughput_gflops: Some(4250.0),
            throughput_gb_s: Some(1850.0),
            items_per_second: Some(10_000_000.0),
        },
        memory_metrics: MemoryMetrics {
            allocated_memory_bytes: 67_108_864,
            peak_memory_bytes: 134_217_728,
            vram_allocated_bytes: 268_435_456,
        },
        workspace_traffic: WorkspaceTraffic {
            bytes_read: 67_108_864,
            bytes_written: 67_108_864,
            bytes_touched: 134_217_728,
            workspace_traffic_bytes: 0,
        },
        artifact_behavior: ArtifactBehavior {
            cold_artifact_latency_ns: p50_latency_ns * 3,
            warm_artifact_latency_ns: p50_latency_ns,
            warmup_ratio: 3.0,
            resident_state_bytes: 67_108_864,
        },
        equality_conditions: conditions,
    }
}

/// Construct valid matching 14 equality conditions.
fn valid_equality_conditions() -> NativeComparisonConditions {
    NativeComparisonConditions {
        semantics: Some("exact".to_string()),
        dtype: Some("f32".to_string()),
        shapes: Some("[4096, 4096]".to_string()),
        raggedness: Some("uniform_contiguous".to_string()),
        initial_and_final_state: Some("clean_buffers_unaliased".to_string()),
        target: Some("sm_90a".to_string()),
        stream: Some("cuda_stream_non_blocking_0".to_string()),
        toolchain_and_flags: Some("nvcc_12.4_-O3".to_string()),
        clock_and_power_state: Some("locked_base_clock_tdp_100pct".to_string()),
        warmup: Some("300_warmup_iterations_discarded".to_string()),
        interleaving: Some("ab_ba_round_robin_interleaving".to_string()),
        repetitions: Some("30_measured_samples_clt".to_string()),
        cache_state: Some("flushed_l2_between_iterations".to_string()),
        objective: Some("minimize_p50_latency".to_string()),
    }
}

/// Construct a valid mock pinned native baseline.
fn valid_mock_native_baseline(
    baseline_id: &str,
    p50_latency_ns: u64,
    conditions: NativeComparisonConditions,
) -> VersionPinnedNativeBaseline {
    let meas = valid_mock_measurement(
        baseline_id,
        p50_latency_ns,
        PayloadProvenance::GenericIrAndScheduleSearch {
            ir_graph_digest: "native_reference".to_string(),
            candidates_searched: 1,
        },
        conditions.clone(),
    );

    VersionPinnedNativeBaseline {
        baseline_id: baseline_id.to_string(),
        name: "Mock Expert-Written Native Baseline".to_string(),
        vendor_or_library: "NVIDIA CUB / CUTLASS".to_string(),
        pinned_version: "3.5.0".to_string(),
        upstream_url: "https://github.com/NVIDIA/cutlass".to_string(),
        commit_or_tag: "v3.5.0".to_string(),
        source_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_string(),
        toolchain: "nvcc 12.4".to_string(),
        compilation_flags: vec!["-O3".to_string(), "-std=c++17".to_string()],
        equality_conditions: conditions,
        measurement: Some(meas),
    }
}

#[test]
fn test_harness_refuses_non_ir_schedule_search_provenance() {
    let case_id = "test.matrix_multiply";

    // 1. Valid provenance: GenericIrAndScheduleSearch with candidates > 0
    let valid_provenance = PayloadProvenance::GenericIrAndScheduleSearch {
        ir_graph_digest: "digest_1234abcd".to_string(),
        candidates_searched: 48,
    };
    assert!(
        validate_payload_provenance(case_id, &valid_provenance).is_ok(),
        "Generic IR and schedule search must be accepted"
    );

    // 2. Refuse: GenericIrAndScheduleSearch with 0 candidates
    let zero_search = PayloadProvenance::GenericIrAndScheduleSearch {
        ir_graph_digest: "digest_1234abcd".to_string(),
        candidates_searched: 0,
    };
    let err = validate_payload_provenance(case_id, &zero_search).unwrap_err();
    assert_eq!(err.case_id, case_id);
    assert!(err.reason.contains("0 candidates"));

    // 3. Refuse: Imported third-party kernel
    let imported = PayloadProvenance::ImportedKernel {
        kernel_name: "cublasSgemm_v2".to_string(),
        origin: "libcublas.so".to_string(),
    };
    let err = validate_payload_provenance(case_id, &imported).unwrap_err();
    assert!(err.reason.contains("imported kernel `cublasSgemm_v2`"));

    // 4. Refuse: Wrapper
    let wrapper = PayloadProvenance::Wrapper {
        wrapper_name: "CudaDriverShimWrapper".to_string(),
    };
    let err = validate_payload_provenance(case_id, &wrapper).unwrap_err();
    assert!(err.reason.contains("wrapper `CudaDriverShimWrapper`"));

    // 5. Refuse: Source template
    let template = PayloadProvenance::SourceTemplate {
        template_name: "templates/gemm_hardcoded.ptx".to_string(),
    };
    let err = validate_payload_provenance(case_id, &template).unwrap_err();
    assert!(err
        .reason
        .contains("source template `templates/gemm_hardcoded.ptx`"));

    // 6. Refuse: Model name dispatch
    let model_dispatch = PayloadProvenance::ModelNameDispatch {
        model_name: "resnet50_conv2d_fused".to_string(),
    };
    let err = validate_payload_provenance(case_id, &model_dispatch).unwrap_err();
    assert!(err
        .reason
        .contains("model name dispatch `resnet50_conv2d_fused`"));
}

#[test]
fn test_harness_refuses_unset_and_mismatched_equality_conditions() {
    let case_id = "test.attention_recurrence";
    let base_conditions = valid_equality_conditions();

    // Verify valid equality conditions pass
    assert!(
        validate_equality_conditions(case_id, &base_conditions, &base_conditions).is_ok(),
        "Identical equality conditions must pass"
    );

    // Test each of the 14 dimensions when UNSET in Vyre measurement
    for dim in EqualityDimension::ALL {
        let mut unset_vyre = base_conditions.clone();
        match dim {
            EqualityDimension::Semantics => unset_vyre.semantics = None,
            EqualityDimension::Dtype => unset_vyre.dtype = None,
            EqualityDimension::Shapes => unset_vyre.shapes = None,
            EqualityDimension::Raggedness => unset_vyre.raggedness = None,
            EqualityDimension::InitialAndFinalState => unset_vyre.initial_and_final_state = None,
            EqualityDimension::Target => unset_vyre.target = None,
            EqualityDimension::Stream => unset_vyre.stream = None,
            EqualityDimension::ToolchainAndFlags => unset_vyre.toolchain_and_flags = None,
            EqualityDimension::ClockAndPowerState => unset_vyre.clock_and_power_state = None,
            EqualityDimension::Warmup => unset_vyre.warmup = None,
            EqualityDimension::Interleaving => unset_vyre.interleaving = None,
            EqualityDimension::Repetitions => unset_vyre.repetitions = None,
            EqualityDimension::CacheState => unset_vyre.cache_state = None,
            EqualityDimension::Objective => unset_vyre.objective = None,
        }

        let err = validate_equality_conditions(case_id, &unset_vyre, &base_conditions).expect_err(
            &format!("Unset dimension `{}` must be refused", dim.as_str()),
        );
        match err {
            vyre_bench::workloads::EqualityConditionRefusal::Unset { dimension, .. } => {
                assert_eq!(dimension, *dim);
            }
            _ => panic!("Expected Unset refusal for {}", dim.as_str()),
        }
    }

    // Test each of the 14 dimensions when MISMATCHED between Vyre and native
    for dim in EqualityDimension::ALL {
        let mut mismatch_vyre = base_conditions.clone();
        match dim {
            EqualityDimension::Semantics => {
                mismatch_vyre.semantics = Some("different_semantics".into())
            }
            EqualityDimension::Dtype => mismatch_vyre.dtype = Some("f16".into()),
            EqualityDimension::Shapes => mismatch_vyre.shapes = Some("[2048, 2048]".into()),
            EqualityDimension::Raggedness => mismatch_vyre.raggedness = Some("ragged_skew".into()),
            EqualityDimension::InitialAndFinalState => {
                mismatch_vyre.initial_and_final_state = Some("dirty_state".into())
            }
            EqualityDimension::Target => mismatch_vyre.target = Some("sm_80".into()),
            EqualityDimension::Stream => mismatch_vyre.stream = Some("stream_1".into()),
            EqualityDimension::ToolchainAndFlags => {
                mismatch_vyre.toolchain_and_flags = Some("clang_-O2".into())
            }
            EqualityDimension::ClockAndPowerState => {
                mismatch_vyre.clock_and_power_state = Some("unthrottled".into())
            }
            EqualityDimension::Warmup => mismatch_vyre.warmup = Some("0_warmups".into()),
            EqualityDimension::Interleaving => {
                mismatch_vyre.interleaving = Some("sequential_no_interleave".into())
            }
            EqualityDimension::Repetitions => mismatch_vyre.repetitions = Some("5_samples".into()),
            EqualityDimension::CacheState => mismatch_vyre.cache_state = Some("cold_l2".into()),
            EqualityDimension::Objective => {
                mismatch_vyre.objective = Some("minimize_binary_size".into())
            }
        }

        let err =
            validate_equality_conditions(case_id, &mismatch_vyre, &base_conditions).expect_err(
                &format!("Mismatched dimension `{}` must be refused", dim.as_str()),
            );
        match err {
            vyre_bench::workloads::EqualityConditionRefusal::Mismatched { dimension, .. } => {
                assert_eq!(dimension, *dim);
            }
            _ => panic!("Expected Mismatched refusal for {}", dim.as_str()),
        }
    }
}

#[test]
fn test_per_cell_verdict_rejects_regression_despite_improving_mean() {
    let conditions = valid_equality_conditions();
    let prov = PayloadProvenance::GenericIrAndScheduleSearch {
        ir_graph_digest: "graph_digest_abc".to_string(),
        candidates_searched: 32,
    };

    // Cell 1: Uniform contiguous layout, stateless mode -> Speedup 4.0x (WIN)
    let cell1 = MeasurementCell::new("workload.gemm", "sm_90a", "uniform_contiguous", "stateless");
    let vyre1 = valid_mock_measurement(
        "case.gemm.uniform",
        1_000_000,
        prov.clone(),
        conditions.clone(),
    );
    let native1 = valid_mock_native_baseline("native.gemm", 4_000_000, conditions.clone());
    let eval1 = evaluate_cell_comparison(&cell1, &vyre1, &native1, 2.5);
    assert!(eval1.verdict.is_win(), "Cell 1 must be a Win");

    // Cell 2: Interleaved strided layout, stateless mode -> Speedup 3.0x (WIN)
    let cell2 = MeasurementCell::new(
        "workload.gemm",
        "sm_90a",
        "interleaved_strided",
        "stateless",
    );
    let vyre2 = valid_mock_measurement(
        "case.gemm.strided",
        1_000_000,
        prov.clone(),
        conditions.clone(),
    );
    let native2 = valid_mock_native_baseline("native.gemm", 3_000_000, conditions.clone());
    let eval2 = evaluate_cell_comparison(&cell2, &vyre2, &native2, 2.5);
    assert!(eval2.verdict.is_win(), "Cell 2 must be a Win");

    // Cell 3: Ragged power-law layout, retained-resident mode -> Speedup 0.60x (REGRESSION: 40% slower)
    let cell3 = MeasurementCell::new(
        "workload.gemm",
        "sm_90a",
        "ragged_power_law",
        "retained_resident_pool",
    );
    let vyre3 = valid_mock_measurement("case.gemm.ragged", 5_000_000, prov, conditions.clone());
    let native3 = valid_mock_native_baseline("native.gemm", 3_000_000, conditions);
    let eval3 = evaluate_cell_comparison(&cell3, &vyre3, &native3, 2.5);
    assert!(
        eval3.verdict.is_regression_or_loss(),
        "Cell 3 must be a Loss/Regression"
    );

    // Generate aggregate report over all three cells:
    // Mean speedup = (4.0 + 3.0 + 0.60) / 3 = 2.533x (> 1.0x, substantially positive mean!)
    let report = generate_floor_comparison_report(vec![eval1, eval2, eval3]);

    assert!(
        report.mean_speedup_x > 2.5,
        "Mean speedup is {:.2}x (> 1.0x)",
        report.mean_speedup_x
    );

    // BACKLOG row 47 requirement:
    // "Report a per-cell verdict, not an aggregate: a regression on one target,
    // sequence layout, or retained-state mode is a finding even when the mean improves."
    match report.overall_verdict {
        AggregateVerdict::Regression {
            regressed_cells,
            mean_speedup_x,
            reason,
        } => {
            assert_eq!(
                regressed_cells.len(),
                1,
                "Must capture the 1 regressed cell"
            );
            assert_eq!(regressed_cells[0], cell3, "Regressed cell must be cell3");
            assert!(mean_speedup_x > 2.0);
            assert!(reason.contains("1 cell(s) regressed"));
        }
        other => panic!("Expected AggregateVerdict::Regression, got: {:?}", other),
    }
}

#[test]
fn test_statistically_indistinguishable_is_reported_as_indistinguishable_never_win() {
    let conditions = valid_equality_conditions();
    let prov = PayloadProvenance::GenericIrAndScheduleSearch {
        ir_graph_digest: "graph_digest_indistinguishable".to_string(),
        candidates_searched: 64,
    };

    // Cell with 1.01x speedup within the 2.5% equivalence band
    let cell = MeasurementCell::new("workload.scan", "sm_90a", "uniform_contiguous", "stateless");
    let vyre = valid_mock_measurement("case.scan", 1_000_000, prov, conditions.clone());
    let native = valid_mock_native_baseline("native.scan", 1_010_000, conditions); // 1.01x ratio

    let eval = evaluate_cell_comparison(&cell, &vyre, &native, 2.5);

    // Assert cell verdict is StatisticallyIndistinguishable and NOT a Win
    assert!(
        eval.verdict.is_indistinguishable(),
        "Must be classified as StatisticallyIndistinguishable"
    );
    assert!(
        !eval.verdict.is_win(),
        "Statistically indistinguishable must NEVER be classified as a Win"
    );

    match &eval.verdict {
        CellVerdict::StatisticallyIndistinguishable {
            speedup_x,
            equivalence_band_pct,
            ..
        } => {
            assert!((speedup_x - 1.01).abs() < 1e-4);
            assert_eq!(*equivalence_band_pct, 2.5);
        }
        other => panic!("Expected StatisticallyIndistinguishable, got: {:?}", other),
    }

    // Assert aggregate verdict is Indistinguishable and NOT Pass
    let report = generate_floor_comparison_report(vec![eval]);
    match report.overall_verdict {
        AggregateVerdict::Indistinguishable {
            cell_count,
            mean_speedup_x,
        } => {
            assert_eq!(cell_count, 1);
            assert!((mean_speedup_x - 1.01).abs() < 1e-4);
        }
        other => panic!(
            "Expected AggregateVerdict::Indistinguishable, got: {:?}",
            other
        ),
    }
}

#[test]
fn test_runtime_derivation_of_row47_required_fields_and_missing_field_detection() {
    // 1. Derive the required field set at run time
    let field_names = RequiredMeasurementField::required_field_names();
    assert_eq!(
        field_names.len(),
        15,
        "a measurement record declares exactly 15 required fields"
    );

    let expected_fields: BTreeSet<&'static str> = [
        "compile_time_ns",
        "candidate_count",
        "prediction_error",
        "register_count",
        "spill_metrics",
        "shared_memory_metrics",
        "raw_device_samples",
        "estimator_and_uncertainty",
        "device_time_ns",
        "p50_latency_ns",
        "p99_latency_ns",
        "throughput",
        "memory_metrics",
        "workspace_traffic",
        "cold_and_warm_artifact_behavior",
    ]
    .into_iter()
    .collect();

    assert_eq!(
        field_names, expected_fields,
        "Derived field set must strictly match row 47 requirements"
    );

    // 2. Validate complete record passes
    let conditions = valid_equality_conditions();
    let prov = PayloadProvenance::GenericIrAndScheduleSearch {
        ir_graph_digest: "graph_digest".to_string(),
        candidates_searched: 16,
    };
    let complete_record = valid_mock_measurement("case.valid", 1_000_000, prov, conditions);
    assert!(
        complete_record.validate_required_fields().is_ok(),
        "Complete record must validate without errors"
    );

    // 3. Mutate fields one by one to verify missing-field detection
    {
        let mut missing_compile = complete_record.clone();
        missing_compile.compile_time_ns = 0;
        let err = missing_compile.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"compile_time_ns".to_string()));
    }

    {
        let mut missing_candidates = complete_record.clone();
        missing_candidates.candidate_count = 0;
        let err = missing_candidates.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"candidate_count".to_string()));
    }

    {
        let mut missing_pred_error = complete_record.clone();
        missing_pred_error.prediction_error = f64::NAN;
        let err = missing_pred_error.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"prediction_error".to_string()));
    }

    {
        let mut missing_regs = complete_record.clone();
        missing_regs.register_count = 0;
        let err = missing_regs.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"register_count".to_string()));
    }

    {
        let mut missing_samples = complete_record.clone();
        missing_samples.raw_device_samples = vec![];
        let err = missing_samples.validate_required_fields().unwrap_err();
        assert!(err
            .missing_fields
            .contains(&"raw_device_samples".to_string()));
    }

    {
        let mut missing_device_time = complete_record.clone();
        missing_device_time.device_time_ns = 0;
        let err = missing_device_time.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"device_time_ns".to_string()));
    }

    {
        let mut missing_p50 = complete_record.clone();
        missing_p50.p50_latency_ns = 0;
        let err = missing_p50.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"p50_latency_ns".to_string()));
    }

    {
        let mut missing_p99 = complete_record.clone();
        missing_p99.p99_latency_ns = 0;
        let err = missing_p99.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"p99_latency_ns".to_string()));
    }

    {
        let mut missing_throughput = complete_record.clone();
        missing_throughput.throughput = ThroughputMetrics::default();
        let err = missing_throughput.validate_required_fields().unwrap_err();
        assert!(err.missing_fields.contains(&"throughput".to_string()));
    }

    {
        let mut missing_warm_artifact = complete_record.clone();
        missing_warm_artifact
            .artifact_behavior
            .warm_artifact_latency_ns = 0;
        let err = missing_warm_artifact
            .validate_required_fields()
            .unwrap_err();
        assert!(err
            .missing_fields
            .contains(&"cold_and_warm_artifact_behavior".to_string()));
    }
}

#[test]
fn test_representative_workloads_and_pinned_native_baselines_manifest_contract() {
    let workloads = WorkloadSpecification::all_representative_workloads();

    // Must cover at least 6 canonical representative workloads
    assert!(
        workloads.len() >= 6,
        "Must define at least 6 canonical representative workloads"
    );

    let mut has_complete_graph = false;
    let mut has_adversarial = false;
    let mut domains = BTreeSet::new();

    for wl in &workloads {
        if wl.is_complete_graph {
            has_complete_graph = true;
        }
        if wl.is_adversarial {
            has_adversarial = true;
        }
        domains.insert(wl.domain);

        // Every workload must have complete 14 equality dimensions set
        let unset = wl.default_conditions.unset_dimensions();
        assert!(
            unset.is_empty(),
            "Workload `{}` default conditions must have no unset dimensions, found: {:?}",
            wl.id,
            unset
        );
    }

    assert!(
        has_complete_graph,
        "Must cover representative complete graph dataflow pipelines"
    );
    assert!(
        has_adversarial,
        "Must cover adversarial kernel-sized regions"
    );
    assert!(
        domains.len() >= 4,
        "Must cover at least 4 distinct workload domains (dense, irregular, recurrence, whole-app)"
    );

    // Validate native baselines TOML manifest
    let manifest_str = include_str!("../baselines/native_baselines.toml");
    let manifest: toml::Value =
        toml::from_str(manifest_str).expect("native_baselines.toml must be valid TOML");

    let baselines = manifest
        .get("baseline")
        .and_then(|v| v.as_array())
        .expect("native_baselines.toml must contain [[baseline]] entries");

    assert!(
        baselines.len() >= 6,
        "native_baselines.toml must define at least 6 pinned native baselines"
    );

    for b in baselines {
        let id = b
            .get("id")
            .and_then(|v| v.as_str())
            .expect("baseline id required");
        let version = b
            .get("pinned_version")
            .and_then(|v| v.as_str())
            .expect("pinned_version required");
        let sha256 = b
            .get("source_sha256")
            .and_then(|v| v.as_str())
            .expect("source_sha256 required");
        let flags = b
            .get("compilation_flags")
            .and_then(|v| v.as_array())
            .expect("compilation_flags required");

        assert!(!id.is_empty(), "baseline id must not be empty");
        assert!(!version.is_empty(), "pinned_version must not be empty");
        assert_eq!(
            sha256.len(),
            64,
            "source_sha256 must be a 64-char hex digest"
        );
        assert!(!flags.is_empty(), "compilation_flags must not be empty");
    }
}
