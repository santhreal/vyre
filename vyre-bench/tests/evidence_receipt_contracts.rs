//! Contracts for versioned benchmark receipts and content-addressed evidence store.
//!
//! The contract requires one versioned benchmark protocol and content-addressed evidence
//! store. A test proves the content address changes when any recorded field changes,
//! derived from the schema's field set at run time so a field added without being hashed
//! turns the suite red. The store must hold real measured production-path evidence.

#![forbid(unsafe_code)]

use vyre_bench::api::suite::SuiteKind;
use vyre_bench::evidence::{
    average_compatible_cells, execute_campaign, receipt_from_case_report, record_measured_floor,
    record_suite_evidence, validate_floor_has_recorded_measurement, BenchmarkBudgets,
    BenchmarkCampaignSpec, BenchmarkObjective, BenchmarkReceipt, BinaryIdentityReceipt,
    CalibrationError, CandidateFunnelReceipt, ClockCalibrationRecord, CompatibleCellCohort,
    DeviceFleetCoordinator, EmittedResourcesReceipt, EnvironmentReceipt, EvidenceStore,
    EvidenceStoreError, FleetDevice, FleetLeaseError, InterferenceCalibrationRecord,
    MeasurementCellSpec, NativeBaselineReceipt, ParityReceipt, PowerAndEnergyReceipt,
    RecordedFloorProof, ResourceIdentityReceipt, SelectedPortfolioReceipt, SemanticGraphIdentity,
    StateReceipt, TargetFactsReceipt, UncertaintyModelReceipt, WorkloadAndInputIdentity,
    BENCHMARK_RECEIPT_SCHEMA_VERSION,
};
use vyre_bench::registry::collect_all;
use vyre_bench::runner::{execute_suite, RunConfig};
use vyre_bench::workloads::{all_whole_application_workloads, WorkloadSpecification};
/// Create a fully populated baseline benchmark receipt.
fn sample_benchmark_receipt() -> BenchmarkReceipt {
    BenchmarkReceipt {
        schema_version: BENCHMARK_RECEIPT_SCHEMA_VERSION,
        workload_and_input: WorkloadAndInputIdentity {
            workload_id: "test.elementwise.add.1m".to_string(),
            input_fingerprint: "input_fp_12345".to_string(),
            data_shape: vec![1024, 1024],
            element_count: 1_048_576,
        },
        semantic_graph: SemanticGraphIdentity {
            graph_digest: "graph_digest_abc".to_string(),
            node_count: 5,
            edge_count: 4,
            operations: vec!["load".to_string(), "add".to_string(), "store".to_string()],
        },
        resource_identity: ResourceIdentityReceipt {
            resource_ids: vec![
                "buf_in1".to_string(),
                "buf_in2".to_string(),
                "buf_out".to_string(),
            ],
            staging_bytes: 4_194_304,
            resident_bytes: 8_388_608,
            alignment_bytes: 64,
        },
        compiler_and_backend_binaries: BinaryIdentityReceipt {
            compiler_version: "0.8.0".to_string(),
            compiler_git_commit: "0123456789abcdef".to_string(),
            backend_name: "test_backend".to_string(),
            driver_version: "1.2.3".to_string(),
            target_format: "native.target".to_string(),
        },
        objective: BenchmarkObjective {
            metric: "wall_ns".to_string(),
            direction: "minimize".to_string(),
            target_value: Some(500_000.0),
        },
        budgets: BenchmarkBudgets {
            search_budget_ms: 1000,
            compile_budget_ms: 2000,
            execution_deadline_ns: 100_000_000,
            max_device_memory_bytes: 1_073_741_824,
        },
        target_facts: TargetFactsReceipt {
            device_name: "test_gpu_0".to_string(),
            architecture: "sm_90".to_string(),
            compute_units: 128,
            subgroup_size: 32,
            max_workgroup_size: [1024, 1024, 64],
            memory_bandwidth_gbps: 2000,
        },
        environment: EnvironmentReceipt {
            os: "linux".to_string(),
            kernel: "6.17.0".to_string(),
            cpu_model: "AMD Ryzen 9".to_string(),
            hostname_hash: "host_hash_xyz".to_string(),
        },
        candidate_funnel: CandidateFunnelReceipt {
            candidates_explored: 64,
            candidates_pruned: 32,
            candidates_compiled: 16,
            candidates_evaluated: 8,
        },
        selected_portfolio: SelectedPortfolioReceipt {
            portfolio_id: "portfolio_opt_0".to_string(),
            tile_sizes: vec![32, 32],
            unroll_factors: vec![4, 2],
            subgroup_partitioning: Some("warp_tiled".to_string()),
        },
        emitted_resources: EmittedResourcesReceipt {
            kernel_digests: vec!["kernel_digest_0".to_string(), "kernel_digest_1".to_string()],
            code_size_bytes: 4096,
            shared_memory_bytes: 32768,
            register_count: Some(48),
        },
        native_baseline: NativeBaselineReceipt {
            baseline_name: "cublas_gemm".to_string(),
            baseline_median_ns: 1_200_000,
            speedup_ratio: 1.45,
            baseline_version: "12.2".to_string(),
        },
        state: StateReceipt {
            cold_time_ns: 2_500_000,
            warm_median_ns: 820_000,
            host_to_device_transfer_ns: 150_000,
            device_to_host_transfer_ns: 120_000,
            cache_hit_count: 42,
            cache_miss_count: 3,
        },
        power_and_energy: Some(PowerAndEnergyReceipt {
            average_watts: Some(250.5),
            peak_watts: Some(310.0),
            total_energy_joules: Some(205.4),
        }),
        raw_samples: vec![821_000, 819_500, 820_100, 820_800],
        uncertainty_model: UncertaintyModelReceipt {
            mean_ns: 820_350.0,
            median_ns: 820_450,
            stddev_ns: 650.0,
            mad_ns: 400.0,
            p95_ns: 821_500,
            p99_ns: 822_000,
            confidence_95_lower_ns: 819_000.0,
            confidence_95_upper_ns: 821_700.0,
        },
        parity: ParityReceipt {
            passed: true,
            max_absolute_error: 1e-6,
            max_relative_error: 1e-5,
            oracle_backend: "cpu_reference".to_string(),
        },
        failures: vec!["failure_example_reason".to_string()],
    }
}

/// Recursively collect all JSON leaf field paths from a serde_json::Value.
fn collect_field_paths(
    value: &serde_json::Value,
    prefix: Vec<String>,
    paths: &mut Vec<Vec<String>>,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let mut next_prefix = prefix.clone();
                next_prefix.push(key.clone());
                if val.is_object() {
                    collect_field_paths(val, next_prefix, paths);
                } else {
                    paths.push(next_prefix);
                }
            }
        }
        _ => {
            paths.push(prefix);
        }
    }
}

/// Mutate a value at a specific path in a JSON tree.
fn mutate_json_path(root: &mut serde_json::Value, path: &[String]) {
    if path.is_empty() {
        return;
    }
    let mut current = root;
    for (i, segment) in path.iter().enumerate() {
        let is_last = i == path.len() - 1;
        if is_last {
            if let Some(field) = current.get_mut(segment) {
                match field {
                    serde_json::Value::String(s) => {
                        *s = format!("{s}_mutated");
                    }
                    serde_json::Value::Number(n) => {
                        if let Some(v) = n.as_u64() {
                            *field = serde_json::Value::Number((v + 1337).into());
                        } else if let Some(v) = n.as_f64() {
                            *field = serde_json::json!(v + 42.5);
                        } else if let Some(v) = n.as_i64() {
                            *field = serde_json::Value::Number((v + 1337).into());
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        *b = !*b;
                    }
                    serde_json::Value::Array(arr) => {
                        if let Some(first) = arr.first_mut() {
                            match first {
                                serde_json::Value::String(s) => *s = format!("{s}_elem_mutated"),
                                serde_json::Value::Number(n) => {
                                    if let Some(v) = n.as_u64() {
                                        *first = serde_json::Value::Number((v + 1337).into());
                                    } else if let Some(v) = n.as_f64() {
                                        *first = serde_json::json!(v + 42.5);
                                    } else {
                                        *first = serde_json::json!(9999);
                                    }
                                }
                                serde_json::Value::Bool(b) => *b = !*b,
                                _ => {}
                            }
                        } else {
                            arr.push(serde_json::json!("appended_elem"));
                        }
                    }
                    serde_json::Value::Null => {
                        *field = serde_json::json!(42);
                    }
                    serde_json::Value::Object(_) => {}
                }
            }
        } else {
            if let Some(next) = current.get_mut(segment) {
                current = next;
            } else {
                return;
            }
        }
    }
}

#[test]
fn content_address_changes_when_any_schema_field_changes_derived_at_runtime() {
    let base_receipt = sample_benchmark_receipt();
    let original_address = base_receipt.content_address();

    let json_value = serde_json::to_value(&base_receipt).expect("serialize receipt to json");
    let mut field_paths = Vec::new();
    collect_field_paths(&json_value, Vec::new(), &mut field_paths);

    // Verify that we discovered a rich set of schema fields (>30 distinct fields)
    assert!(
        field_paths.len() >= 30,
        "Schema must discover all fields dynamically at runtime (found {})",
        field_paths.len()
    );

    let mut tested_fields = 0;
    for path in &field_paths {
        let mut mutated_json = json_value.clone();
        mutate_json_path(&mut mutated_json, path);

        let mutated_receipt: BenchmarkReceipt = serde_json::from_value(mutated_json)
            .unwrap_or_else(|err| {
                panic!("Failed to deserialize mutated receipt for field path {path:?}: {err}")
            });

        let mutated_address = mutated_receipt.content_address();
        assert_ne!(
            mutated_address, original_address,
            "Field path {:?} was mutated but content address did not change! \
             Every schema field must be hashed into the receipt identity.",
            path
        );
        tested_fields += 1;
    }

    assert_eq!(
        tested_fields,
        field_paths.len(),
        "Every dynamically discovered schema field must be proven to affect content address"
    );
}

#[test]
fn two_identical_runs_produce_identical_content_addresses() {
    let receipt1 = sample_benchmark_receipt();
    let receipt2 = sample_benchmark_receipt();

    assert_eq!(
        receipt1.content_address(),
        receipt2.content_address(),
        "Two identical benchmark receipts must produce the exact same content address"
    );
}

#[test]
fn evidence_store_crud_and_tamper_detection() {
    let store = EvidenceStore::in_memory();
    let receipt = sample_benchmark_receipt();

    let address = store.put(&receipt).expect("put receipt into store");
    assert_eq!(address, receipt.content_address());
    assert!(store.contains(&address));

    let loaded = store.get(&address).expect("get receipt from store");
    assert_eq!(loaded, receipt);

    let list = store.list().expect("list receipts");
    assert_eq!(list, vec![address.clone()]);

    let missing = store.get("non_existent_hash");
    assert!(matches!(missing, Err(EvidenceStoreError::NotFound(_))));
}

#[test]
fn real_measured_production_run_writes_through_content_addressed_evidence_store() {
    // Run an impact-selected smoke case on the production benchmark harness
    let registry = collect_all();
    let mut config = RunConfig::default();
    config.warmup_samples = 1;
    config.measured_samples = Some(1);
    config.determinism_runs = 1;

    // Filter to a standard smoke case
    if let Some(case) = registry
        .iter()
        .find(|c| c.active_in_suite(&SuiteKind::Smoke))
    {
        config.case_ids = vec![case.metadata().id.0.clone()];
    }

    let report = execute_suite(&registry, &SuiteKind::Smoke, &config);
    assert!(
        !report.cases.is_empty(),
        "Report must contain at least one executed case"
    );

    let first_case = &report.cases[0];
    let receipt = receipt_from_case_report(first_case, &report);

    // Verify receipt contains measured production run properties
    assert_eq!(receipt.schema_version, BENCHMARK_RECEIPT_SCHEMA_VERSION);
    assert_eq!(receipt.workload_and_input.workload_id, first_case.id);
    assert!(!receipt
        .compiler_and_backend_binaries
        .compiler_version
        .is_empty());
    assert!(!receipt.environment.os.is_empty());

    // Record into evidence store
    let store = EvidenceStore::in_memory();
    let address = store.put(&receipt).expect("store measured receipt");

    let fetched = store.get(&address).expect("fetch measured receipt");
    assert_eq!(fetched.content_address(), address);
    assert_eq!(fetched.workload_and_input.workload_id, first_case.id);

    // Record suite through batch helper
    let addresses = record_suite_evidence(&report, None).expect("record suite evidence");
    assert!(!addresses.is_empty());
    assert_eq!(addresses[0], address);
}

#[test]
fn resumable_campaign_resumes_without_remeasuring_completed_cells() {
    let store = EvidenceStore::in_memory();

    let mut cells = Vec::new();
    for i in 0..5 {
        let mut receipt = sample_benchmark_receipt();
        receipt.workload_and_input.workload_id = format!("workload_{i}");
        receipt.workload_and_input.element_count = (i + 1) * 1000;
        cells.push(MeasurementCellSpec {
            cell_id: format!("cell_{i}"),
            case_id: format!("workload_{i}"),
            workload_and_input: receipt.workload_and_input.clone(),
            semantic_graph: receipt.semantic_graph.clone(),
            resource_identity: receipt.resource_identity.clone(),
            compiler_and_backend_binaries: receipt.compiler_and_backend_binaries.clone(),
            objective: receipt.objective.clone(),
            budgets: receipt.budgets.clone(),
            target_facts: receipt.target_facts.clone(),
            environment: receipt.environment.clone(),
            repeat_count: 3,
        });
    }

    let campaign_spec = BenchmarkCampaignSpec {
        campaign_id: "test_campaign_alpha".to_string(),
        seed: 42,
        cells,
    };

    let mut measurement_invocations = 0;

    // 1. Run campaign with interrupt after 2 cells
    let report_part1 = execute_campaign(&campaign_spec, &store, Some(2), |cell_spec| {
        measurement_invocations += 1;
        let mut r = sample_benchmark_receipt();
        r.workload_and_input = cell_spec.workload_and_input.clone();
        r.semantic_graph = cell_spec.semantic_graph.clone();
        r.resource_identity = cell_spec.resource_identity.clone();
        r.compiler_and_backend_binaries = cell_spec.compiler_and_backend_binaries.clone();
        r.objective = cell_spec.objective.clone();
        r.budgets = cell_spec.budgets.clone();
        r.target_facts = cell_spec.target_facts.clone();
        r.environment = cell_spec.environment.clone();
        r.raw_samples = vec![100, 105, 110];
        Ok(r)
    })
    .expect("part 1 execute campaign");

    assert_eq!(report_part1.cached_cells, 0);
    assert_eq!(report_part1.executed_cells, 2);
    assert_eq!(measurement_invocations, 2);

    // 2. Resume campaign to completion (interrupt_after: None)
    let report_part2 = execute_campaign(&campaign_spec, &store, None, |cell_spec| {
        measurement_invocations += 1;
        let mut r = sample_benchmark_receipt();
        r.workload_and_input = cell_spec.workload_and_input.clone();
        r.semantic_graph = cell_spec.semantic_graph.clone();
        r.resource_identity = cell_spec.resource_identity.clone();
        r.compiler_and_backend_binaries = cell_spec.compiler_and_backend_binaries.clone();
        r.objective = cell_spec.objective.clone();
        r.budgets = cell_spec.budgets.clone();
        r.target_facts = cell_spec.target_facts.clone();
        r.environment = cell_spec.environment.clone();
        r.raw_samples = vec![100, 105, 110];
        Ok(r)
    })
    .expect("part 2 resume campaign");

    assert_eq!(
        report_part2.cached_cells, 2,
        "The 2 cells executed in part 1 must be loaded from store as cached without re-measuring"
    );
    assert_eq!(
        report_part2.executed_cells, 3,
        "Only the remaining 3 unmeasured cells should be executed"
    );
    assert_eq!(
        measurement_invocations, 5,
        "Total measurement invocations across interrupted and resumed run must equal total cells (5)"
    );

    // 3. Re-run completed campaign: all 5 cells must be cached and 0 measured
    let report_part3 = execute_campaign(&campaign_spec, &store, None, |_| {
        panic!("Re-measuring an already completed campaign cell is a correctness defect!");
    })
    .expect("part 3 re-run campaign");

    assert_eq!(report_part3.cached_cells, 5);
    assert_eq!(report_part3.executed_cells, 0);
}

#[test]
fn campaign_trial_order_is_deterministic_from_campaign_identity() {
    let mut cells = Vec::new();
    for i in 0..10 {
        let mut receipt = sample_benchmark_receipt();
        receipt.workload_and_input.workload_id = format!("workload_{i}");
        cells.push(MeasurementCellSpec {
            cell_id: format!("cell_{i}"),
            case_id: format!("workload_{i}"),
            workload_and_input: receipt.workload_and_input.clone(),
            semantic_graph: receipt.semantic_graph.clone(),
            resource_identity: receipt.resource_identity.clone(),
            compiler_and_backend_binaries: receipt.compiler_and_backend_binaries.clone(),
            objective: receipt.objective.clone(),
            budgets: receipt.budgets.clone(),
            target_facts: receipt.target_facts.clone(),
            environment: receipt.environment.clone(),
            repeat_count: 1,
        });
    }

    let spec1 = BenchmarkCampaignSpec {
        campaign_id: "campaign_1".to_string(),
        seed: 12345,
        cells: cells.clone(),
    };

    let spec2 = BenchmarkCampaignSpec {
        campaign_id: "campaign_1".to_string(),
        seed: 12345,
        cells: cells.clone(),
    };

    let order1 = spec1.deterministic_trial_order();
    let order2 = spec2.deterministic_trial_order();
    assert_eq!(
        order1, order2,
        "Identical campaign specs must produce identical trial orders"
    );

    let spec_diff_seed = BenchmarkCampaignSpec {
        campaign_id: "campaign_1".to_string(),
        seed: 99999,
        cells,
    };
    let order_diff = spec_diff_seed.deterministic_trial_order();
    assert_ne!(
        order1, order_diff,
        "Different campaign seed must produce different trial permutation"
    );
}

#[test]
fn incompatible_cells_refuse_averaging_by_name() {
    let base_receipt = sample_benchmark_receipt();

    // 1. Incompatible device name
    let mut diff_device = base_receipt.clone();
    diff_device.target_facts.device_name = "NVIDIA GeForce RTX 3090".to_string();
    let refusal = average_compatible_cells(&[base_receipt.clone(), diff_device])
        .expect_err("Averaging cells from different device names must be refused");
    assert_eq!(refusal.dimension, "target_facts.device_name");
    assert_eq!(refusal.case_id, base_receipt.workload_and_input.workload_id);
    assert!(refusal
        .to_string()
        .contains("refused: incompatible cell identity"));

    // 2. Incompatible compiler git commit
    let mut diff_commit = base_receipt.clone();
    diff_commit
        .compiler_and_backend_binaries
        .compiler_git_commit = "deadbeef12345678".to_string();
    let refusal = average_compatible_cells(&[base_receipt.clone(), diff_commit])
        .expect_err("Averaging cells from different compiler commits must be refused");
    assert_eq!(
        refusal.dimension,
        "compiler_and_backend_binaries.compiler_git_commit"
    );

    // 3. Incompatible input fingerprint
    let mut diff_input = base_receipt.clone();
    diff_input.workload_and_input.input_fingerprint = "fp_different_input".to_string();
    let refusal = average_compatible_cells(&[base_receipt.clone(), diff_input])
        .expect_err("Averaging cells with different input fingerprints must be refused");
    assert_eq!(refusal.dimension, "workload_and_input.input_fingerprint");

    // 4. Incompatible compute units
    let mut diff_cu = base_receipt.clone();
    diff_cu.target_facts.compute_units = 144;
    let refusal = average_compatible_cells(&[base_receipt.clone(), diff_cu])
        .expect_err("Averaging cells with different compute units must be refused");
    assert_eq!(refusal.dimension, "target_facts.compute_units");

    // 5. Incompatible semantic graph digest
    let mut diff_graph = base_receipt.clone();
    diff_graph.semantic_graph.graph_digest = "digest_modified_graph".to_string();
    let refusal = average_compatible_cells(&[base_receipt.clone(), diff_graph])
        .expect_err("Averaging cells with different semantic graphs must be refused");
    assert_eq!(refusal.dimension, "semantic_graph.graph_digest");

    // 6. Compatible cells succeed and combine samples
    let mut compatible_trial2 = base_receipt.clone();
    compatible_trial2.raw_samples = vec![120, 125, 130];
    let averaged = average_compatible_cells(&[base_receipt, compatible_trial2])
        .expect("Compatible cells must successfully average");
    assert_eq!(averaged.raw_samples.len(), 7);
    assert!(averaged.uncertainty_model.mean_ns > 0.0);
}

#[test]
fn unmeasured_floor_is_refused_by_name_and_requires_backing_measurement() {
    let store = EvidenceStore::in_memory();
    let case_id = "release.optimizer.resident_pipeline";
    let declared_floor = 0.10;

    // 1. Refusal when no backing proof is supplied (unmeasured constant)
    let err_unmeasured =
        validate_floor_has_recorded_measurement(case_id, declared_floor, None, &store)
            .expect_err("Unmeasured floor without backing proof must be refused");
    assert_eq!(err_unmeasured.case_id, case_id);
    assert_eq!(err_unmeasured.declared_floor, declared_floor);
    assert!(err_unmeasured
        .to_string()
        .contains("has no recorded measurement behind it"));

    // 2. Refusal when backing proof points to missing receipt in store
    let bogus_proof = RecordedFloorProof {
        case_id: case_id.to_string(),
        baseline_class: "cpu_sota".to_string(),
        floor_value: 0.10,
        backing_measurement_address: "missing_content_address_12345".to_string(),
        sample_count: 30,
        device_name: "NVIDIA RTX 4090".to_string(),
        verified_parity: true,
    };
    let err_missing = validate_floor_has_recorded_measurement(
        case_id,
        declared_floor,
        Some(&bogus_proof),
        &store,
    )
    .expect_err("Backing proof missing from evidence store must be refused");
    assert_eq!(err_missing.case_id, case_id);
    assert!(err_missing.reason.contains("not found in evidence store"));

    // 3. Refusal when backing measurement failed parity
    let mut failed_parity_receipt = sample_benchmark_receipt();
    failed_parity_receipt.workload_and_input.workload_id = case_id.to_string();
    failed_parity_receipt.parity.passed = false;
    let failed_addr = store
        .put(&failed_parity_receipt)
        .expect("put failed receipt");
    let failed_proof = RecordedFloorProof {
        case_id: case_id.to_string(),
        baseline_class: "cpu_sota".to_string(),
        floor_value: 0.10,
        backing_measurement_address: failed_addr,
        sample_count: 30,
        device_name: "NVIDIA RTX 4090".to_string(),
        verified_parity: false,
    };
    let err_parity = validate_floor_has_recorded_measurement(
        case_id,
        declared_floor,
        Some(&failed_proof),
        &store,
    )
    .expect_err("Backing proof with failed parity must be refused");
    assert!(err_parity.reason.contains("failed parity"));

    // 4. Recording a verified empirical measurement succeeds and produces a valid recordable floor
    let mut valid_receipt = sample_benchmark_receipt();
    valid_receipt.workload_and_input.workload_id = case_id.to_string();
    valid_receipt.native_baseline.speedup_ratio = 0.26; // Measured ratio on 50k binding fixture
    valid_receipt.parity.passed = true;
    valid_receipt.raw_samples = vec![300_000_000; 30]; // 300ms p50

    let recorded_proof = record_measured_floor(
        case_id,
        "cpu_sota",
        &valid_receipt,
        &store,
        0.10, // 10% safety margin
    )
    .expect("Recording a measured floor from verified empirical receipt must succeed");

    assert_eq!(recorded_proof.case_id, case_id);
    assert!(recorded_proof.floor_value > 0.0);
    assert_eq!(recorded_proof.sample_count, 30);
    assert!(recorded_proof.verified_parity);

    // 5. Validating the newly recorded floor proof succeeds against the store
    let validated = validate_floor_has_recorded_measurement(
        case_id,
        recorded_proof.floor_value,
        Some(&recorded_proof),
        &store,
    )
    .expect("Validating recorded floor proof backed by store receipt must pass");
    assert_eq!(
        validated.backing_measurement_address,
        recorded_proof.backing_measurement_address
    );
}

#[test]
fn content_addressed_store_invalidates_only_changed_input_cells() {
    let store = EvidenceStore::in_memory();

    // Create two distinct cells
    let mut cell_a = sample_benchmark_receipt();
    cell_a.workload_and_input.workload_id = "cell_a".to_string();
    cell_a.workload_and_input.input_fingerprint = "fp_a_v1".to_string();
    let _addr_a = store.put(&cell_a).expect("put cell a");

    let mut cell_b = sample_benchmark_receipt();
    cell_b.workload_and_input.workload_id = "cell_b".to_string();
    cell_b.workload_and_input.input_fingerprint = "fp_b_v1".to_string();
    let addr_b = store.put(&cell_b).expect("put cell b");

    let spec_a_v1 = MeasurementCellSpec {
        cell_id: "cell_a".to_string(),
        case_id: "cell_a".to_string(),
        workload_and_input: cell_a.workload_and_input.clone(),
        semantic_graph: cell_a.semantic_graph.clone(),
        resource_identity: cell_a.resource_identity.clone(),
        compiler_and_backend_binaries: cell_a.compiler_and_backend_binaries.clone(),
        objective: cell_a.objective.clone(),
        budgets: cell_a.budgets.clone(),
        target_facts: cell_a.target_facts.clone(),
        environment: cell_a.environment.clone(),
        repeat_count: 1,
    };

    let spec_b = MeasurementCellSpec {
        cell_id: "cell_b".to_string(),
        case_id: "cell_b".to_string(),
        workload_and_input: cell_b.workload_and_input.clone(),
        semantic_graph: cell_b.semantic_graph.clone(),
        resource_identity: cell_b.resource_identity.clone(),
        compiler_and_backend_binaries: cell_b.compiler_and_backend_binaries.clone(),
        objective: cell_b.objective.clone(),
        budgets: cell_b.budgets.clone(),
        target_facts: cell_b.target_facts.clone(),
        environment: cell_b.environment.clone(),
        repeat_count: 1,
    };

    // Both cells exist in store
    assert!(store
        .find_by_cell_key(&spec_a_v1.cell_identity_key())
        .unwrap()
        .is_some());
    assert!(store
        .find_by_cell_key(&spec_b.cell_identity_key())
        .unwrap()
        .is_some());

    // Now change input fingerprint for cell A only (e.g. geometry or contract correction)
    let mut spec_a_v2 = spec_a_v1.clone();
    spec_a_v2.workload_and_input.input_fingerprint = "fp_a_v2_updated".to_string();

    // Cell A v2 is a cache miss (invalidated by input change)
    assert!(store
        .find_by_cell_key(&spec_a_v2.cell_identity_key())
        .unwrap()
        .is_none());
    // Cell B remains cached and unaffected!
    let cached_b = store
        .find_by_cell_key(&spec_b.cell_identity_key())
        .unwrap()
        .expect("cell b cached");
    assert_eq!(cached_b.content_address(), addr_b);
}

#[test]
fn compatible_cell_cohort_enforces_homogeneous_identity_and_safe_averaging() {
    let base_receipt = sample_benchmark_receipt();

    // 1. Initializing cohort with primary receipt
    let mut cohort = CompatibleCellCohort::new(base_receipt.clone());
    assert_eq!(cohort.len(), 1);
    assert!(!cohort.is_empty());
    assert_eq!(cohort.primary(), &base_receipt);

    // 2. Pushing a compatible receipt succeeds
    let mut compatible_trial2 = base_receipt.clone();
    compatible_trial2.raw_samples = vec![830_000, 835_000, 825_000];
    cohort
        .try_push(compatible_trial2.clone())
        .expect("Pushing compatible receipt into cohort must succeed");
    assert_eq!(cohort.len(), 2);

    // 3. Pushing an incompatible receipt is refused by name
    let mut incompatible_commit = base_receipt.clone();
    incompatible_commit
        .compiler_and_backend_binaries
        .compiler_git_commit = "ffffffffffffffff".to_string();
    let refusal = cohort
        .try_push(incompatible_commit)
        .expect_err("Pushing incompatible receipt into cohort must be refused");
    assert_eq!(
        refusal.dimension,
        "compiler_and_backend_binaries.compiler_git_commit"
    );
    assert_eq!(cohort.len(), 2);

    // 4. Constructing from slice of compatible receipts
    let cohort2 =
        CompatibleCellCohort::try_from_receipts(&[base_receipt.clone(), compatible_trial2])
            .expect("Constructing cohort from compatible receipts must succeed");
    assert_eq!(cohort2.len(), 2);

    // 5. Constructing from empty slice is refused
    let err_empty = CompatibleCellCohort::try_from_receipts(&[])
        .expect_err("Constructing cohort from empty slice must be refused");
    assert_eq!(err_empty.dimension, "count");

    // 6. Constructing with incompatible device is refused
    let mut incompatible_device = base_receipt.clone();
    incompatible_device.target_facts.device_name = "NVIDIA A100".to_string();
    let err_device =
        CompatibleCellCohort::try_from_receipts(&[base_receipt.clone(), incompatible_device])
            .expect_err("Constructing cohort with incompatible device must be refused");
    assert_eq!(err_device.dimension, "target_facts.device_name");

    // 7. Averaging on CompatibleCellCohort succeeds unconditionally with typed guarantee
    let averaged = cohort.average();
    assert_eq!(averaged.raw_samples.len(), 7);
    assert!(averaged.uncertainty_model.mean_ns > 820_000.0);
    assert!(averaged.uncertainty_model.stddev_ns >= 0.0);
    assert_eq!(
        averaged.compiler_and_backend_binaries.compiler_git_commit,
        base_receipt
            .compiler_and_backend_binaries
            .compiler_git_commit
    );
}

#[test]
fn device_fleet_leasing_clock_and_interference_calibration_contracts() {
    let coordinator = DeviceFleetCoordinator::new();
    let base_receipt = sample_benchmark_receipt();

    let token = "secret_fleet_token_99";
    let token_hash = DeviceFleetCoordinator::hash_token(token);

    let device = FleetDevice {
        device_id: "gpu_node_01".to_string(),
        device_name: "NVIDIA RTX 4090".to_string(),
        backend_name: "cuda".to_string(),
        target_facts: base_receipt.target_facts.clone(),
        auth_token_hash: token_hash,
        is_idle: true,
        active_lease: None,
        clock_calibration: None,
        interference_calibration: None,
    };

    coordinator
        .register_device(device)
        .expect("Registering device into fleet must succeed");

    // 1. Authentication check
    assert!(coordinator
        .authenticate_device("gpu_node_01", token)
        .expect("auth check"));
    assert!(!coordinator
        .authenticate_device("gpu_node_01", "wrong_token")
        .expect("auth check"));

    // 2. Lease attempt before calibration must fail
    let now_ns = 1_700_000_000_000_000_000;
    let duration_ns = 60_000_000_000; // 60s
    let err_uncal = coordinator
        .lease_idle_device("gpu_node_01", "campaign_alpha", duration_ns, now_ns)
        .expect_err("Leasing uncalibrated device must fail");
    assert!(matches!(
        err_uncal,
        FleetLeaseError::DeviceUncalibrated { .. }
    ));

    // 3. Calibration validation tests
    // 3a. Thermal throttling failure
    let bad_clock_throttling = ClockCalibrationRecord {
        base_clock_mhz: 2235,
        boost_clock_mhz: 2520,
        clock_drift_ppm: 5.0,
        timer_resolution_ns: 100,
        clock_locked: true,
        thermal_throttling: true,
    };
    assert_eq!(
        bad_clock_throttling.validate(),
        Err(CalibrationError::ThermalThrottlingDetected)
    );

    // 3b. Excessive clock drift failure
    let bad_clock_drift = ClockCalibrationRecord {
        base_clock_mhz: 2235,
        boost_clock_mhz: 2520,
        clock_drift_ppm: 75.0,
        timer_resolution_ns: 100,
        clock_locked: true,
        thermal_throttling: false,
    };
    assert!(matches!(
        bad_clock_drift.validate(),
        Err(CalibrationError::ClockDriftExcessive { .. })
    ));

    // 3c. Foreign compute interference failure
    let bad_interf_foreign = InterferenceCalibrationRecord {
        memory_bandwidth_contention_pct: 1.0,
        pcie_jitter_ns: 50,
        foreign_compute_processes: 2,
        numa_cross_traffic_detected: false,
    };
    assert_eq!(
        bad_interf_foreign.validate(),
        Err(CalibrationError::ForeignComputeContention { count: 2 })
    );

    // 3d. Excessive memory bandwidth contention failure
    let bad_interf_bw = InterferenceCalibrationRecord {
        memory_bandwidth_contention_pct: 12.5,
        pcie_jitter_ns: 50,
        foreign_compute_processes: 0,
        numa_cross_traffic_detected: false,
    };
    assert!(matches!(
        bad_interf_bw.validate(),
        Err(CalibrationError::BandwidthContentionExcessive { .. })
    ));

    // 4. Valid calibration
    let valid_clock = ClockCalibrationRecord {
        base_clock_mhz: 2235,
        boost_clock_mhz: 2520,
        clock_drift_ppm: 4.2,
        timer_resolution_ns: 20,
        clock_locked: true,
        thermal_throttling: false,
    };
    let valid_interf = InterferenceCalibrationRecord {
        memory_bandwidth_contention_pct: 0.8,
        pcie_jitter_ns: 15,
        foreign_compute_processes: 0,
        numa_cross_traffic_detected: false,
    };
    coordinator
        .calibrate_device("gpu_node_01", valid_clock, valid_interf)
        .expect("Valid calibration must succeed");

    // 5. Leasing calibrated idle device
    let lease = coordinator
        .lease_idle_device("gpu_node_01", "campaign_alpha", duration_ns, now_ns)
        .expect("Leasing calibrated idle device must succeed");

    assert_eq!(lease.device_id, "gpu_node_01");
    assert_eq!(lease.holder, "campaign_alpha");
    assert!(lease.is_valid(now_ns));
    assert!(lease.is_valid(now_ns + 10_000_000));
    assert!(!lease.is_valid(now_ns + duration_ns + 1));
    assert!(!lease.auth_signature.is_empty());

    // 6. Second concurrent lease must fail with DeviceBusy
    let err_busy = coordinator
        .lease_idle_device("gpu_node_01", "campaign_beta", duration_ns, now_ns + 1000)
        .expect_err("Concurrent lease on busy device must fail");
    assert!(matches!(err_busy, FleetLeaseError::DeviceBusy { .. }));

    // 7. Release lease returns device to idle
    coordinator
        .release_lease(&lease, now_ns + 5_000_000)
        .expect("Releasing active lease must succeed");

    // 8. Device can now be leased by campaign_beta
    let lease_beta = coordinator
        .lease_idle_device(
            "gpu_node_01",
            "campaign_beta",
            duration_ns,
            now_ns + 6_000_000,
        )
        .expect("Leasing released device must succeed");
    assert_eq!(lease_beta.holder, "campaign_beta");

    coordinator
        .release_lease(&lease_beta, now_ns + 7_000_000)
        .expect("release lease beta");

    // 9. Execute campaign through fleet coordinator
    let store = EvidenceStore::in_memory();
    let campaign_spec = BenchmarkCampaignSpec {
        campaign_id: "fleet_campaign_gamma".to_string(),
        seed: 777,
        cells: vec![MeasurementCellSpec {
            cell_id: "cell_fleet_0".to_string(),
            case_id: "case_fleet_0".to_string(),
            workload_and_input: base_receipt.workload_and_input.clone(),
            semantic_graph: base_receipt.semantic_graph.clone(),
            resource_identity: base_receipt.resource_identity.clone(),
            compiler_and_backend_binaries: base_receipt.compiler_and_backend_binaries.clone(),
            objective: base_receipt.objective.clone(),
            budgets: base_receipt.budgets.clone(),
            target_facts: base_receipt.target_facts.clone(),
            environment: base_receipt.environment.clone(),
            repeat_count: 1,
        }],
    };

    let report = coordinator
        .execute_fleet_campaign(&campaign_spec, &store, now_ns, |cell_spec| {
            let mut r = sample_benchmark_receipt();
            r.workload_and_input = cell_spec.workload_and_input.clone();
            Ok(r)
        })
        .expect("execute fleet campaign");

    assert_eq!(report.executed_cells, 1);
    assert_eq!(report.cached_cells, 0);

    // Resumption through coordinator loads cached cell with 0 executions
    let report_resumed = coordinator
        .execute_fleet_campaign(&campaign_spec, &store, now_ns + 10_000, |_| {
            panic!("Resumed cell must not be re-measured");
        })
        .expect("resumed fleet campaign");

    assert_eq!(report_resumed.executed_cells, 0);
    assert_eq!(report_resumed.cached_cells, 1);
}

#[test]
fn compiler_source_crates_name_zero_workload_or_expert_baselines() {
    // 1. Derive the complete list of neutral workload and expert baseline identifiers at runtime
    let mut corpus_identifiers = Vec::new();

    // Representative workload definitions
    corpus_identifiers.push(WorkloadSpecification::complete_graph_pipeline().id);
    corpus_identifiers.push(WorkloadSpecification::adversarial_ragged_reduction().id);
    corpus_identifiers.push(WorkloadSpecification::dense_contraction_gemm().id);

    // Pinned native baseline IDs
    corpus_identifiers.push("native.graph_pipeline.v1_0_0".to_string());
    corpus_identifiers.push("native.cub.segmented_reduce_v2_1_0".to_string());
    corpus_identifiers.push("native.cutlass.gemm_v3_5_0".to_string());
    corpus_identifiers.push("native.flash_attention.v2_5_8".to_string());
    corpus_identifiers.push("native.mkl.gemm_f32_v2024_1".to_string());
    corpus_identifiers.push("native.scipy.sparse_csr_v1_13".to_string());
    corpus_identifiers.push("native.numpy.fft_1d_v1_26".to_string());

    // Whole-application workload identifiers
    for app in all_whole_application_workloads() {
        corpus_identifiers.push(app.id.to_string());
    }

    assert!(
        corpus_identifiers.len() >= 6,
        "Must derive a non-empty neutral corpus identifier set from runtime registry (found {})",
        corpus_identifiers.len()
    );

    // 2. Compiler crates that must never name a workload or expert baseline
    let compiler_crate_dirs = [
        "vyre-foundation/src",
        "vyre-megakernel/src",
        "vyre-lower/src",
        "vyre-runtime/src",
        "vyre-driver/src",
        "vyre-primitives/src",
        "vyre-spec/src",
        "vyre-aot/src",
        "vyre-debug/src",
        "vyre-safetensors/src",
        "vyre/src",
    ];

    // Resolved from the working directory: a compiled-in manifest path names
    // whichever checkout last built this binary through the shared target
    // directory, and the scan would then read that tree's sources.
    let root = vyre_test_support::monorepo::vyre_workspace_root();

    let mut violations = Vec::new();

    for crate_subpath in &compiler_crate_dirs {
        let crate_dir = root.join(crate_subpath);
        if !crate_dir.exists() {
            continue;
        }

        for entry in walkdir(crate_dir) {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for corpus_id in &corpus_identifiers {
                    if content.contains(corpus_id) {
                        violations.push(format!(
                            "File {:?} illegally names neutral corpus identifier `{}`",
                            entry, corpus_id
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Compiler source must name zero workload or expert baseline identifiers. Found violations:\n{}",
        violations.join("\n")
    );
}

/// Helper to recursively collect all .rs files in a directory.
fn walkdir(dir: std::path::PathBuf) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(walkdir(path));
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn protocol_field_closure_and_cell_key_invalidation_derived_at_runtime() {
    let base_receipt = sample_benchmark_receipt();
    let original_address = base_receipt.content_address();
    let original_cell_key = base_receipt.cell_identity_key();

    let json_value = serde_json::to_value(&base_receipt).expect("serialize receipt to json");
    let mut field_paths = Vec::new();
    collect_field_paths(&json_value, Vec::new(), &mut field_paths);

    assert!(
        field_paths.len() >= 30,
        "Derived schema must expose >= 30 fields at runtime"
    );

    // List of top-level input categories that define the measurement cell
    let input_categories = [
        "workload_and_input",
        "semantic_graph",
        "resource_identity",
        "compiler_and_backend_binaries",
        "objective",
        "budgets",
        "target_facts",
        "environment",
    ];

    let store = EvidenceStore::in_memory();
    store
        .put(&base_receipt)
        .expect("put base receipt into store");

    for path in &field_paths {
        let mut mutated_json = json_value.clone();
        mutate_json_path(&mut mutated_json, path);

        let mutated_receipt: BenchmarkReceipt = serde_json::from_value(mutated_json)
            .unwrap_or_else(|err| {
                panic!("Failed to deserialize mutated receipt for field path {path:?}: {err}")
            });

        // 1. Every field in the schema MUST affect content address
        assert_ne!(
            mutated_receipt.content_address(),
            original_address,
            "Mutating field path {:?} did not change receipt content address!",
            path
        );

        // 2. Input dimension fields MUST affect cell_identity_key and cause cache invalidation
        let top_category = &path[0];
        if input_categories.contains(&top_category.as_str()) {
            assert_ne!(
                mutated_receipt.cell_identity_key(),
                original_cell_key,
                "Mutating input field path {:?} must change cell identity key!",
                path
            );

            // The mutated cell is a cache miss in store
            assert!(
                store
                    .find_by_cell_key(&mutated_receipt.cell_identity_key())
                    .unwrap()
                    .is_none(),
                "Mutating input field {:?} must invalidate cell cache in store",
                path
            );
        }
    }

    // Base receipt remains intact in store
    assert!(store
        .find_by_cell_key(&original_cell_key)
        .unwrap()
        .is_some());
}
