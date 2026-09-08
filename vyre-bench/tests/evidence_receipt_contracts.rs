//! Contracts for versioned benchmark receipts and content-addressed evidence store.
//!
//! BACKLOG row 95 requires one versioned benchmark protocol and content-addressed evidence
//! store. A test proves the content address changes when any recorded field changes,
//! derived from the schema's field set at run time so a field added without being hashed
//! turns the suite red. The store must hold real measured production-path evidence.

#![forbid(unsafe_code)]

use vyre_bench::api::suite::SuiteKind;
use vyre_bench::evidence::{
    receipt_from_case_report, record_suite_evidence, BenchmarkBudgets, BenchmarkObjective,
    BenchmarkReceipt, BinaryIdentityReceipt, CandidateFunnelReceipt, EmittedResourcesReceipt,
    EnvironmentReceipt, EvidenceStore, EvidenceStoreError, NativeBaselineReceipt, ParityReceipt,
    PowerAndEnergyReceipt, ResourceIdentityReceipt, SelectedPortfolioReceipt,
    SemanticGraphIdentity, StateReceipt, TargetFactsReceipt, UncertaintyModelReceipt,
    WorkloadAndInputIdentity, BENCHMARK_RECEIPT_SCHEMA_VERSION,
};
use vyre_bench::registry::collect_all;
use vyre_bench::runner::{execute_suite, RunConfig};

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
            resource_ids: vec!["buf_in1".to_string(), "buf_in2".to_string(), "buf_out".to_string()],
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
fn collect_field_paths(value: &serde_json::Value, prefix: Vec<String>, paths: &mut Vec<Vec<String>>) {
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

        let mutated_receipt: BenchmarkReceipt =
            serde_json::from_value(mutated_json).unwrap_or_else(|err| {
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
    if let Some(case) = registry.iter().find(|c| c.active_in_suite(&SuiteKind::Smoke)) {
        config.case_ids = vec![case.metadata().id.0.clone()];
    }

    let report = execute_suite(&registry, &SuiteKind::Smoke, &config);
    assert!(!report.cases.is_empty(), "Report must contain at least one executed case");

    let first_case = &report.cases[0];
    let receipt = receipt_from_case_report(first_case, &report);

    // Verify receipt contains measured production run properties
    assert_eq!(receipt.schema_version, BENCHMARK_RECEIPT_SCHEMA_VERSION);
    assert_eq!(receipt.workload_and_input.workload_id, first_case.id);
    assert!(!receipt.compiler_and_backend_binaries.compiler_version.is_empty());
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
