//! Report fixtures shared by the report emitters' tests.
//!
//! `flame` and `kernel_time_table` both need a `ReportSchema` carrying a few
//! cases with a few metrics. They each carried a copy of these three builders,
//! and the copies drifted: one recorded a single percentile per metric and the
//! other recorded p50 and p99 separately. The two-percentile form is the
//! general one, so it is what lives here.

use crate::api::case::Correctness;
use crate::api::metric::MetricStats;
use crate::probes::environment::EnvironmentData;
use crate::report::{CaseReport, ReportSchema, ReportSummary};
use std::collections::BTreeMap;

/// One metric's statistics, flat below p50 and flat above p99.
pub(super) fn stat(p50: u64, p99: u64) -> MetricStats {
    MetricStats {
        min: p50,
        p50,
        p90: p50,
        p95: p50,
        p99,
        p999: p99,
        p9999: p99,
        max: p99,
        mean: p50 as f64,
        stddev: 0.0,
        samples: 30,
        determinism_cv: None,
    }
}

/// One case carrying the named metrics, each as `(key, p50, p99)`.
pub(super) fn case(id: &str, stages: &[(&str, u64, u64)]) -> CaseReport {
    let mut metrics = BTreeMap::new();
    for (key, p50, p99) in stages {
        metrics.insert((*key).to_string(), stat(*p50, *p99));
    }
    CaseReport {
        id: id.to_string(),
        workload_fingerprint: format!("bench-case:{id}"),
        name: id.to_string(),
        owner_crate: "vyre-bench-test".to_string(),
        workload_class: "Micro".to_string(),
        tags: Vec::new(),
        backend_id: Some("test".to_string()),
        device_signature: Some("device-profile-v1:test".to_string()),
        held_out_corpus_id: Some(format!("heldout:bench-case:{id}")),
        needs_gpu: false,
        min_vram_bytes: None,
        min_input_bytes: None,
        required_features: Vec::new(),
        status: "ok".to_string(),
        wall_ns: None,
        correctness: Correctness::Exact,
        contract: None,
        performance: None,
        metrics,
        optimization_passes_applied: Vec::new(),
        artifacts: Vec::new(),
    }
}

/// A whole report around those cases, named by the emitter under test.
pub(super) fn schema(suite: &str, cases: Vec<CaseReport>) -> ReportSchema {
    ReportSchema {
            schema: "vyre-bench/v1".to_string(),
            run_id: "test".to_string(),
            suite: suite.to_string(),
            selected_backend: Some("test".to_string()),
            backend_profile: None,
            git: BTreeMap::new(),
            source_fingerprint: "test-source".to_string(),
            source_tree_fingerprint: "test-source-tree".to_string(),
            environment: EnvironmentData {
                os: "test".to_string(),
                architecture: "x86_64".to_string(),
                cpu_model: Some("test-cpu".to_string()),
                cpu_cores: 1,
                has_gpu: true,
                gpu_devices: vec![crate::probes::environment::GpuDeviceInfo {
                    name: "NVIDIA GeForce RTX 5090".to_string(),
                    driver_version: "test-driver".to_string(),
                    memory_total_mib: Some(32_768),
                    compute_capability_major: Some(12),
                    compute_capability_minor: Some(0),
                }],
                nvidia_driver_version: Some("test-driver".to_string()),
                nvidia_cuda_version: Some("test-cuda".to_string()),
                features: vec!["gpu.nvidia_smi".to_string()],
            },
            features: Vec::new(),
            cases,
            summary: ReportSummary {
                total_cases: 0,
                passed: 0,
                failed: 0,
                total_time_ns: 0,
                cache_hit_rate: None,
            },
            blockers: Vec::new(),
        }
}
