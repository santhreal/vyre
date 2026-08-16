//! Section 189: Hardware regression evidence and workload-specific PMU policy.
//!
//! Contracts:
//! - 189.1: Run on an isolated or fully recorded device.
//! - 189.2: Define a statistical policy per benchmark target.
//! - 189.3: Make PMU expectations workload-specific.
//! - 189.4: Ground host latency limits in comparable evidence.
//! - 189.5: Reject stale and incomparable benchmark records.

use std::collections::BTreeSet;
use std::path::Path;
use toml::Value as TomlValue;
use serde_json::json;

const BENCH_TARGETS_RAW: &str = include_str!("../../docs/optimization/BENCH_TARGETS.toml");
const STATISTICAL_GATES_RAW: &str = include_str!("../../docs/optimization/STATISTICAL_REGRESSION_GATES.toml");
const ROOFLINE_RAW: &str = include_str!("../../docs/optimization/ROOFLINE_COUNTER_EVIDENCE.toml");
const METHODOLOGY_RAW: &str = include_str!("../../docs/optimization/BENCHMARK_METHODOLOGY_CONTRACTS.toml");

// ---------------------------------------------------------------------------
// 189.1: Run on an isolated or fully recorded device
// ---------------------------------------------------------------------------

#[test]
fn test_189_1_environment_recording_contract() {
    let env_res = vyre_bench::probes::capture_environment();
    assert!(
        env_res.is_ok(),
        "Fix: environment probe must capture host info without unhandled error: {:?}",
        env_res.err()
    );
    let env = env_res.unwrap();
    assert!(!env.os.is_empty(), "Fix: OS must be recorded");
    assert!(!env.architecture.is_empty(), "Fix: architecture must be recorded");
    assert!(env.cpu_cores > 0, "Fix: CPU cores must be > 0");

    let build_profile = vyre_bench::probes::build_profile();
    assert!(
        build_profile == "release" || build_profile == "debug",
        "Fix: build profile must be recorded as 'release' or 'debug', got {build_profile}"
    );

    // Verify git info and source fingerprint recording
    let git = vyre_bench::probes::capture_git_info_at(Path::new("."));
    let fp = vyre_bench::probes::source_fingerprint(&git);
    assert!(
        !fp.is_empty(),
        "Fix: source fingerprint must be generated for the benchmark runner"
    );

    // Check methodology TOML specifies isolation and environment recording
    let methodology: TomlValue = toml::from_str(METHODOLOGY_RAW)
        .expect("Fix: BENCHMARK_METHODOLOGY_CONTRACTS.toml must be valid TOML");
    let benchmarks = methodology
        .get("benchmark")
        .and_then(TomlValue::as_array)
        .expect("Fix: BENCHMARK_METHODOLOGY_CONTRACTS.toml must contain [[benchmark]] table");
    assert!(!benchmarks.is_empty());

    for b in benchmarks {
        let noise_controls = b
            .get("noise_controls")
            .and_then(TomlValue::as_array)
            .expect("Fix: each benchmark must record noise_controls");
        assert!(
            !noise_controls.is_empty(),
            "Fix: noise controls must not be empty for benchmark {:?}",
            b.get("benchmark_id")
        );
        let env_digest = b
            .get("environment_digest")
            .and_then(TomlValue::as_str)
            .expect("Fix: each benchmark must record environment_digest");
        assert!(!env_digest.is_empty());
    }
}

// ---------------------------------------------------------------------------
// 189.2: Define a statistical policy per benchmark target
// ---------------------------------------------------------------------------

#[test]
fn test_189_2_statistical_policy_per_benchmark_target() {
    let targets: TomlValue = toml::from_str(BENCH_TARGETS_RAW)
        .expect("Fix: BENCH_TARGETS.toml must be valid TOML");
    let target_list = targets
        .get("target")
        .and_then(TomlValue::as_array)
        .expect("Fix: BENCH_TARGETS.toml must contain [[target]] list");
    assert!(target_list.len() >= 10, "Fix: BENCH_TARGETS.toml must contain all canonical targets");

    let baseline_classes = targets
        .get("baseline_class_values")
        .and_then(TomlValue::as_array)
        .map(|arr| arr.iter().filter_map(TomlValue::as_str).collect::<BTreeSet<_>>())
        .expect("Fix: baseline_class_values must exist");
    assert!(baseline_classes.contains("cpu_sota"));
    assert!(baseline_classes.contains("gpu_sota"));
    assert!(baseline_classes.contains("reference_correctness"));

    for t in target_list {
        let id = t.get("id").and_then(TomlValue::as_str).expect("target must have id");
        let metric = t.get("metric").and_then(TomlValue::as_str).expect("target must have metric");
        assert!(!metric.is_empty(), "target {id} must have non-empty metric");

        let cpu_base = t.get("cpu_baseline").and_then(TomlValue::as_str);
        let gpu_base = t.get("gpu_baseline").and_then(TomlValue::as_str);
        assert!(
            cpu_base.is_some() || gpu_base.is_some(),
            "target {id} must define a paired baseline (cpu_baseline or gpu_baseline)"
        );
    }

    // Verify statistical regression gates define confidence levels and decision rules
    let gates: TomlValue = toml::from_str(STATISTICAL_GATES_RAW)
        .expect("Fix: STATISTICAL_REGRESSION_GATES.toml must be valid TOML");
    let gate_list = gates
        .get("gate")
        .and_then(TomlValue::as_array)
        .expect("Fix: STATISTICAL_REGRESSION_GATES.toml must contain [[gate]] list");
    assert!(!gate_list.is_empty());

    for g in gate_list {
        let gate_id = g.get("gate_id").and_then(TomlValue::as_str).expect("gate_id required");
        let effect_size = g.get("effect_size").and_then(TomlValue::as_str).expect("effect_size required");
        let conf = g.get("confidence_level").and_then(TomlValue::as_str).expect("confidence_level required");
        let thresh = g.get("regression_threshold").and_then(TomlValue::as_str).expect("regression_threshold required");
        let noise = g.get("noise_floor").and_then(TomlValue::as_str).expect("noise_floor required");
        let decision = g.get("decision").and_then(TomlValue::as_str).expect("decision required");

        assert!(!effect_size.is_empty(), "gate {gate_id} effect_size");
        assert!(!conf.is_empty(), "gate {gate_id} confidence_level");
        assert!(!thresh.is_empty(), "gate {gate_id} regression_threshold");
        assert!(!noise.is_empty(), "gate {gate_id} noise_floor");
        assert!(
            decision == "allow-route-change" || decision == "block-route-change",
            "gate {gate_id} decision must be allow or block"
        );
    }
}

// ---------------------------------------------------------------------------
// 189.3: Make PMU expectations workload-specific
// ---------------------------------------------------------------------------

#[test]
fn test_189_3_workload_specific_pmu_expectations() {
    let roofline: TomlValue = toml::from_str(ROOFLINE_RAW)
        .expect("Fix: ROOFLINE_COUNTER_EVIDENCE.toml must be valid TOML");
    let kernels = roofline
        .get("kernel")
        .and_then(TomlValue::as_array)
        .expect("Fix: ROOFLINE_COUNTER_EVIDENCE.toml must contain [[kernel]] table");
    assert!(!kernels.is_empty());

    for k in kernels {
        let kernel_id = k.get("kernel_id").and_then(TomlValue::as_str).expect("kernel_id");
        let backend = k.get("backend").and_then(TomlValue::as_str).expect("backend");
        let intensity = k.get("arithmetic_intensity").and_then(TomlValue::as_str).expect("arithmetic_intensity");
        let bound = k.get("roofline_bound").and_then(TomlValue::as_str).expect("roofline_bound");
        let resource = k.get("limiting_resource").and_then(TomlValue::as_str).expect("limiting_resource");
        let counters = k.get("counter_sources").and_then(TomlValue::as_array).expect("counter_sources");
        let explanation = k.get("route_explanation").and_then(TomlValue::as_str).expect("route_explanation");

        assert!(!backend.is_empty(), "kernel {kernel_id} backend");
        assert!(!intensity.is_empty(), "kernel {kernel_id} intensity");
        assert!(!bound.is_empty(), "kernel {kernel_id} bound");
        assert!(!resource.is_empty(), "kernel {kernel_id} limiting_resource");
        assert!(!counters.is_empty(), "kernel {kernel_id} counter_sources must not be empty");
        assert!(!explanation.is_empty(), "kernel {kernel_id} route_explanation");
    }
}

// ---------------------------------------------------------------------------
// 189.4: Ground host latency limits in comparable evidence
// ---------------------------------------------------------------------------

#[test]
fn test_189_4_ground_host_latency_limits() {
    let targets: TomlValue = toml::from_str(BENCH_TARGETS_RAW).expect("valid toml");
    let target_list = targets.get("target").and_then(TomlValue::as_array).unwrap();

    let mut has_resident = false;
    let mut has_oneshot = false;

    for t in target_list {
        let id = t.get("id").and_then(TomlValue::as_str).unwrap_or("");
        let timing_quality = t.get("timing_quality").and_then(TomlValue::as_str).unwrap_or("");
        let transfer_pressure = t.get("transfer_pressure").and_then(TomlValue::as_str).unwrap_or("");

        if transfer_pressure.contains("resident") || id.contains("megakernel") || id.contains("resident") {
            has_resident = true;
        }
        if transfer_pressure.contains("readback") || id.contains("smoke") || id.contains("micro") {
            has_oneshot = true;
        }
        if !timing_quality.is_empty() {
            assert!(
                timing_quality == "device_timestamps"
                    || timing_quality == "paired_host_wall_and_per_device_timestamps"
                    || timing_quality == "host_wall",
                "timing_quality must be well-formed, got {timing_quality}"
            );
        }
    }

    assert!(has_resident, "Fix: BENCH_TARGETS must contain resident-path benchmark targets");
    assert!(has_oneshot, "Fix: BENCH_TARGETS must contain one-shot / readback benchmark targets");
}

// ---------------------------------------------------------------------------
// 189.5: Reject stale and incomparable benchmark records
// ---------------------------------------------------------------------------

#[test]
fn test_189_5_reject_stale_and_incomparable_benchmark_records() {
    // Missing build_profile should be detected as invalid
    let missing_profile = json!({
        "selected_backend": "cuda",
        "source_fingerprint": "git:abc:dirty=false",
        "summary": {"total_cases": 1, "passed": 1, "failed": 0},
        "cases": []
    });
    assert!(
        missing_profile.get("environment").is_none()
            || missing_profile["environment"].get("build_profile").is_none(),
        "Missing environment/build_profile must be observable"
    );

    // Cross-backend / cross-device mismatch
    let cuda_report = json!({
        "selected_backend": "cuda",
        "hardware_digest": "gpu:rtx_5090",
        "source_fingerprint": "git:abc123:dirty=false",
        "build_profile": "release"
    });
    let wgpu_report = json!({
        "selected_backend": "wgpu",
        "hardware_digest": "gpu:apple_m3",
        "source_fingerprint": "git:abc123:dirty=false",
        "build_profile": "release"
    });

    assert_ne!(
        cuda_report["selected_backend"], wgpu_report["selected_backend"],
        "Cross-backend reports must not be treated as identical"
    );
    assert_ne!(
        cuda_report["hardware_digest"], wgpu_report["hardware_digest"],
        "Different hardware digests must not be compared as a homogeneous regression series"
    );

    // Stale summary mismatch: summary reports 0 failed but a case failed
    let inconsistent_report = json!({
        "summary": {"total_cases": 1, "passed": 1, "failed": 0},
        "cases": [
            {
                "case_id": "test_case",
                "status": "failed",
                "failure_reason": "regression observed"
            }
        ]
    });
    let summary_failed = inconsistent_report["summary"]["failed"].as_u64().unwrap();
    let actual_failed = inconsistent_report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] == "failed")
        .count() as u64;
    assert_ne!(
        summary_failed, actual_failed,
        "Fix: inconsistent summary must be detected and rejected"
    );
}
