use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use super::metrics::write_json;
use super::optimization::suite_case_has_cpu_sota_contract;
use super::runner::run_command_status;
use super::types::{
    BackendSuiteArtifact, BackendSuiteArtifactInput, BackendSuiteEvidence,
    MAX_RELEASE_BENCHMARK_TEXT_BYTES, MIN_CPU_SOTA_100X_RELEASE_CASES,
    MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR, MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR,
    MIN_CUDA_RELEASE_MEMORY_MIB, REQUIRED_CPU_SOTA_100X_CASES,
};

pub(super) fn write_cpu_100x_proof(workspace_root: &Path, artifacts: &[String]) {
    let mut cases = Vec::new();
    let mut blockers = Vec::new();
    let mut contract_case_count = 0usize;
    let mut passing_contract_case_count = 0usize;
    let mut min_wall_samples = None::<u64>;
    let mut min_baseline_wall_samples = None::<u64>;
    let mut min_wall_p50 = None::<u64>;
    let mut min_wall_p95 = None::<u64>;
    let mut min_wall_p99 = None::<u64>;
    let mut min_baseline_wall_p50 = None::<u64>;
    let mut min_baseline_wall_p95 = None::<u64>;
    let mut min_baseline_wall_p99 = None::<u64>;
    let mut observed_required_cases = std::collections::BTreeSet::new();
    let mut environment = None::<Value>;
    let mut git = None::<Value>;
    let mut source_fingerprint = None::<String>;
    let mut source_tree_fingerprint = None::<String>;
    for artifact in artifacts {
        let path = workspace_root.join(artifact);
        let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "100x source artifact `{artifact}` is unreadable: {error}"
                ));
                continue;
            }
        };
        let Ok(report) = serde_json::from_str::<Value>(&text) else {
            blockers.push(format!("100x source artifact `{artifact}` is invalid JSON"));
            continue;
        };
        if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
            blockers.push(format!(
                "100x source artifact `{artifact}` was not produced for cuda"
            ));
        }
        if environment.is_none() {
            environment = report.get("environment").cloned();
        }
        if git.is_none() {
            git = report.get("git").cloned();
        }
        if source_fingerprint.is_none() {
            source_fingerprint = report
                .get("source_fingerprint")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .or_else(|| {
                    report
                        .get("git")
                        .and_then(|git| git.get("commit"))
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .map(|commit| format!("git:{commit}"))
                });
        }
        if source_tree_fingerprint.is_none() {
            source_tree_fingerprint = report
                .get("source_tree_fingerprint")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
        let Some(report_cases) = report.get("cases").and_then(Value::as_array) else {
            blockers.push(format!(
                "100x source artifact `{artifact}` has no cases array"
            ));
            continue;
        };
        let report_backend = report.get("selected_backend").and_then(Value::as_str);
        for case in report_cases {
            let case_id = case
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("<unknown>");
            let case_failure_reason =
                crate::benchmark_evidence_semantics::benchmark_case_failure_reason(case);
            if let Some(reason) = &case_failure_reason {
                blockers.push(format!(
                    "100x source artifact `{artifact}` case `{case_id}` failed: {reason}"
                ));
            }
            let metrics = case.get("metrics").and_then(Value::as_object);
            if REQUIRED_CPU_SOTA_100X_CASES.contains(&case_id) {
                observed_required_cases.insert(case_id.to_string());
            }
            let wall_samples = metrics
                .and_then(|metrics| suite_metric_samples(metrics.get("wall_ns")))
                .unwrap_or(0);
            min_wall_samples =
                Some(min_wall_samples.map_or(wall_samples, |min| min.min(wall_samples)));
            if wall_samples < 30 {
                blockers.push(format!(
                    "100x source artifact `{artifact}` case `{case_id}` has {wall_samples} wall_ns sample(s), needs at least 30"
                ));
            }
            let baseline_wall_samples = metrics
                .and_then(|metrics| suite_metric_samples(metrics.get("baseline_wall_ns")))
                .unwrap_or(0);
            min_baseline_wall_samples = Some(
                min_baseline_wall_samples
                    .map_or(baseline_wall_samples, |min| min.min(baseline_wall_samples)),
            );
            if baseline_wall_samples < 30 {
                blockers.push(format!(
                    "100x source artifact `{artifact}` case `{case_id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
                ));
            }
            record_required_metric_percentile(
                &mut min_wall_p50,
                metrics,
                "wall_ns",
                "p50",
                &mut blockers,
                case_id,
            );
            record_required_metric_percentile(
                &mut min_wall_p95,
                metrics,
                "wall_ns",
                "p95",
                &mut blockers,
                case_id,
            );
            record_required_metric_percentile(
                &mut min_wall_p99,
                metrics,
                "wall_ns",
                "p99",
                &mut blockers,
                case_id,
            );
            record_required_metric_percentile(
                &mut min_baseline_wall_p50,
                metrics,
                "baseline_wall_ns",
                "p50",
                &mut blockers,
                case_id,
            );
            record_required_metric_percentile(
                &mut min_baseline_wall_p95,
                metrics,
                "baseline_wall_ns",
                "p95",
                &mut blockers,
                case_id,
            );
            record_required_metric_percentile(
                &mut min_baseline_wall_p99,
                metrics,
                "baseline_wall_ns",
                "p99",
                &mut blockers,
                case_id,
            );
            let case_backend = case
                .get("backend_id")
                .and_then(Value::as_str)
                .or(report_backend);
            if case_backend
                .is_some_and(|backend| suite_case_has_cpu_sota_contract(case, backend, 100.0))
            {
                contract_case_count += 1;
                let contract_passed = case
                    .get("performance")
                    .and_then(|performance| performance.get("contract_passed"))
                    .and_then(Value::as_bool)
                    == Some(true);
                let speedup_passed = case
                    .get("performance")
                    .and_then(|performance| performance.get("speedup_x"))
                    .and_then(Value::as_f64)
                    .is_some_and(|speedup| speedup >= 100.0);
                if crate::benchmark_evidence_semantics::benchmark_case_passes_summary_evidence(case)
                    && contract_passed
                    && speedup_passed
                {
                    passing_contract_case_count += 1;
                }
            }
        }
        cases.extend(report_cases.iter().cloned());
    }
    if artifacts.len() < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {} source artifact(s); release requires at least {} CPU-SOTA 100x workload families",
            artifacts.len(),
            MIN_CPU_SOTA_100X_RELEASE_CASES
        ));
    }
    if cases.len() < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {} benchmark case(s); release requires at least {}",
            cases.len(),
            MIN_CPU_SOTA_100X_RELEASE_CASES
        ));
    }
    if contract_case_count < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {contract_case_count} CPU-SOTA 100x contract case(s); release requires at least {MIN_CPU_SOTA_100X_RELEASE_CASES}"
        ));
    }
    if passing_contract_case_count < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {passing_contract_case_count} passing CPU-SOTA 100x case(s); release requires at least {MIN_CPU_SOTA_100X_RELEASE_CASES}"
        ));
    }
    let missing_required_cases = REQUIRED_CPU_SOTA_100X_CASES
        .iter()
        .copied()
        .filter(|required| !observed_required_cases.contains(*required))
        .collect::<Vec<_>>();
    for required in &missing_required_cases {
        blockers.push(format!(
            "100x proof is missing required release-defining case `{required}`"
        ));
    }
    let aggregate_failed = cases.len().saturating_sub(passing_contract_case_count);
    let evidence = json!({
        "schema_version": 1,
        "selected_backend": "cuda",
        "environment": environment,
        "git": git,
        "source_fingerprint": source_fingerprint,
        "source_tree_fingerprint": source_tree_fingerprint,
        "source_artifacts": artifacts,
        "source_artifact_count": artifacts.len(),
        "required_cpu_sota_100x_cases": REQUIRED_CPU_SOTA_100X_CASES,
        "missing_required_cpu_sota_100x_cases": missing_required_cases,
        "cpu_sota_100x_contract_case_count": contract_case_count,
        "cpu_sota_100x_passing_case_count": passing_contract_case_count,
        "min_wall_samples": min_wall_samples,
        "min_wall_p50": min_wall_p50,
        "min_wall_p95": min_wall_p95,
        "min_wall_p99": min_wall_p99,
        "min_baseline_wall_samples": min_baseline_wall_samples,
        "min_baseline_wall_p50": min_baseline_wall_p50,
        "min_baseline_wall_p95": min_baseline_wall_p95,
        "min_baseline_wall_p99": min_baseline_wall_p99,
        "summary": {
            "total_cases": cases.len(),
            "passed": passing_contract_case_count,
            "failed": aggregate_failed,
            "total_time_ns": 0,
            "cache_hit_rate": null,
        },
        "cases": cases,
        "blockers": blockers,
    });
    write_json(
        &workspace_root.join("release/evidence/benchmarks/cpu-only-100x-proof.json"),
        &evidence,
    );
}

pub(super) fn read_text_bounded(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    use std::io::Read as _;

    let mut file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("release benchmark evidence exceeds {max_bytes} byte limit"),
        ));
    }
    let mut text = String::with_capacity(metadata.len() as usize);
    file.by_ref()
        .take(max_bytes + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "release benchmark evidence exceeded bounded read limit",
        ));
    }
    Ok(text)
}

pub(super) fn run_workload_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) -> Result<(), String> {
    let mut owned_args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-bench".to_string(),
        "--quiet".to_string(),
        "--".to_string(),
        "run".to_string(),
        "--suite".to_string(),
        "release".to_string(),
        "--case".to_string(),
        case_id.to_string(),
        "--backend".to_string(),
        backend.to_string(),
        "--enforce-budgets".to_string(),
        "--output".to_string(),
        output.to_string(),
        "--sample-timeout-secs".to_string(),
        sample_timeout_secs.to_string(),
    ];
    if let Some(samples) = measured_samples {
        owned_args.push("--measured-samples".to_string());
        owned_args.push(samples.to_string());
    }
    let borrowed = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
    run_command_status(workspace_root, &borrowed)
}

pub(super) fn prefixed_benchmark_artifact(path: &str, prefix: &str) -> String {
    let path = Path::new(path);
    let Some(file_name) = path.file_name().and_then(|file| file.to_str()) else {
        return format!("{prefix}-{path}", path = path.display());
    };
    let file_name = format!("{prefix}-{file_name}");
    path.parent()
        .map(|parent| parent.join(&file_name).display().to_string())
        .unwrap_or(file_name)
}

pub(super) fn write_backend_suite(
    workspace_root: &Path,
    backend: &str,
    artifact_inputs: Vec<BackendSuiteArtifactInput>,
) {
    let output = backend_suite_output_path(backend);
    let mut blockers = Vec::new();
    if artifact_inputs.is_empty() {
        blockers.push(format!(
            "backend `{backend}` release suite has zero artifacts"
        ));
    }
    let artifacts = artifact_inputs
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    let artifact_statuses = artifact_inputs
        .iter()
        .map(|artifact| inspect_backend_suite_artifact(workspace_root, backend, artifact))
        .inspect(|status| {
            blockers.extend(status.blockers.iter().map(|blocker| {
                format!(
                    "backend `{backend}` release suite artifact `{}`: {blocker}",
                    status.path
                )
            }));
        })
        .collect::<Vec<_>>();
    let evidence = BackendSuiteEvidence {
        schema_version: 2,
        backend: backend.to_string(),
        family_count: artifact_inputs.len(),
        artifacts,
        artifact_statuses,
        blockers,
    };
    let path = workspace_root.join(output);
    if let Some(parent) = path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    let json = match serde_json::to_string_pretty(&evidence) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize backend suite evidence: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(&path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn backend_suite_output_path(backend: &str) -> String {
    match backend {
        "cuda" => "release/evidence/benchmarks/cuda-release-suite.json".to_string(),
        "wgpu" => "release/evidence/benchmarks/wgpu-fallback-suite.json".to_string(),
        other => format!("release/evidence/benchmarks/{other}-release-suite.json"),
    }
}

pub(super) fn inspect_backend_suite_artifact(
    workspace_root: &Path,
    backend: &str,
    artifact: &BackendSuiteArtifactInput,
) -> BackendSuiteArtifact {
    let path = workspace_root.join(&artifact.path);
    let (exists, bytes, read_error) = match fs::metadata(&path) {
        Ok(metadata) => (metadata.is_file(), metadata.len(), None),
        Err(error) => {
            let label = if error.kind() == std::io::ErrorKind::NotFound {
                "missing".to_string()
            } else {
                format!("unreadable metadata: {error}")
            };
            (false, 0, Some(label))
        }
    };
    let mut blockers = Vec::new();
    if let Some(error) = &read_error {
        blockers.push(error.clone());
    }
    if !exists {
        if read_error.is_none() {
            blockers.push("not a file".to_string());
        }
        return BackendSuiteArtifact {
            path: artifact.path.clone(),
            family_id: artifact.family_id.clone(),
            requested_case_id: artifact.requested_case_id.clone(),
            exists,
            bytes,
            read_error,
            source_fingerprint: None,
            source_tree_fingerprint: None,
            selected_backend: None,
            host_cpu_model: None,
            gpu_model: None,
            gpu_memory_total_mib: None,
            gpu_compute_capability_major: None,
            gpu_compute_capability_minor: None,
            nvidia_driver_version: None,
            nvidia_cuda_version: None,
            min_cuda_ptx_source_cache_entries: None,
            min_cuda_ptx_source_cache_hits: None,
            min_cuda_ptx_source_cache_misses: None,
            min_kernel_launches: None,
            case_count: 0,
            failed_count: None,
            nonmatching_case_backend_count: 0,
            min_wall_samples: None,
            min_wall_p50: None,
            min_wall_p95: None,
            min_wall_p99: None,
            min_baseline_wall_samples: None,
            min_baseline_wall_p50: None,
            min_baseline_wall_p95: None,
            min_baseline_wall_p99: None,
            cpu_sota_100x_required: artifact.cpu_sota_100x_required,
            cpu_sota_100x_contract_cases: 0,
            cpu_sota_100x_passing_cases: 0,
            blockers,
        };
    }
    if bytes == 0 {
        blockers.push("empty".to_string());
    }
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!("unreadable JSON: {error}"));
            String::new()
        }
    };
    let report = if text.is_empty() {
        Value::Null
    } else {
        match serde_json::from_str::<Value>(&text) {
            Ok(report) => report,
            Err(error) => {
                blockers.push(format!("invalid JSON: {error}"));
                Value::Null
            }
        }
    };
    let selected_backend = report
        .get("selected_backend")
        .and_then(Value::as_str)
        .map(str::to_string);
    let source_fingerprint = report
        .get("source_fingerprint")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            report
                .get("git")
                .and_then(|git| git.get("commit"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(|commit| format!("git:{commit}"))
        });
    let source_tree_fingerprint = report
        .get("source_tree_fingerprint")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if selected_backend.as_deref() != Some(backend) {
        blockers.push(format!(
            "selected_backend `{:?}` does not match requested backend `{backend}`",
            selected_backend
        ));
    }
    let environment = report.get("environment");
    let first_gpu = environment
        .and_then(|environment| environment.get("gpu_devices"))
        .and_then(Value::as_array)
        .and_then(|devices| devices.first());
    let gpu_model = first_gpu
        .and_then(|device| device.get("name"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let gpu_memory_total_mib = first_gpu
        .and_then(|device| device.get("memory_total_mib"))
        .and_then(Value::as_u64);
    let gpu_compute_capability_major = first_gpu
        .and_then(|device| device.get("compute_capability_major"))
        .and_then(Value::as_u64);
    let gpu_compute_capability_minor = first_gpu
        .and_then(|device| device.get("compute_capability_minor"))
        .and_then(Value::as_u64);
    let host_cpu_model = environment
        .and_then(|environment| {
            environment
                .get("host_cpu_model")
                .or_else(|| environment.get("cpu_model"))
                .or_else(|| environment.get("host_cpu"))
        })
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let nvidia_driver_version = environment
        .and_then(|environment| environment.get("nvidia_driver_version"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let nvidia_cuda_version = environment
        .and_then(|environment| environment.get("nvidia_cuda_version"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if backend == "cuda" {
        if gpu_model.as_deref().is_none_or(str::is_empty) {
            blockers.push("CUDA artifact has no nvidia-smi GPU model provenance".to_string());
        }
        if nvidia_driver_version.as_deref().is_none_or(str::is_empty) {
            blockers.push(
                "CUDA artifact has no nvidia-smi NVIDIA driver version provenance".to_string(),
            );
        }
        if nvidia_cuda_version.as_deref().is_none_or(str::is_empty) {
            blockers.push(
                "CUDA artifact has no nvidia-smi CUDA runtime version provenance".to_string(),
            );
        }
        match gpu_memory_total_mib {
            Some(mib) if mib >= MIN_CUDA_RELEASE_MEMORY_MIB => {}
            Some(mib) => blockers.push(format!(
                "CUDA artifact GPU memory is {mib} MiB, below release floor {MIN_CUDA_RELEASE_MEMORY_MIB} MiB"
            )),
            None => blockers.push("CUDA artifact has no nvidia-smi GPU memory provenance".to_string()),
        }
        match (gpu_compute_capability_major, gpu_compute_capability_minor) {
            (Some(major), Some(minor))
                if (major, minor)
                    >= (
                        MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR,
                        MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR,
                    ) => {}
            (Some(major), Some(minor)) => blockers.push(format!(
                "CUDA artifact compute capability is {major}.{minor}, below release floor {MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR}.{MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR}"
            )),
            _ => blockers.push(
                "CUDA artifact has no nvidia-smi compute capability provenance".to_string(),
            ),
        }
    }
    let summary_failed_count = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64);
    if summary_failed_count != Some(0) {
        blockers.push(format!(
            "summary.failed is `{:?}`, expected 0",
            summary_failed_count
        ));
    }
    let cases = report
        .get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if cases.is_empty() {
        blockers.push("cases array is empty or missing".to_string());
    }
    if let Some(mismatch) =
        crate::benchmark_evidence_semantics::benchmark_report_summary_case_evidence_mismatch(
            &report,
        )
    {
        blockers.push(format!("benchmark summary is invalid: {mismatch}"));
    }
    let mut case_failed_count = 0u64;
    let mut nonmatching_case_backend_count = 0usize;
    let mut min_wall_samples = None::<u64>;
    let mut min_baseline_wall_samples = None::<u64>;
    let mut min_wall_p50 = None::<u64>;
    let mut min_wall_p95 = None::<u64>;
    let mut min_wall_p99 = None::<u64>;
    let mut min_baseline_wall_p50 = None::<u64>;
    let mut min_baseline_wall_p95 = None::<u64>;
    let mut min_baseline_wall_p99 = None::<u64>;
    let mut min_cuda_ptx_source_cache_entries = None::<u64>;
    let mut min_cuda_ptx_source_cache_hits = None::<u64>;
    let mut min_cuda_ptx_source_cache_misses = None::<u64>;
    let mut min_kernel_launches = None::<u64>;
    let mut cpu_sota_100x_contract_cases = 0usize;
    let mut cpu_sota_100x_passing_cases = 0usize;
    let mut requested_case_present = false;
    for case in &cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        let case_failure_reason =
            crate::benchmark_evidence_semantics::benchmark_case_failure_reason(case);
        if let Some(reason) = &case_failure_reason {
            case_failed_count += 1;
            blockers.push(format!("case `{case_id}` failed: {reason}"));
        }
        if case_id == artifact.requested_case_id {
            requested_case_present = true;
        }
        if case.get("backend_id").and_then(Value::as_str) != Some(backend) {
            nonmatching_case_backend_count += 1;
        }
        let metrics = case.get("metrics").and_then(Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| suite_metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        min_wall_samples = Some(min_wall_samples.map_or(wall_samples, |min| min.min(wall_samples)));
        if wall_samples < 30 {
            blockers.push(format!(
                "case `{case_id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| suite_metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        min_baseline_wall_samples = Some(
            min_baseline_wall_samples
                .map_or(baseline_wall_samples, |min| min.min(baseline_wall_samples)),
        );
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "case `{case_id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        record_required_metric_percentile(
            &mut min_wall_p50,
            metrics,
            "wall_ns",
            "p50",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_wall_p95,
            metrics,
            "wall_ns",
            "p95",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_wall_p99,
            metrics,
            "wall_ns",
            "p99",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p50,
            metrics,
            "baseline_wall_ns",
            "p50",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p95,
            metrics,
            "baseline_wall_ns",
            "p95",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p99,
            metrics,
            "baseline_wall_ns",
            "p99",
            &mut blockers,
            case_id,
        );
        if matches!(backend, "cuda" | "wgpu") {
            record_required_metric_percentile(
                &mut min_kernel_launches,
                metrics,
                "kernel_launches",
                "p50",
                &mut blockers,
                case_id,
            );
        }
        if backend == "cuda" {
            record_required_metric_percentile(
                &mut min_cuda_ptx_source_cache_entries,
                metrics,
                "cuda_ptx_source_cache_entries",
                "p50",
                &mut blockers,
                case_id,
            );
            record_observed_metric_percentile(
                &mut min_cuda_ptx_source_cache_hits,
                metrics,
                "cuda_ptx_source_cache_hits",
                "p50",
                &mut blockers,
                case_id,
            );
            record_observed_metric_percentile(
                &mut min_cuda_ptx_source_cache_misses,
                metrics,
                "cuda_ptx_source_cache_misses",
                "p50",
                &mut blockers,
                case_id,
            );
        }
        let has_100x_contract = suite_case_has_cpu_sota_contract(case, backend, 100.0);
        if has_100x_contract {
            cpu_sota_100x_contract_cases += 1;
            let contract_passed = case
                .get("performance")
                .and_then(|performance| performance.get("contract_passed"))
                .and_then(Value::as_bool)
                == Some(true);
            let speedup_passed = case
                .get("performance")
                .and_then(|performance| performance.get("speedup_x"))
                .and_then(Value::as_f64)
                .is_some_and(|speedup| speedup >= 100.0);
            if crate::benchmark_evidence_semantics::benchmark_case_passes_summary_evidence(case)
                && contract_passed
                && speedup_passed
            {
                cpu_sota_100x_passing_cases += 1;
            }
        }
    }
    if !cases.is_empty() && summary_failed_count != Some(case_failed_count) {
        blockers.push(format!(
            "summary.failed is `{:?}` but case evidence reports {case_failed_count} failed case(s)",
            summary_failed_count
        ));
    }
    let failed_count = (!cases.is_empty())
        .then_some(case_failed_count)
        .or(summary_failed_count);
    if nonmatching_case_backend_count > 0 {
        blockers.push(format!(
            "{nonmatching_case_backend_count} case(s) do not match requested backend `{backend}`"
        ));
    }
    if !requested_case_present {
        blockers.push(format!(
            "requested case `{}` is absent from artifact cases",
            artifact.requested_case_id
        ));
    }
    if artifact.cpu_sota_100x_required && cpu_sota_100x_contract_cases == 0 {
        blockers.push("CPU-SOTA 100x workload artifact has no 100x contract case".to_string());
    }
    if artifact.cpu_sota_100x_required && cpu_sota_100x_passing_cases == 0 {
        blockers.push("CPU-SOTA 100x workload artifact has no passing 100x case".to_string());
    }
    BackendSuiteArtifact {
        path: artifact.path.clone(),
        family_id: artifact.family_id.clone(),
        requested_case_id: artifact.requested_case_id.clone(),
        exists,
        bytes,
        read_error,
        source_fingerprint,
        source_tree_fingerprint,
        selected_backend,
        host_cpu_model,
        gpu_model,
        gpu_memory_total_mib,
        gpu_compute_capability_major,
        gpu_compute_capability_minor,
        nvidia_driver_version,
        nvidia_cuda_version,
        min_cuda_ptx_source_cache_entries,
        min_cuda_ptx_source_cache_hits,
        min_cuda_ptx_source_cache_misses,
        min_kernel_launches,
        case_count: cases.len(),
        failed_count,
        nonmatching_case_backend_count,
        min_wall_samples,
        min_wall_p50,
        min_wall_p95,
        min_wall_p99,
        min_baseline_wall_samples,
        min_baseline_wall_p50,
        min_baseline_wall_p95,
        min_baseline_wall_p99,
        cpu_sota_100x_required: artifact.cpu_sota_100x_required,
        cpu_sota_100x_contract_cases,
        cpu_sota_100x_passing_cases,
        blockers,
    }
}

pub(super) fn suite_metric_samples(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(|metric| metric.get("samples"))
        .and_then(Value::as_u64)
}

pub(super) fn suite_metric_percentile(value: Option<&Value>, percentile: &str) -> Option<u64> {
    value
        .and_then(|metric| metric.get(percentile))
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .and_then(|metric| metric.get(percentile))
                .and_then(Value::as_f64)
                .filter(|value| *value >= 0.0)
                .map(|value| value as u64)
        })
}

pub(super) fn record_required_metric_percentile(
    current_min: &mut Option<u64>,
    metrics: Option<&serde_json::Map<String, Value>>,
    metric_name: &str,
    percentile: &str,
    blockers: &mut Vec<String>,
    case_id: &str,
) {
    match metrics.and_then(|metrics| suite_metric_percentile(metrics.get(metric_name), percentile))
    {
        Some(value) if value > 0 => {
            *current_min = Some(current_min.map_or(value, |min| min.min(value)));
        }
        _ => blockers.push(format!(
            "case `{case_id}` must include positive {percentile} {metric_name}"
        )),
    }
}

pub(super) fn record_observed_metric_percentile(
    current_min: &mut Option<u64>,
    metrics: Option<&serde_json::Map<String, Value>>,
    metric_name: &str,
    percentile: &str,
    blockers: &mut Vec<String>,
    case_id: &str,
) {
    match metrics.and_then(|metrics| suite_metric_percentile(metrics.get(metric_name), percentile))
    {
        Some(value) => {
            *current_min = Some(current_min.map_or(value, |min| min.min(value)));
        }
        None => blockers.push(format!(
            "case `{case_id}` must include {percentile} {metric_name}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    #[test]
    fn wgpu_suite_output_matches_release_gate_contract() {
        assert_eq!(
            backend_suite_output_path("wgpu"),
            "release/evidence/benchmarks/wgpu-fallback-suite.json",
            "Fix: release-benchmarks must regenerate the WGPU suite artifact consumed by the release gate and completion audit."
        );
    }

    #[test]
    fn cuda_suite_output_matches_release_gate_contract() {
        assert_eq!(
            backend_suite_output_path("cuda"),
            "release/evidence/benchmarks/cuda-release-suite.json",
            "Fix: release-benchmarks must regenerate the CUDA suite artifact consumed by the release gate and completion audit."
        );
    }

    #[test]
    fn write_wgpu_suite_regenerates_gated_fallback_artifact() {
        let dir = TempDir::new().expect("Fix: create a temporary workspace for suite output test.");

        write_backend_suite(dir.path(), "wgpu", Vec::new());

        let fallback = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-fallback-suite.json");
        let comparison = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-comparison-suite.json");
        assert!(
            fallback.exists(),
            "Fix: WGPU release benchmark generation must write the suite artifact consumed by the release gate."
        );
        assert!(
            !comparison.exists(),
            "Fix: WGPU release benchmark generation must not write the stale comparison suite path instead of the gated fallback suite."
        );
        let text = fs::read_to_string(&fallback)
            .expect("Fix: read generated WGPU fallback suite JSON for contract assertions.");
        let suite = serde_json::from_str::<Value>(&text)
            .expect("Fix: generated WGPU fallback suite JSON must be parseable.");
        assert_eq!(
            suite.get("backend").and_then(Value::as_str),
            Some("wgpu"),
            "Fix: generated WGPU fallback suite must retain backend provenance."
        );
    }

    #[test]
    fn failed_suite_artifact_blocker_preserves_case_failure_reason() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for failed suite artifact test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-workload-failed.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create failed suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "wgpu",
                "summary": {
                    "total_cases": 1,
                    "passed": 0,
                    "failed": 1,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "sparse.compaction.count.1m",
                        "backend_id": "wgpu",
                        "status": "failed",
                        "correctness": {
                            "Invalid": {
                                "reason": "Performance contract failed: sparse output compaction count requires 100.00x over optimized CPU fired-rule collection over predicate masks, observed 91.75x"
                            }
                        },
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002},
                            "kernel_launches": {"samples": 1, "p50": 1}
                        },
                        "contract": {
                            "primitive": "sparse output compaction count",
                            "baselines": [
                                {
                                    "name": "optimized CPU fired-rule collection over predicate masks",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["cuda", "wgpu"]
                                }
                            ]
                        },
                        "performance": null,
                        "optimization_passes_applied": ["wgpu-release-path"]
                    }
                ]
            }))
            .expect("Fix: serialize failed benchmark artifact JSON."),
        )
        .expect("Fix: write failed benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "wgpu",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "sparse-output-compaction".to_string(),
                requested_case_id: "sparse.compaction.count.1m".to_string(),
                cpu_sota_100x_required: false,
            },
        );

        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "case `sparse.compaction.count.1m` failed: Performance contract failed"
            ) && blocker.contains("observed 91.75x")),
            "Fix: WGPU suite blockers must preserve the benchmark case failure reason instead of exposing only missing metric fallout: {:?}",
            status.blockers
        );
    }

    #[test]
    fn suite_artifact_status_recomputes_hidden_case_failures() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for hidden suite failure test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-hidden-invalid.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create hidden suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "wgpu",
                "summary": {
                    "total_cases": 1,
                    "passed": 1,
                    "failed": 0,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "correctness": {
                            "Invalid": {
                                "reason": "CUDA/WGPU output mismatch at row 17"
                            }
                        },
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002},
                            "kernel_launches": {"samples": 1, "p50": 1}
                        },
                        "contract": {
                            "primitive": "release condition eval",
                            "baselines": [
                                {
                                    "name": "CPU-SOTA",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["wgpu"]
                                }
                            ]
                        },
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize hidden-invalid WGPU benchmark artifact JSON."),
        )
        .expect("Fix: write hidden-invalid WGPU benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "wgpu",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: true,
            },
        );

        assert_eq!(
            status.failed_count,
            Some(1),
            "Fix: backend suite status rows must derive failed_count from case evidence, not stale summary.failed."
        );
        assert_eq!(
            status.cpu_sota_100x_contract_cases, 1,
            "Fix: hidden invalid correctness must not erase the applicable CPU-SOTA contract count."
        );
        assert_eq!(
            status.cpu_sota_100x_passing_cases, 0,
            "Fix: hidden invalid correctness must disqualify a case from passing CPU-SOTA status proof."
        );
        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "case `release.condition_eval.1m` failed: CUDA/WGPU output mismatch at row 17"
            )),
            "Fix: backend suite blockers must preserve hidden case failure reasons; blockers={:?}",
            status.blockers
        );
        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "summary.failed is `Some(0)` but case evidence reports 1 failed case(s)"
            )),
            "Fix: backend suite blockers must expose stale summary.failed drift; blockers={:?}",
            status.blockers
        );
    }

    #[test]
    fn suite_artifact_status_rejects_stale_summary_passed_count() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for stale suite summary test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-stale-passed.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create stale summary suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "wgpu",
                "summary": {
                    "total_cases": 1,
                    "passed": 0,
                    "failed": 0,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002},
                            "kernel_launches": {"samples": 1, "p50": 1}
                        },
                        "contract": {
                            "primitive": "release condition eval",
                            "baselines": [
                                {
                                    "name": "CPU-SOTA",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["wgpu"]
                                }
                            ]
                        },
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize stale-passed WGPU benchmark artifact JSON."),
        )
        .expect("Fix: write stale-passed WGPU benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "wgpu",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: true,
            },
        );

        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "benchmark summary is invalid: summary total/pass/fail (Some(1)/Some(0)/Some(0)) contradicts case evidence (1/1/0)"
            )),
            "Fix: backend suite inspector must reject stale summary.passed drift before suite rows prove release evidence; blockers={:?}",
            status.blockers
        );
    }

    #[test]
    fn suite_artifact_status_rejects_unproven_cpu_sota_pass_status() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for unproven CPU-SOTA suite test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-unproven-pass.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create unproven CPU-SOTA suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "wgpu",
                "summary": {
                    "total_cases": 1,
                    "passed": 0,
                    "failed": 1,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002},
                            "kernel_launches": {"samples": 1, "p50": 1}
                        },
                        "contract": {
                            "primitive": "release condition eval",
                            "baselines": [
                                {
                                    "name": "CPU-SOTA",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["wgpu"]
                                }
                            ]
                        },
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize unproven CPU-SOTA WGPU benchmark artifact JSON."),
        )
        .expect("Fix: write unproven CPU-SOTA WGPU benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "wgpu",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: true,
            },
        );

        assert_eq!(
            status.failed_count,
            Some(1),
            "Fix: missing pass status must count as a failed suite artifact case."
        );
        assert_eq!(
            status.cpu_sota_100x_contract_cases, 1,
            "Fix: missing pass status must not erase the applicable CPU-SOTA contract count."
        );
        assert_eq!(
            status.cpu_sota_100x_passing_cases, 0,
            "Fix: CPU-SOTA passing suite rows must require explicit pass status evidence."
        );
        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "case `release.condition_eval.1m` failed: missing pass status"
            )),
            "Fix: unproven CPU-SOTA suite rows must expose the missing pass status reason; blockers={:?}",
            status.blockers
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_case_failure_hidden_by_passing_contract() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CPU-SOTA proof regression test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-hidden-invalid.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA proof artifact path must have a parent directory."),
        )
        .expect("Fix: create CPU-SOTA proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": {
                    "total_cases": 1,
                    "passed": 1,
                    "failed": 0,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "correctness": {
                            "Invalid": {
                                "reason": "CUDA/WGPU output mismatch at row 17"
                            }
                        },
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": {
                            "primitive": "release condition eval",
                            "baselines": [
                                {
                                    "name": "CPU-SOTA",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["cuda"]
                                }
                            ]
                        },
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize hidden-invalid CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write hidden-invalid CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");

        assert_eq!(
            proof
                .get("cpu_sota_100x_passing_case_count")
                .and_then(Value::as_u64),
            Some(0),
            "Fix: invalid correctness evidence must disqualify a case from passing CPU-SOTA proof even when performance says contract_passed=true."
        );
        assert_eq!(
            proof
                .get("summary")
                .and_then(|summary| summary.get("failed"))
                .and_then(Value::as_u64),
            Some(1),
            "Fix: aggregate CPU-SOTA proof summary must count hidden invalid cases as failed."
        );
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");
        assert!(
            blockers
                .iter()
                .filter_map(Value::as_str)
                .any(|blocker| blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-hidden-invalid.json` case `release.condition_eval.1m` failed: CUDA/WGPU output mismatch at row 17"
                )),
            "Fix: aggregate CPU-SOTA proof blockers must preserve hidden case failure reasons; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_missing_pass_status_with_passing_contract() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for missing-status CPU-SOTA proof test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-missing-status.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA proof artifact path must have a parent directory."),
        )
        .expect("Fix: create missing-status CPU-SOTA proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": {
                    "total_cases": 1,
                    "passed": 0,
                    "failed": 1,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": {
                            "primitive": "release condition eval",
                            "baselines": [
                                {
                                    "name": "CPU-SOTA",
                                    "crate_name": "vyre-runtime",
                                    "class": "CpuSota",
                                    "min_speedup_x": 100.0,
                                    "backend_ids": ["cuda"]
                                }
                            ]
                        },
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize missing-status CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write missing-status CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");

        assert_eq!(
            proof
                .get("cpu_sota_100x_contract_case_count")
                .and_then(Value::as_u64),
            Some(1),
            "Fix: missing pass status must not erase applicable CPU-SOTA contracts from aggregate proof."
        );
        assert_eq!(
            proof
                .get("cpu_sota_100x_passing_case_count")
                .and_then(Value::as_u64),
            Some(0),
            "Fix: aggregate CPU-SOTA proof must require explicit pass status before counting a passing 100x case."
        );
        assert_eq!(
            proof
                .get("summary")
                .and_then(|summary| summary.get("failed"))
                .and_then(Value::as_u64),
            Some(1),
            "Fix: aggregate CPU-SOTA proof summary must count missing pass status cases as failed."
        );
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");
        assert!(
            blockers
                .iter()
                .filter_map(Value::as_str)
                .any(|blocker| blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-missing-status.json` case `release.condition_eval.1m` failed: missing pass status"
                )),
            "Fix: aggregate CPU-SOTA proof blockers must expose missing pass status; blockers={blockers:?}"
        );
    }
}
