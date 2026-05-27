//! Generate long-running release benchmark evidence artifacts.
//!
//! `release-evidence` intentionally avoids expensive benchmark runs.
//! This command is the explicit release path for producing the per
//! workload benchmark JSON artifacts listed by `release-matrix`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const REQUIRED_CPU_SOTA_100X_CASES: &[&str] = &[
    "release.condition_eval.1m",
    "release.string_bitmap_scatter.1m",
    "release.offset_count_aggregation.1m",
    "release.entropy_window.1m",
    "release.quantified_condition_loops.1m",
    "release.alias_reaching_def.1m",
    "release.ifds_witness.1m",
    "release.c_ast_traversal.1m",
    "release.megakernel_queue.1m",
    "release.egraph_saturation.1m",
    "sparse.compaction.count.1m",
];
const MIN_CPU_SOTA_100X_RELEASE_CASES: usize = 10;
const MAX_RELEASE_BENCHMARK_TEXT_BYTES: u64 = 256 * 1024 * 1024;
const MIN_CUDA_RELEASE_MEMORY_MIB: u64 = 16 * 1024;
const MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR: u64 = 8;
const MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR: u64 = 0;

#[derive(Debug, Deserialize)]
struct ReleaseWorkloadMatrix {
    families: Vec<ReleaseWorkloadFamily>,
}

#[derive(Debug, Deserialize)]
struct ReleaseWorkloadFamily {
    id: String,
    required: bool,
    matched_cases: Vec<String>,
    evidence_artifact: String,
    #[serde(default)]
    max_cpu_sota_min_speedup_x: Option<f64>,
    #[serde(default)]
    cpu_sota_100x_cases: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BackendSuiteEvidence {
    schema_version: u32,
    backend: String,
    family_count: usize,
    artifacts: Vec<String>,
    artifact_statuses: Vec<BackendSuiteArtifact>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BackendSuiteArtifact {
    path: String,
    family_id: String,
    requested_case_id: String,
    exists: bool,
    bytes: u64,
    read_error: Option<String>,
    source_fingerprint: Option<String>,
    selected_backend: Option<String>,
    host_cpu_model: Option<String>,
    gpu_model: Option<String>,
    gpu_memory_total_mib: Option<u64>,
    gpu_compute_capability_major: Option<u64>,
    gpu_compute_capability_minor: Option<u64>,
    nvidia_driver_version: Option<String>,
    nvidia_cuda_version: Option<String>,
    min_cuda_ptx_source_cache_entries: Option<u64>,
    min_cuda_ptx_source_cache_hits: Option<u64>,
    min_cuda_ptx_source_cache_misses: Option<u64>,
    case_count: usize,
    failed_count: Option<u64>,
    nonmatching_case_backend_count: usize,
    min_wall_samples: Option<u64>,
    min_wall_p50: Option<u64>,
    min_wall_p95: Option<u64>,
    min_wall_p99: Option<u64>,
    min_baseline_wall_samples: Option<u64>,
    min_baseline_wall_p50: Option<u64>,
    min_baseline_wall_p95: Option<u64>,
    min_baseline_wall_p99: Option<u64>,
    cpu_sota_100x_required: bool,
    cpu_sota_100x_contract_cases: usize,
    cpu_sota_100x_passing_cases: usize,
    blockers: Vec<String>,
}

#[derive(Debug)]
struct BackendSuiteArtifactInput {
    path: String,
    family_id: String,
    requested_case_id: String,
    cpu_sota_100x_required: bool,
}

#[derive(Debug, Serialize)]
struct ReleaseAxesEvidence {
    schema_version: u32,
    warm_us_per_file: Option<f64>,
    cold_pipeline_build_ms: Option<f64>,
    gbs_scan_throughput: Option<f64>,
    ulp_drift_max: Option<u32>,
    max_vram_mib: Option<u64>,
    source_artifacts: Vec<String>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationBenchmarkManifest {
    schema_version: u32,
    backend: String,
    required_case_count: usize,
    required_pass_families: Vec<&'static str>,
    covered_pass_families: Vec<&'static str>,
    uncovered_pass_families: Vec<&'static str>,
    cases: Vec<OptimizationBenchmarkEvidence>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OptimizationBenchmarkEvidence {
    case_id: &'static str,
    artifact: &'static str,
    covered_pass_families: Vec<&'static str>,
    required_custom_metrics: Vec<&'static str>,
    required_positive_metrics: Vec<&'static str>,
    exists: bool,
    read_error: Option<String>,
    case_count: usize,
    min_wall_samples: Option<u64>,
    min_wall_p50: Option<u64>,
    min_wall_p95: Option<u64>,
    min_wall_p99: Option<u64>,
    min_baseline_wall_samples: Option<u64>,
    min_baseline_wall_p50: Option<u64>,
    min_baseline_wall_p95: Option<u64>,
    min_baseline_wall_p99: Option<u64>,
    min_wall_speedup_x1000: Option<u64>,
    missing_custom_metrics: Vec<String>,
    non_positive_required_metrics: Vec<String>,
    non_winning_cases: Vec<String>,
    blockers: Vec<String>,
}

#[derive(Debug)]
struct OptimizationArtifactInspection {
    exists: bool,
    read_error: Option<String>,
    case_count: usize,
    min_wall_samples: Option<u64>,
    min_wall_p50: Option<u64>,
    min_wall_p95: Option<u64>,
    min_wall_p99: Option<u64>,
    min_baseline_wall_samples: Option<u64>,
    min_baseline_wall_p50: Option<u64>,
    min_baseline_wall_p95: Option<u64>,
    min_baseline_wall_p99: Option<u64>,
    min_wall_speedup_x1000: Option<u64>,
    missing_custom_metrics: Vec<String>,
    non_positive_required_metrics: Vec<String>,
    non_winning_cases: Vec<String>,
    blockers: Vec<String>,
}

pub(crate) fn run(args: &[String]) {
    let config = match parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let matrix_path =
        workspace_root.join("release/evidence/benchmarks/release-workload-matrix.json");
    if let Some(parent) = matrix_path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    run_command(
        &workspace_root,
        &[
            "run",
            "-p",
            "vyre-bench",
            "--quiet",
            "--",
            "release-matrix",
            "--format",
            "json",
            "--output",
            "release/evidence/benchmarks/release-workload-matrix.json",
            "--enforce",
        ],
    );
    let matrix_text = match read_text_bounded(&matrix_path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("Fix: failed to read `{}`: {error}", matrix_path.display());
            std::process::exit(1);
        }
    };
    let matrix = match serde_json::from_str::<ReleaseWorkloadMatrix>(&matrix_text) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("Fix: release workload matrix JSON is invalid: {error}");
            std::process::exit(1);
        }
    };

    let mut suite_artifacts = Vec::new();
    let mut cpu_100x_artifacts = Vec::new();
    let mut ran = 0usize;
    for family in matrix.families.iter().filter(|family| family.required) {
        let cpu_100x_family = config.backend == "cuda"
            && family
                .max_cpu_sota_min_speedup_x
                .is_some_and(|speedup| speedup >= 100.0);
        let selected_cases = if cpu_100x_family && !family.cpu_sota_100x_cases.is_empty() {
            &family.cpu_sota_100x_cases
        } else {
            &family.matched_cases
        };
        let preferred_release_case = selected_cases
            .iter()
            .find(|case_id| REQUIRED_CPU_SOTA_100X_CASES.contains(&case_id.as_str()));
        let Some(case_id) = preferred_release_case.or_else(|| selected_cases.first()) else {
            eprintln!(
                "Fix: required release workload `{}` has no matched benchmark case.",
                family.id
            );
            std::process::exit(1);
        };
        if config.only.as_ref().is_some_and(|only| only != &family.id) {
            continue;
        }
        if !config.reuse_existing
            || !benchmark_artifact_is_reusable(
                &workspace_root,
                &config.backend,
                &family.id,
                case_id,
                &family.evidence_artifact,
                cpu_100x_family,
            )
        {
            run_workload_benchmark(
                &workspace_root,
                case_id,
                &config.backend,
                &family.evidence_artifact,
                config.measured_samples,
                config.sample_timeout_secs,
            );
        }
        if config.backend == "cuda" && family.id == "megakernel-queued-batches" {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-condition-cuda.json",
            );
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-condition-100x-proof.json",
            );
        }
        if cpu_100x_family {
            cpu_100x_artifacts.push(family.evidence_artifact.clone());
        }
        if config.backend == "cuda" && family.id == "megakernel-queued-batches" {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-latency-cuda.json",
            );
        }
        if config.backend == "cuda" && family.id == "alias-reaching-def" {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/dataflow-analysis-release.json",
            );
        }
        suite_artifacts.push(BackendSuiteArtifactInput {
            path: family.evidence_artifact.clone(),
            family_id: family.id.clone(),
            requested_case_id: case_id.clone(),
            cpu_sota_100x_required: cpu_100x_family,
        });
        ran += 1;
    }
    if config.only.is_none() && config.backend == "cuda" && config.include_wgpu_comparison {
        let mut wgpu_artifacts = Vec::new();
        for family in matrix.families.iter().filter(|family| family.required) {
            let Some(case_id) = family.matched_cases.first() else {
                eprintln!(
                    "Fix: required release workload `{}` has no matched benchmark case.",
                    family.id
                );
                std::process::exit(1);
            };
            let output = prefixed_benchmark_artifact(&family.evidence_artifact, "wgpu");
            run_workload_benchmark(
                &workspace_root,
                case_id,
                "wgpu",
                &output,
                config.measured_samples,
                config.sample_timeout_secs,
            );
            wgpu_artifacts.push(BackendSuiteArtifactInput {
                path: output,
                family_id: family.id.clone(),
                requested_case_id: case_id.clone(),
                cpu_sota_100x_required: false,
            });
        }
        write_backend_suite(&workspace_root, "wgpu", wgpu_artifacts);
    }
    if config.only.is_none() {
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.rewrites.impact.corpus",
            &config.backend,
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        copy_artifact(
            &workspace_root,
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            "release/evidence/optimization/pass-family-benchmarks.json",
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "foundation.optimizer.impact",
            &config.backend,
            "release/evidence/optimization/optimizer-impact-cuda.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "cuda.ptx.patterns.release.corpus",
            &config.backend,
            "release/evidence/benchmarks/cuda-ptx-patterns.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.egraph_saturation",
            &config.backend,
            "release/evidence/optimization/egraph-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        copy_artifact(
            &workspace_root,
            "release/evidence/optimization/egraph-before-after.json",
            "release/evidence/benchmarks/egraph-before-after.json",
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.alias_aware_optimizations",
            &config.backend,
            "release/evidence/benchmarks/alias-aware-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_command(
            &workspace_root,
            &[
                "run",
                "--bin",
                "xtask",
                "--quiet",
                "--",
                "optimization-matrix",
                "--output",
                "release/evidence/optimization/optimization-integration-matrix.json",
            ],
        );
        write_optimization_benchmark_manifest(&workspace_root, &config.backend);
    }
    if ran == 0 {
        eprintln!("Fix: release-benchmarks selected zero benchmark families.");
        std::process::exit(1);
    }
    if config.backend == "cuda" {
        write_cpu_100x_proof(&workspace_root, &cpu_100x_artifacts);
    }
    write_backend_suite(&workspace_root, &config.backend, suite_artifacts);
    write_release_axes(&workspace_root);
    println!("release-benchmarks: wrote {ran} benchmark artifact(s)");
}

fn write_cpu_100x_proof(workspace_root: &Path, artifacts: &[String]) {
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
        let Some(report_cases) = report.get("cases").and_then(Value::as_array) else {
            blockers.push(format!(
                "100x source artifact `{artifact}` has no cases array"
            ));
            continue;
        };
        for case in report_cases {
            let case_id = case
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("<unknown>");
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
            if suite_case_has_cpu_sota_contract(case, 100.0) {
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
                if contract_passed && speedup_passed {
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

fn read_text_bounded(path: &Path, max_bytes: u64) -> std::io::Result<String> {
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

fn run_workload_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) {
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
    run_command(workspace_root, &borrowed);
}

fn prefixed_benchmark_artifact(path: &str, prefix: &str) -> String {
    let path = Path::new(path);
    let Some(file_name) = path.file_name().and_then(|file| file.to_str()) else {
        return format!("{prefix}-{path}", path = path.display());
    };
    let file_name = format!("{prefix}-{file_name}");
    path.parent()
        .map(|parent| parent.join(&file_name).display().to_string())
        .unwrap_or(file_name)
}

fn write_backend_suite(
    workspace_root: &Path,
    backend: &str,
    artifact_inputs: Vec<BackendSuiteArtifactInput>,
) {
    let output = match backend {
        "cuda" => "release/evidence/benchmarks/cuda-release-suite.json".to_string(),
        "wgpu" => "release/evidence/benchmarks/wgpu-comparison-suite.json".to_string(),
        other => format!("release/evidence/benchmarks/{other}-release-suite.json"),
    };
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

fn inspect_backend_suite_artifact(
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
    let failed_count = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64);
    if failed_count != Some(0) {
        blockers.push(format!(
            "summary.failed is `{:?}`, expected 0",
            failed_count
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
    let mut cpu_sota_100x_contract_cases = 0usize;
    let mut cpu_sota_100x_passing_cases = 0usize;
    let mut requested_case_present = false;
    for case in &cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
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
        let has_100x_contract = suite_case_has_cpu_sota_contract(case, 100.0);
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
            if contract_passed && speedup_passed {
                cpu_sota_100x_passing_cases += 1;
            }
        }
    }
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

fn suite_metric_samples(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(|metric| metric.get("samples"))
        .and_then(Value::as_u64)
}

fn suite_metric_percentile(value: Option<&Value>, percentile: &str) -> Option<u64> {
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

fn record_required_metric_percentile(
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

fn record_observed_metric_percentile(
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

fn metric_p50(metric: &Value) -> Option<u64> {
    metric.get("p50").and_then(Value::as_u64)
}

fn suite_case_has_cpu_sota_contract(case: &Value, required_speedup: f64) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
            })
        })
}

fn inspect_optimization_benchmark_artifact(
    workspace_root: &Path,
    artifact: &str,
    required_custom_metrics: &[&str],
    required_positive_metrics: &[&str],
) -> OptimizationArtifactInspection {
    let mut blockers = Vec::new();
    let path = workspace_root.join(artifact);
    let (exists, mut read_error) = match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => (true, None),
        Ok(metadata) if metadata.is_file() => {
            blockers.push("empty".to_string());
            (true, None)
        }
        Ok(_) => {
            blockers.push("not a file".to_string());
            (false, None)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            blockers.push("missing".to_string());
            (false, Some(error.to_string()))
        }
        Err(error) => {
            let message = error.to_string();
            blockers.push(format!("unreadable metadata: {error}"));
            (false, Some(message))
        }
    };
    if !blockers.is_empty() {
        return OptimizationArtifactInspection {
            exists,
            read_error,
            case_count: 0,
            min_wall_samples: None,
            min_wall_p50: None,
            min_wall_p95: None,
            min_wall_p99: None,
            min_baseline_wall_samples: None,
            min_baseline_wall_p50: None,
            min_baseline_wall_p95: None,
            min_baseline_wall_p99: None,
            min_wall_speedup_x1000: None,
            missing_custom_metrics: required_custom_metrics
                .iter()
                .map(|metric| (*metric).to_string())
                .collect(),
            non_positive_required_metrics: required_positive_metrics
                .iter()
                .map(|metric| (*metric).to_string())
                .collect(),
            non_winning_cases: Vec::new(),
            blockers,
        };
    }
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(error) => {
            read_error = Some(error.to_string());
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
    let cases = report
        .get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if cases.is_empty() {
        blockers.push("cases array is empty or missing".to_string());
    }
    let mut min_wall_samples = None::<u64>;
    let mut min_baseline_wall_samples = None::<u64>;
    let mut min_wall_p50 = None::<u64>;
    let mut min_wall_p95 = None::<u64>;
    let mut min_wall_p99 = None::<u64>;
    let mut min_baseline_wall_p50 = None::<u64>;
    let mut min_baseline_wall_p95 = None::<u64>;
    let mut min_baseline_wall_p99 = None::<u64>;
    let mut min_wall_speedup_x1000 = None::<u64>;
    let mut missing_custom_metrics = Vec::new();
    let mut non_positive_required_metrics = Vec::new();
    let mut non_winning_cases = Vec::new();
    for case in &cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
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
        match (
            metrics.and_then(|metrics| suite_metric_percentile(metrics.get("wall_ns"), "p50")),
            metrics.and_then(|metrics| {
                suite_metric_percentile(metrics.get("baseline_wall_ns"), "p50")
            }),
        ) {
            (Some(wall), Some(baseline)) if wall > 0 && baseline > wall => {
                let speedup_x1000 = baseline.saturating_mul(1_000) / wall;
                min_wall_speedup_x1000 = Some(
                    min_wall_speedup_x1000.map_or(speedup_x1000, |min| min.min(speedup_x1000)),
                );
            }
            (Some(_), Some(_)) if optimization_semantic_win(case_id, metrics) => {}
            (Some(wall), Some(baseline)) => {
                non_winning_cases.push(format!(
                    "{case_id}:wall_p50={wall}:baseline_wall_p50={baseline}"
                ));
            }
            _ => {
                non_winning_cases.push(format!("{case_id}:missing-wall-or-baseline-p50"));
            }
        }
        for metric in required_custom_metrics {
            if !metrics.is_some_and(|metrics| metrics.contains_key(*metric)) {
                missing_custom_metrics.push(format!("{case_id}:{metric}"));
            }
        }
        for metric in required_positive_metrics {
            let positive = metrics
                .and_then(|metrics| metrics.get(*metric))
                .and_then(metric_p50)
                .is_some_and(|value| value > 0);
            if !positive {
                non_positive_required_metrics.push(format!("{case_id}:{metric}"));
            }
        }
    }
    if !missing_custom_metrics.is_empty() {
        blockers.push(format!(
            "missing required metric(s): {}",
            missing_custom_metrics.join(", ")
        ));
    }
    if !non_positive_required_metrics.is_empty() {
        blockers.push(format!(
            "non-positive required metric(s): {}",
            non_positive_required_metrics.join(", ")
        ));
    }
    if !non_winning_cases.is_empty() {
        blockers.push(format!(
            "optimized wall_ns p50 must beat baseline_wall_ns p50 for every case: {}",
            non_winning_cases.join(", ")
        ));
    }
    OptimizationArtifactInspection {
        exists,
        read_error,
        case_count: cases.len(),
        min_wall_samples,
        min_wall_p50,
        min_wall_p95,
        min_wall_p99,
        min_baseline_wall_samples,
        min_baseline_wall_p50,
        min_baseline_wall_p95,
        min_baseline_wall_p99,
        min_wall_speedup_x1000,
        missing_custom_metrics,
        non_positive_required_metrics,
        non_winning_cases,
        blockers,
    }
}

fn optimization_semantic_win(
    case_id: &str,
    metrics: Option<&serde_json::Map<String, Value>>,
) -> bool {
    let Some(metrics) = metrics else {
        return false;
    };
    match case_id {
        "lower.rewrites.impact.corpus" => {
            suite_metric_percentile(metrics.get("lower_ops_eliminated"), "p50")
                .is_some_and(|value| value > 0)
                || suite_metric_percentile(metrics.get("lower_optimized_issue_score"), "p50")
                    .zip(suite_metric_percentile(
                        metrics.get("lower_baseline_issue_score"),
                        "p50",
                    ))
                    .is_some_and(|(optimized, baseline)| optimized < baseline)
        }
        "foundation.optimizer.impact" => {
            suite_metric_percentile(metrics.get("optimizer_nodes_eliminated"), "p50")
                .is_some_and(|value| value > 0)
        }
        "lower.egraph_saturation" => {
            suite_metric_percentile(metrics.get("egraph_applied_rewrites"), "p50")
                .is_some_and(|value| value > 0)
                && suite_metric_percentile(metrics.get("egraph_output_ops"), "p50")
                    .zip(suite_metric_percentile(
                        metrics.get("egraph_baseline_ops_after"),
                        "p50",
                    ))
                    .is_some_and(|(output, baseline)| output < baseline)
        }
        "lower.alias_aware_optimizations" => {
            suite_metric_percentile(metrics.get("alias_pass_wins"), "p50")
                .is_some_and(|value| value >= 5)
        }
        _ => false,
    }
}

fn write_release_axes(workspace_root: &Path) {
    let evidence_dir = workspace_root.join("release/evidence/benchmarks");
    let mut reports = Vec::new();
    let mut source_artifacts = Vec::new();
    let mut blockers = Vec::new();
    match fs::read_dir(&evidence_dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        blockers.push(format!(
                            "failed to read benchmark evidence directory entry: {error}"
                        ));
                        continue;
                    }
                };
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }
                let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
                    Ok(text) => text,
                    Err(error) => {
                        blockers.push(format!(
                            "failed to read benchmark evidence `{}`: {error}",
                            path.display()
                        ));
                        continue;
                    }
                };
                let value = match serde_json::from_str::<Value>(&text) {
                    Ok(value) => value,
                    Err(error) => {
                        blockers.push(format!(
                            "invalid benchmark evidence JSON `{}`: {error}",
                            path.display()
                        ));
                        continue;
                    }
                };
                if value.get("cases").and_then(Value::as_array).is_none() {
                    continue;
                }
                source_artifacts.push(
                    path.strip_prefix(workspace_root)
                        .unwrap_or(&path)
                        .display()
                        .to_string(),
                );
                reports.push(value);
            }
        }
        Err(error) => blockers.push(format!(
            "failed to read benchmark evidence directory `{}`: {error}",
            evidence_dir.display()
        )),
    }
    blockers.extend(release_axis_blockers(&reports));
    let evidence = ReleaseAxesEvidence {
        schema_version: 1,
        warm_us_per_file: min_metric_p50(&reports, "wall_ns").map(|ns| ns as f64 / 1_000.0),
        cold_pipeline_build_ms: min_first_available_metric_p50(
            &reports,
            &[
                "cold_compile_ns",
                "cold_wall_ns",
                "compile_ns",
                "lower_ns",
                "optimize_ns",
            ],
        )
        .map(|ns| ns as f64 / 1_000_000.0),
        gbs_scan_throughput: max_metric_p50(&reports, "wall_gb_s_x1000")
            .or_else(|| max_metric_p50(&reports, "device_gb_s_x1000"))
            .map(|gb_s_x1000| gb_s_x1000 as f64 / 1_000.0),
        ulp_drift_max: Some(max_observed_ulp(&reports).unwrap_or(0)),
        max_vram_mib: max_vram_mib(&reports),
        source_artifacts,
        blockers,
    };
    write_json(
        &workspace_root.join("release/evidence/benchmarks/bench-release-axes.json"),
        &evidence,
    );
}

fn write_optimization_benchmark_manifest(workspace_root: &Path, backend: &str) {
    let specs = [
        (
            "lower.rewrites.impact.corpus",
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            vec![
                "memory-layout",
                "control-flow",
                "vector-layout",
                "A13-coalesce-fixture",
                "A14-shared-mem-promote-fixture",
                "A15-bank-conflict-fixture",
                "A16-vec-pack-fixture",
            ],
            vec![
                "lower_ops_before",
                "lower_ops_after",
                "lower_ops_eliminated",
                "lower_coalesce_problematic_before",
                "lower_shared_candidates_before",
                "lower_bank_critical_before",
                "lower_vec_pack_chains_before",
                "lower_vec_pack_ops_eliminable_before",
            ],
            vec![
                "lower_ops_before",
                "lower_ops_eliminated",
                "lower_coalesce_problematic_before",
                "lower_shared_candidates_before",
                "lower_bank_critical_before",
                "lower_vec_pack_chains_before",
                "lower_vec_pack_ops_eliminable_before",
            ],
        ),
        (
            "foundation.optimizer.impact",
            "release/evidence/optimization/optimizer-impact-cuda.json",
            vec!["algebraic", "predicate"],
            vec![
                "optimizer_input_nodes",
                "optimizer_output_nodes",
                "optimizer_nodes_eliminated",
            ],
            vec!["optimizer_input_nodes", "optimizer_output_nodes"],
        ),
        (
            "lower.egraph_saturation",
            "release/evidence/optimization/egraph-before-after.json",
            vec!["egraph"],
            vec![
                "egraph_case_count",
                "egraph_bitwise_case_count",
                "egraph_boolean_case_count",
                "egraph_equality_classes",
                "egraph_applied_rewrites",
            ],
            vec![
                "egraph_case_count",
                "egraph_bitwise_case_count",
                "egraph_boolean_case_count",
                "egraph_equality_classes",
                "egraph_applied_rewrites",
            ],
        ),
        (
            "lower.alias_aware_optimizations",
            "release/evidence/benchmarks/alias-aware-before-after.json",
            vec![
                "dataflow-analysis-dse",
                "dataflow-analysis-loop-fusion",
                "dataflow-analysis-loop-fission",
                "dataflow-analysis-licm",
            ],
            vec![
                "alias_pass_wins",
                "alias_fact_count",
                "alias_cross_binding_fact_count",
                "reaching_def_fact_count",
                "alias_total_ops_after",
                "conservative_total_ops_after",
                "alias_dse_store_count",
                "conservative_dse_store_count",
                "alias_stlf_final_value_id",
                "conservative_stlf_final_value_id",
                "alias_licm_loop_loads",
                "conservative_licm_loop_loads",
                "alias_fusion_loop_count",
                "conservative_fusion_loop_count",
                "alias_fission_loop_count",
                "conservative_fission_loop_count",
                "benchmark_repeats",
            ],
            vec![
                "alias_pass_wins",
                "alias_fact_count",
                "alias_cross_binding_fact_count",
                "reaching_def_fact_count",
                "benchmark_repeats",
            ],
        ),
    ];
    let required_pass_families = vec![
        "algebraic",
        "predicate",
        "egraph",
        "memory-layout",
        "control-flow",
        "vector-layout",
        "A13-coalesce-fixture",
        "A14-shared-mem-promote-fixture",
        "A15-bank-conflict-fixture",
        "A16-vec-pack-fixture",
        "dataflow-analysis-dse",
        "dataflow-analysis-loop-fusion",
        "dataflow-analysis-loop-fission",
        "dataflow-analysis-licm",
    ];
    let required_case_count = specs.len();
    let mut blockers = Vec::new();
    let mut covered_pass_families = Vec::new();
    let cases = specs
        .into_iter()
        .map(|(
            case_id,
            artifact,
            pass_families,
            required_custom_metrics,
            required_positive_metrics,
        )| {
            let inspection = inspect_optimization_benchmark_artifact(
                workspace_root,
                artifact,
                &required_custom_metrics,
                &required_positive_metrics,
            );
            if !inspection.exists {
                blockers.push(format!(
                    "required optimization benchmark artifact `{artifact}` for `{case_id}` is missing"
                ));
            }
            blockers.extend(inspection.blockers.iter().map(|blocker| {
                format!("optimization benchmark `{case_id}` artifact `{artifact}`: {blocker}")
            }));
            for family in &pass_families {
                covered_pass_families.push(*family);
            }
            OptimizationBenchmarkEvidence {
                case_id,
                artifact,
                covered_pass_families: pass_families,
                required_custom_metrics,
                required_positive_metrics,
                exists: inspection.exists,
                read_error: inspection.read_error,
                case_count: inspection.case_count,
                min_wall_samples: inspection.min_wall_samples,
                min_wall_p50: inspection.min_wall_p50,
                min_wall_p95: inspection.min_wall_p95,
                min_wall_p99: inspection.min_wall_p99,
                min_baseline_wall_samples: inspection.min_baseline_wall_samples,
                min_baseline_wall_p50: inspection.min_baseline_wall_p50,
                min_baseline_wall_p95: inspection.min_baseline_wall_p95,
                min_baseline_wall_p99: inspection.min_baseline_wall_p99,
                min_wall_speedup_x1000: inspection.min_wall_speedup_x1000,
                missing_custom_metrics: inspection.missing_custom_metrics,
                non_positive_required_metrics: inspection.non_positive_required_metrics,
                non_winning_cases: inspection.non_winning_cases,
                blockers: inspection.blockers,
            }
        })
        .collect::<Vec<_>>();
    covered_pass_families.sort_unstable();
    covered_pass_families.dedup();
    let uncovered_pass_families = required_pass_families
        .iter()
        .copied()
        .filter(|family| !covered_pass_families.contains(family))
        .collect::<Vec<_>>();
    for family in &uncovered_pass_families {
        blockers.push(format!(
            "required optimization pass family `{family}` has no benchmark manifest coverage"
        ));
    }
    write_json(
        &workspace_root.join("release/evidence/optimization/pass-family-benchmark-manifest.json"),
        &OptimizationBenchmarkManifest {
            schema_version: 1,
            backend: backend.to_string(),
            required_case_count,
            required_pass_families,
            covered_pass_families,
            uncovered_pass_families,
            cases,
            blockers,
        },
    );
}

fn write_json(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    let json = match serde_json::to_string_pretty(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(path, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn release_axis_blockers(reports: &[Value]) -> Vec<String> {
    let mut blockers = Vec::new();
    if reports.is_empty() {
        blockers.push("no benchmark case reports available for release axes".to_string());
    }
    if reports.len() < 12 {
        blockers.push(format!(
            "only {} benchmark report(s) available for release axes; release needs at least 12 workload reports",
            reports.len()
        ));
    }
    if min_metric_p50(reports, "wall_ns").is_none() {
        blockers.push("missing wall_ns metric for warm_us_per_file".to_string());
    }
    if min_first_available_metric_p50(
        reports,
        &[
            "cold_compile_ns",
            "cold_wall_ns",
            "compile_ns",
            "lower_ns",
            "optimize_ns",
        ],
    )
    .is_none()
    {
        blockers.push("missing cold/compile metric for cold_pipeline_build_ms".to_string());
    }
    if max_metric_p50(reports, "wall_gb_s_x1000")
        .or_else(|| max_metric_p50(reports, "device_gb_s_x1000"))
        .is_none()
    {
        blockers.push("missing throughput metric for gbs_scan_throughput".to_string());
    }
    if max_vram_mib(reports).is_none() {
        blockers.push("missing GPU memory evidence for max_vram_mib".to_string());
    }
    blockers
}

fn min_first_available_metric_p50(reports: &[Value], keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| min_metric_p50(reports, key))
}

fn min_metric_p50(reports: &[Value], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().min()
}

fn max_metric_p50(reports: &[Value], key: &str) -> Option<u64> {
    metric_p50_values(reports, key).into_iter().max()
}

fn metric_p50_values(reports: &[Value], key: &str) -> Vec<u64> {
    let mut values = Vec::new();
    for report in reports {
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            let Some(metrics) = case.get("metrics").and_then(Value::as_object) else {
                continue;
            };
            let Some(value) = metrics
                .get(key)
                .and_then(|metric| metric.get("p50"))
                .and_then(Value::as_u64)
            else {
                continue;
            };
            values.push(value);
        }
    }
    values
}

fn max_observed_ulp(reports: &[Value]) -> Option<u32> {
    let mut max_ulp = None::<u32>;
    for report in reports {
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            if let Some(ulp) = case
                .get("correctness")
                .and_then(|correctness| correctness.get("Toleranced"))
                .and_then(|toleranced| toleranced.get("max_observed_ulp"))
                .and_then(Value::as_u64)
            {
                let ulp = ulp.min(u64::from(u32::MAX)) as u32;
                max_ulp = Some(max_ulp.map_or(ulp, |current| current.max(ulp)));
            }
        }
    }
    max_ulp
}

fn max_vram_mib(reports: &[Value]) -> Option<u64> {
    let mut max_mib = None::<u64>;
    for report in reports {
        if let Some(devices) = report
            .get("environment")
            .and_then(|environment| environment.get("gpu_devices"))
            .and_then(Value::as_array)
        {
            for device in devices {
                if let Some(mib) = device.get("memory_total_mib").and_then(Value::as_u64) {
                    max_mib = Some(max_mib.map_or(mib, |current| current.max(mib)));
                }
            }
        }
        let Some(cases) = report.get("cases").and_then(Value::as_array) else {
            continue;
        };
        for case in cases {
            if let Some(mib) = case
                .get("metrics")
                .and_then(|metrics| metrics.get("memory_total_mib"))
                .and_then(|metric| metric.get("p50"))
                .and_then(Value::as_u64)
            {
                max_mib = Some(max_mib.map_or(mib, |current| current.max(mib)));
            }
        }
    }
    max_mib
}

fn run_named_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) {
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
    run_command(workspace_root, &borrowed);
}

fn run_named_benchmark_if_needed(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
    reuse_existing: bool,
) {
    if reuse_existing
        && benchmark_artifact_is_reusable(workspace_root, backend, case_id, case_id, output, false)
    {
        return;
    }
    run_named_benchmark(
        workspace_root,
        case_id,
        backend,
        output,
        measured_samples,
        sample_timeout_secs,
    );
}

fn benchmark_artifact_is_reusable(
    workspace_root: &Path,
    backend: &str,
    family_id: &str,
    case_id: &str,
    output: &str,
    cpu_sota_100x_required: bool,
) -> bool {
    let path = workspace_root.join(output);
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(_) => return false,
    };
    let Ok(report) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    if report.get("selected_backend").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    if report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64)
        != Some(0)
    {
        return false;
    }
    let Some(case) = report
        .get("cases")
        .and_then(Value::as_array)
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case.get("id").and_then(Value::as_str) == Some(case_id))
        })
    else {
        return false;
    };
    if case.get("backend_id").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    if case.get("status").and_then(Value::as_str) != Some("pass") {
        return false;
    }
    if cpu_sota_100x_required && !suite_case_has_cpu_sota_contract(case, 100.0) {
        return false;
    }
    if cpu_sota_100x_required {
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
        if !contract_passed || !speedup_passed {
            return false;
        }
    }
    let _ = family_id;
    true
}

fn copy_artifact(workspace_root: &Path, source: &str, target: &str) {
    let source = workspace_root.join(source);
    let target = workspace_root.join(target);
    if let Some(parent) = target.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::copy(&source, &target) {
        eprintln!(
            "Fix: failed to copy `{}` to `{}`: {error}",
            source.display(),
            target.display()
        );
        std::process::exit(1);
    }
}

fn run_command(workspace_root: &Path, args: &[&str]) {
    let runner = cargo_runner(workspace_root);
    let status = Command::new(&runner)
        .args(args)
        .current_dir(workspace_root)
        .status();
    let display = format!("{} {}", runner.display(), args.join(" "));
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("Fix: `{display}` failed with {status}");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!(
                "Fix: failed to run `{display}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`."
            );
            std::process::exit(1);
        }
    }
}

fn cargo_runner(workspace_root: &Path) -> PathBuf {
    if let Some(runner) = std::env::var_os("VYRE_CARGO_RUNNER") {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
}

struct Config {
    backend: String,
    only: Option<String>,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
    include_wgpu_comparison: bool,
    reuse_existing: bool,
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut backend = "cuda".to_string();
    let mut only = None;
    let mut measured_samples = Some(30usize);
    let mut sample_timeout_secs = 120u64;
    let mut include_wgpu_comparison = false;
    let mut reuse_existing = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --backend requires a backend id.".to_string());
                };
                if value != "cuda" && value != "wgpu" {
                    return Err(
                        "Fix: release-benchmarks only accepts `cuda` or `wgpu` backends."
                            .to_string(),
                    );
                }
                backend = value.clone();
                index += 2;
            }
            "--only" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --only requires a release workload family id.".to_string());
                };
                only = Some(value.clone());
                index += 2;
            }
            "--measured-samples" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --measured-samples requires a positive integer.".to_string());
                };
                let parsed = value.parse::<usize>().map_err(|error| {
                    format!("Fix: --measured-samples must be a positive integer: {error}")
                })?;
                if parsed == 0 {
                    return Err("Fix: --measured-samples must be greater than zero.".to_string());
                }
                if parsed < 30 {
                    return Err(
                        "Fix: release-benchmarks requires --measured-samples >= 30 for release evidence."
                            .to_string(),
                    );
                }
                measured_samples = Some(parsed);
                index += 2;
            }
            "--sample-timeout-secs" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --sample-timeout-secs requires seconds.".to_string());
                };
                sample_timeout_secs = value.parse::<u64>().map_err(|error| {
                    format!("Fix: --sample-timeout-secs must be a positive integer: {error}")
                })?;
                if sample_timeout_secs == 0 {
                    return Err("Fix: --sample-timeout-secs must be greater than zero.".to_string());
                }
                index += 2;
            }
            "--include-wgpu-comparison" => {
                include_wgpu_comparison = true;
                index += 1;
            }
            "--reuse-existing" => {
                reuse_existing = true;
                index += 1;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- release-benchmarks [--backend cuda] [--only FAMILY] [--measured-samples N] [--sample-timeout-secs N] [--include-wgpu-comparison] [--reuse-existing]\n\n\
                     Generates CUDA-first release benchmark JSON artifacts from the release workload matrix. WGPU comparison evidence is opt-in so CUDA release validation time is not spent on non-release-path backends by default. --reuse-existing validates already-written artifacts and reruns only missing or invalid cases."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown release-benchmarks option `{other}`.")),
        }
    }
    Ok(Config {
        backend,
        only,
        measured_samples,
        sample_timeout_secs,
        include_wgpu_comparison,
        reuse_existing,
    })
}
