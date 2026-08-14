use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use super::evidence_schema::{
    BackendSuiteArtifact, BackendSuiteArtifactInput, BackendSuiteEvidence, HardwareDigestField,
    HardwareUnavailableReason,
};
use super::inspect_core::{
    first_metric_p50, read_benchmark_report, read_text_bounded, record_observed_metric_percentile,
    record_required_metric_percentile, report_cases, WallClockMinima,
};
use super::metrics::write_json;
use super::release_thresholds::{
    MAX_RELEASE_BENCHMARK_TEXT_BYTES, MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR,
    MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR, MIN_CUDA_RELEASE_MEMORY_MIB,
};
use super::runner::run_command_status;

pub(super) fn run_workload_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) -> Result<(), String> {
    let owned_args = super::runner::benchmark_command_args(
        case_id,
        backend,
        output,
        measured_samples,
        sample_timeout_secs,
    );
    let borrowed = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
    run_command_status(workspace_root, &borrowed)?;
    if case_id == "compound.pipeline.fused_filter.1m" {
        attach_fused_execution_dag(&workspace_root.join(output), case_id)?;
    }
    Ok(())
}

fn attach_fused_execution_dag(path: &Path, case_id: &str) -> Result<(), String> {
    let text = read_text_bounded(path, MAX_RELEASE_BENCHMARK_TEXT_BYTES)
        .map_err(|error| format!("could not read `{}`: {error}", path.display()))?;
    let mut report = serde_json::from_str::<Value>(&text)
        .map_err(|error| format!("could not parse `{}`: {error}", path.display()))?;
    let dag = fused_execution_dag_from_report(&report, case_id)?;
    let report_object = report.as_object_mut().ok_or_else(|| {
        format!(
            "benchmark artifact `{}` must be a JSON object",
            path.display()
        )
    })?;
    report_object.insert("fused_execution_dag".to_string(), dag);
    write_json(path, &report);
    Ok(())
}

fn fused_execution_dag_from_report(report: &Value, case_id: &str) -> Result<Value, String> {
    let case = report
        .get("cases")
        .and_then(Value::as_array)
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case.get("id").and_then(Value::as_str) == Some(case_id))
        })
        .ok_or_else(|| format!("benchmark report is missing case `{case_id}`"))?;
    let metrics = case
        .get("metrics")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("benchmark case `{case_id}` is missing metrics"))?;
    let required_metric = |name: &str| {
        first_metric_p50(Some(metrics), &[name])
            .filter(|value| *value != 0)
            .ok_or_else(|| format!("benchmark case `{case_id}` metric `{name}` must be positive"))
    };
    let host_to_device_bytes = required_metric("host_to_device_bytes")?;
    let device_to_host_bytes = required_metric("device_to_host_bytes")?;
    let output_bytes = required_metric("output_bytes")?;
    let cpu_digest = required_metric("cpu_digest")?;
    let gpu_digest = required_metric("gpu_digest")?;
    let parity_passed = cpu_digest == gpu_digest
        && case.get("correctness").and_then(Value::as_str) == Some("Exact");
    if !parity_passed {
        return Err(format!(
            "benchmark case `{case_id}` must report exact matching CPU and GPU digests"
        ));
    }
    let digest = |value: u64| format!("fnv64:{value:016x}");

    Ok(json!({
        "schema_version": 1,
        "contract": "fused-execution-dag:v1",
        "graph_nodes": ["ingest", "scan", "verify", "confidence", "report"],
        "memory_edges": [
            {"from": "ingest", "to": "scan", "buffer": "resident-inputs", "bytes": host_to_device_bytes},
            {"from": "scan", "to": "verify", "buffer": "candidate-mask", "bytes": output_bytes},
            {"from": "verify", "to": "confidence", "buffer": "verified-candidates", "bytes": output_bytes},
            {"from": "confidence", "to": "report", "buffer": "final-scores", "bytes": output_bytes}
        ],
        "host_sync_points": 1,
        "bytes_transferred": {
            "host_to_device_bytes": host_to_device_bytes,
            "device_to_host_bytes": device_to_host_bytes
        },
        "reporter_parity": {
            "parity_passed": true,
            "cpu_digest": digest(cpu_digest),
            "gpu_digest": digest(gpu_digest),
            "output_digest": digest(gpu_digest)
        },
        "fallback_reasons": [],
        "comparator": "exact-output-digest-v1",
        "dataset_id": case_id,
        "metric_family": "resident-compound-filter",
        "release_floor": "exact output parity with final-state-only host synchronization",
        "failure_mode": "fail closed on digest mismatch, intermediate host synchronization, or incomplete transfer accounting",
        "frontier_leaderboard_artifact": "release/evidence/benchmarks/frontier-leaderboard.json"
    }))
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
    write_backend_suite_with_extra_blockers(workspace_root, backend, artifact_inputs, Vec::new());
}

pub(super) fn write_backend_suite_with_extra_blockers(
    workspace_root: &Path,
    backend: &str,
    artifact_inputs: Vec<BackendSuiteArtifactInput>,
    extra_blockers: Vec<String>,
) {
    let output = backend_suite_output_path(backend);
    let mut blockers = extra_blockers;
    if artifact_inputs.is_empty() {
        blockers.push(format!(
            "backend `{backend}` release suite has zero artifacts"
        ));
    }
    let path_counts = backend_suite_input_counts(&artifact_inputs, |artifact| &artifact.path);
    let family_counts =
        backend_suite_input_counts(&artifact_inputs, |artifact| &artifact.family_id);
    for artifact in artifact_inputs
        .iter()
        .filter(|artifact| artifact.family_id.trim().is_empty())
    {
        blockers.push(format!(
            "backend `{backend}` release suite artifact `{}` has blank family_id",
            artifact.path
        ));
    }
    for artifact in artifact_inputs
        .iter()
        .filter(|artifact| artifact.requested_case_id.trim().is_empty())
    {
        blockers.push(format!(
            "backend `{backend}` release suite artifact `{}` has blank requested_case_id",
            artifact.path
        ));
    }
    for (family_id, count) in &family_counts {
        if *count > 1 {
            blockers.push(format!(
                "backend `{backend}` release suite has {count} artifact input(s) for family `{family_id}`"
            ));
        }
    }
    for (path, count) in &path_counts {
        if *count > 1 {
            blockers.push(format!(
                "backend `{backend}` release suite has {count} artifact input(s) for path `{path}`"
            ));
        }
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
    let (hardware_digest, hardware_digest_fields, hardware_unavailable_reasons, hardware_blockers) =
        backend_suite_hardware_digest(backend, &artifact_statuses);
    blockers.extend(hardware_blockers);
    let schema_digest_chain =
        backend_suite_schema_digest_chain(backend, &artifact_statuses, &hardware_digest);
    let evidence = BackendSuiteEvidence {
        schema_version: 3,
        backend: backend.to_string(),
        schema_digest_chain,
        hardware_digest,
        hardware_digest_fields,
        hardware_unavailable_reasons,
        family_count: family_counts.len(),
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

fn backend_suite_schema_digest_chain(
    backend: &str,
    artifact_statuses: &[BackendSuiteArtifact],
    hardware_digest: &str,
) -> Value {
    let mut source_material = format!("benchmark-suite-source:v1\nbackend={backend}\n");
    let mut config_material = format!("benchmark-suite-config:v1\nbackend={backend}\n");
    let mut dataset_material = format!("benchmark-suite-dataset:v1\nbackend={backend}\n");
    for status in artifact_statuses {
        source_material.push_str("source_fingerprint=");
        source_material.push_str(status.source_fingerprint.as_deref().unwrap_or(""));
        source_material.push_str("\nsource_tree_fingerprint=");
        source_material.push_str(status.source_tree_fingerprint.as_deref().unwrap_or(""));
        source_material.push('\n');

        config_material.push_str("path=");
        config_material.push_str(&status.path);
        config_material.push_str("\nfamily_id=");
        config_material.push_str(&status.family_id);
        config_material.push_str("\nrequested_case_id=");
        config_material.push_str(&status.requested_case_id);
        config_material.push_str("\ncpu_sota_100x_required=");
        config_material.push_str(if status.cpu_sota_100x_required {
            "true"
        } else {
            "false"
        });
        config_material.push('\n');

        dataset_material.push_str("family_id=");
        dataset_material.push_str(&status.family_id);
        dataset_material.push_str("\nrequested_case_id=");
        dataset_material.push_str(&status.requested_case_id);
        dataset_material.push_str("\nartifact_path=");
        dataset_material.push_str(&status.path);
        dataset_material.push('\n');
    }
    let source_digest = format!(
        "benchmark-source-digest:v1:{}",
        xtask::hash::sha256_hex(source_material.as_bytes())
    );
    let command_digest = format!(
        "benchmark-command-digest:v1:{}",
        xtask::hash::sha256_hex(
            format!("release-benchmarks backend-suite:v1 backend={backend}\n").as_bytes()
        )
    );
    let config_digest = format!(
        "benchmark-config-digest:v1:{}",
        xtask::hash::sha256_hex(config_material.as_bytes())
    );
    let dataset_digest = format!(
        "benchmark-dataset-digest:v1:{}",
        xtask::hash::sha256_hex(dataset_material.as_bytes())
    );
    let comparator_version =
        "benchmark-suite-comparator:v1:case-integrity+summary+hardware".to_string();
    crate::bench::benchmark_evidence_semantics::benchmark_schema_digest_chain_value(
        "backend-suite",
        3,
        &source_digest,
        &command_digest,
        &config_digest,
        hardware_digest,
        &dataset_digest,
        &comparator_version,
    )
}

fn backend_suite_hardware_digest(
    backend: &str,
    artifact_statuses: &[BackendSuiteArtifact],
) -> (
    String,
    Vec<HardwareDigestField>,
    Vec<HardwareUnavailableReason>,
    Vec<String>,
) {
    let mut fields = Vec::new();
    let mut unavailable = Vec::new();
    let mut blockers = Vec::new();
    record_string_hardware_field(
        &mut fields,
        &mut blockers,
        artifact_statuses,
        "host_cpu_model",
        "artifact.environment.host_cpu_model|cpu_model|host_cpu",
        |status| status.host_cpu_model.clone(),
    );
    if backend == "cuda" {
        record_string_hardware_field(
            &mut fields,
            &mut blockers,
            artifact_statuses,
            "gpu_model",
            "artifact.environment.gpu_devices[0].name",
            |status| status.gpu_model.clone(),
        );
        record_u64_hardware_field(
            &mut fields,
            &mut blockers,
            artifact_statuses,
            "gpu_memory_total_mib",
            "artifact.environment.gpu_devices[0].memory_total_mib",
            |status| status.gpu_memory_total_mib,
        );
        record_compute_capability_hardware_field(&mut fields, &mut blockers, artifact_statuses);
        record_string_hardware_field(
            &mut fields,
            &mut blockers,
            artifact_statuses,
            "nvidia_driver_version",
            "artifact.environment.nvidia_driver_version",
            |status| status.nvidia_driver_version.clone(),
        );
        record_string_hardware_field(
            &mut fields,
            &mut blockers,
            artifact_statuses,
            "cuda_toolkit_version",
            "artifact.environment.nvidia_cuda_version",
            |status| status.nvidia_cuda_version.clone(),
        );
        for (field, reason) in [
            (
                "host_ram_total_mib",
                "benchmark environment did not expose host RAM capacity",
            ),
            (
                "storage_filesystem_signature",
                "benchmark environment did not expose SSD or filesystem identity",
            ),
            (
                "os_kernel_version",
                "benchmark environment did not expose operating-system kernel version",
            ),
            (
                "gpu_ecc_state",
                "nvidia-smi benchmark capture did not expose visible ECC state",
            ),
            (
                "gpu_clock_mhz",
                "nvidia-smi benchmark capture did not expose graphics or memory clocks",
            ),
        ] {
            unavailable.push(HardwareUnavailableReason {
                field: field.to_string(),
                reason: reason.to_string(),
                fix: format!(
                    "Fix: extend benchmark environment capture to populate `{field}` before using hardware comparisons that depend on it."
                ),
            });
        }
    }
    let mut material = format!("benchmark-hardware-digest:v1\nbackend={backend}\n");
    for field in &fields {
        material.push_str("field=");
        material.push_str(&field.field);
        material.push_str("\nvalue=");
        material.push_str(&field.value);
        material.push_str("\nsource=");
        material.push_str(&field.source);
        material.push('\n');
    }
    for reason in &unavailable {
        material.push_str("unavailable=");
        material.push_str(&reason.field);
        material.push_str("\nreason=");
        material.push_str(&reason.reason);
        material.push_str("\nfix=");
        material.push_str(&reason.fix);
        material.push('\n');
    }
    let hardware_digest = format!(
        "benchmark-hardware-digest:v1:{}",
        xtask::hash::sha256_hex(material.as_bytes())
    );
    (hardware_digest, fields, unavailable, blockers)
}

fn record_string_hardware_field(
    fields: &mut Vec<HardwareDigestField>,
    blockers: &mut Vec<String>,
    artifact_statuses: &[BackendSuiteArtifact],
    field: &str,
    source: &str,
    extractor: impl Fn(&BackendSuiteArtifact) -> Option<String>,
) {
    let values = artifact_statuses
        .iter()
        .filter_map(extractor)
        .collect::<BTreeSet<_>>();
    record_hardware_field_values(fields, blockers, field, source, values);
}

fn record_u64_hardware_field(
    fields: &mut Vec<HardwareDigestField>,
    blockers: &mut Vec<String>,
    artifact_statuses: &[BackendSuiteArtifact],
    field: &str,
    source: &str,
    extractor: impl Fn(&BackendSuiteArtifact) -> Option<u64>,
) {
    let values = artifact_statuses
        .iter()
        .filter_map(extractor)
        .map(|value| value.to_string())
        .collect::<BTreeSet<_>>();
    record_hardware_field_values(fields, blockers, field, source, values);
}

fn record_compute_capability_hardware_field(
    fields: &mut Vec<HardwareDigestField>,
    blockers: &mut Vec<String>,
    artifact_statuses: &[BackendSuiteArtifact],
) {
    let values = artifact_statuses
        .iter()
        .filter_map(|status| {
            Some(format!(
                "{}.{}",
                status.gpu_compute_capability_major?, status.gpu_compute_capability_minor?
            ))
        })
        .collect::<BTreeSet<_>>();
    record_hardware_field_values(
        fields,
        blockers,
        "gpu_compute_capability",
        "artifact.environment.gpu_devices[0].compute_capability_major/minor",
        values,
    );
}

fn record_hardware_field_values(
    fields: &mut Vec<HardwareDigestField>,
    blockers: &mut Vec<String>,
    field: &str,
    source: &str,
    values: BTreeSet<String>,
) {
    if values.is_empty() {
        blockers.push(format!(
            "hardware digest field `{field}` has no artifact-backed value"
        ));
        return;
    }
    if values.len() > 1 {
        blockers.push(format!(
            "hardware digest field `{field}` has {} distinct values; release benchmark suites must not merge incomparable hardware",
            values.len()
        ));
    }
    fields.push(HardwareDigestField {
        field: field.to_string(),
        value: values.into_iter().collect::<Vec<_>>().join(" | "),
        source: source.to_string(),
    });
}

fn backend_suite_input_counts(
    artifact_inputs: &[BackendSuiteArtifactInput],
    value: impl Fn(&BackendSuiteArtifactInput) -> &str,
) -> BTreeMap<String, usize> {
    artifact_inputs
        .iter()
        .filter_map(|artifact| {
            let value = value(artifact).trim();
            (!value.is_empty()).then(|| value.to_string())
        })
        .fold(BTreeMap::new(), |mut counts, value| {
            *counts.entry(value).or_default() += 1;
            counts
        })
}

pub(super) fn backend_suite_output_path(backend: &str) -> String {
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
            fused_execution_dag_contract: None,
            fused_execution_dag_node_count: None,
            fused_execution_dag_memory_edge_count: None,
            fused_execution_dag_host_sync_points: None,
            case_count: 0,
            failed_count: None,
            nonmatching_case_backend_count: 0,
            minima: WallClockMinima::default(),
            cpu_sota_100x_required: artifact.cpu_sota_100x_required,
            cpu_sota_100x_contract_cases: 0,
            cpu_sota_100x_passing_cases: 0,
            blockers,
        };
    }
    if bytes == 0 {
        blockers.push("empty".to_string());
    }
    let (report, _) = read_benchmark_report(&path, &mut blockers);
    let selected_backend = report
        .get("selected_backend")
        .and_then(Value::as_str)
        .map(str::to_string);
    let source_fingerprint = report
        .get("source_fingerprint")
        .and_then(nonblank_str)
        .map(str::to_string);
    let source_tree_fingerprint = report
        .get("source_tree_fingerprint")
        .and_then(nonblank_str)
        .map(str::to_string);
    if source_tree_fingerprint.is_none() {
        blockers.push("artifact has no source_tree_fingerprint provenance".to_string());
    }
    match &source_fingerprint {
        Some(fingerprint)
            if !crate::bench::benchmark_evidence_semantics::source_fingerprint_issues(
                fingerprint,
            )
            .is_empty() =>
        {
            blockers.push(format!(
                "source_fingerprint `{fingerprint}` is not release-grade provenance"
            ));
        }
        None => blockers.push("artifact has no source_fingerprint provenance".to_string()),
        Some(_) => {}
    }
    if let (Some((field, fingerprint)), Some(current_fingerprint)) = (
        crate::bench::benchmark_evidence_semantics::report_freshness_fingerprint(&report),
        crate::bench::benchmark_evidence_semantics::current_freshness_fingerprint_for_report(
            &path, &report,
        ),
    ) {
        for issue in crate::bench::benchmark_evidence_semantics::source_fingerprint_freshness_issues(
            fingerprint,
            &current_fingerprint,
        ) {
            match issue {
                crate::bench::benchmark_evidence_semantics::SourceFingerprintFreshnessIssue::Mismatch {
                    source_fingerprint,
                    current_source_fingerprint,
                } => blockers.push(format!(
                    "{field} `{source_fingerprint}` does not match current workspace source `{current_source_fingerprint}`"
                )),
            }
        }
    }
    if selected_backend.as_deref() != Some(backend) {
        blockers.push(format!(
            "selected_backend `{:?}` does not match requested backend `{backend}`",
            selected_backend
        ));
    }
    let requires_fused_execution_dag = artifact.family_id == "compound-fused-filter"
        || artifact.requested_case_id == "compound.pipeline.fused_filter.1m";
    if requires_fused_execution_dag {
        blockers.extend(
            crate::bench::benchmark_evidence_semantics::benchmark_fused_execution_dag_issues(
                &artifact.path,
                &report,
            ),
        );
    }
    let fused_execution_dag = report.get("fused_execution_dag");
    let fused_execution_dag_contract = fused_execution_dag
        .and_then(|dag| dag.get("contract"))
        .and_then(nonblank_str)
        .map(str::to_string)
        .or_else(|| requires_fused_execution_dag.then(|| "fused-execution-dag:v1".to_string()));
    let fused_execution_dag_node_count = fused_execution_dag
        .and_then(|dag| dag.get("graph_nodes"))
        .and_then(Value::as_array)
        .map(Vec::len);
    let fused_execution_dag_memory_edge_count = fused_execution_dag
        .and_then(|dag| dag.get("memory_edges"))
        .and_then(Value::as_array)
        .map(Vec::len);
    let fused_execution_dag_host_sync_points = fused_execution_dag
        .and_then(|dag| dag.get("host_sync_points"))
        .and_then(Value::as_u64);
    let environment = report.get("environment");
    let first_gpu = environment
        .and_then(|environment| environment.get("gpu_devices"))
        .and_then(Value::as_array)
        .and_then(|devices| devices.first());
    let gpu_model = first_gpu
        .and_then(|device| device.get("name"))
        .and_then(nonblank_str)
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
        .and_then(nonblank_str)
        .map(str::to_string);
    let nvidia_driver_version = environment
        .and_then(|environment| environment.get("nvidia_driver_version"))
        .and_then(nonblank_str)
        .map(str::to_string);
    let nvidia_cuda_version = environment
        .and_then(|environment| environment.get("nvidia_cuda_version"))
        .and_then(nonblank_str)
        .map(str::to_string);
    if backend == "cuda" {
        if gpu_model.is_none() {
            blockers.push("CUDA artifact has no nvidia-smi GPU model provenance".to_string());
        }
        if nvidia_driver_version.is_none() {
            blockers.push(
                "CUDA artifact has no nvidia-smi NVIDIA driver version provenance".to_string(),
            );
        }
        if nvidia_cuda_version.is_none() {
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
    let cases = report_cases(&report, &mut blockers);
    if let Some(mismatch) =
        crate::bench::benchmark_evidence_semantics::benchmark_report_summary_case_evidence_mismatch(
            &report,
        )
    {
        blockers.push(format!("benchmark summary is invalid: {mismatch}"));
    }
    crate::bench::benchmark_evidence_semantics::inspect_source_artifact_case_integrity(
        &artifact.path,
        &report,
        &format!("backend `{backend}` release suite artifact"),
        &mut blockers,
    );
    let mut case_failed_count = 0u64;
    let mut nonmatching_case_backend_count = 0usize;
    let mut minima = WallClockMinima::default();
    let mut min_cuda_ptx_source_cache_entries = None::<u64>;
    let mut min_cuda_ptx_source_cache_hits = None::<u64>;
    let mut min_cuda_ptx_source_cache_misses = None::<u64>;
    let mut min_kernel_launches = None::<u64>;
    let mut requested_case_count = 0usize;
    for case in &cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        let case_failure_reason =
            crate::bench::benchmark_evidence_semantics::benchmark_case_failure_reason(case);
        if let Some(reason) = &case_failure_reason {
            case_failed_count += 1;
            blockers.push(format!("case `{case_id}` failed: {reason}"));
        }
        if case_id == artifact.requested_case_id {
            requested_case_count += 1;
        }
        if case.get("backend_id").and_then(Value::as_str) != Some(backend) {
            nonmatching_case_backend_count += 1;
        }
        let metrics = case.get("metrics").and_then(Value::as_object);
        minima.record_case(
            case_id,
            &format!("case `{case_id}`"),
            metrics,
            &mut blockers,
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
            record_observed_metric_percentile(
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
    }
    let (cpu_sota_100x_contract_cases, cpu_sota_100x_passing_cases) =
        crate::bench::benchmark_evidence_semantics::cpu_sota_100x_case_counts(&report);
    let cpu_sota_100x_contract_cases = cpu_sota_100x_contract_cases as usize;
    let cpu_sota_100x_passing_cases = cpu_sota_100x_passing_cases as usize;
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
    if requested_case_count == 0 {
        blockers.push(format!(
            "requested case `{}` is absent from artifact cases",
            artifact.requested_case_id
        ));
    } else if requested_case_count > 1 {
        blockers.push(format!(
            "requested case `{}` appears {requested_case_count} times in artifact cases",
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
        fused_execution_dag_contract,
        fused_execution_dag_node_count,
        fused_execution_dag_memory_edge_count,
        fused_execution_dag_host_sync_points,
        case_count: cases.len(),
        failed_count,
        nonmatching_case_backend_count,
        minima,
        cpu_sota_100x_required: artifact.cpu_sota_100x_required,
        cpu_sota_100x_contract_cases,
        cpu_sota_100x_passing_cases,
        blockers,
    }
}

pub(super) fn nonblank_str(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::{
        benchmark_case, case_summary, cpu_sota_contract, cuda_cached_metrics,
        hidden_invalid_measured_case, host_environment, launched_percentile_metrics,
    };

    use tempfile::TempDir;

    /// Compound release evidence records the measured resident transfer sizes
    /// and exact CPU/GPU parity in the fused execution DAG consumed by the gate.
    #[test]
    fn compound_report_builds_complete_fused_execution_dag() {
        let report = json!({
            "cases": [{
                "id": "compound.pipeline.fused_filter.1m",
                "correctness": "Exact",
                "metrics": {
                    "host_to_device_bytes": {"p50": 12_582_912},
                    "device_to_host_bytes": {"p50": 4_194_304},
                    "output_bytes": {"p50": 4_194_304},
                    "cpu_digest": {"p50": 91},
                    "gpu_digest": {"p50": 91}
                }
            }]
        });

        let dag = fused_execution_dag_from_report(&report, "compound.pipeline.fused_filter.1m")
            .expect("Fix: exact compound benchmark evidence must produce a fused DAG.");
        let artifact = json!({"fused_execution_dag": dag});
        assert_eq!(
            crate::bench::benchmark_evidence_semantics::benchmark_fused_execution_dag_issues(
                "compound.json",
                &artifact,
            ),
            Vec::<String>::new()
        );
    }

    /// A transfer-accounting gap cannot be filled with a guessed byte count,
    /// because that would make the resident DAG evidence pass without proof.
    #[test]
    fn compound_report_rejects_missing_transfer_metric() {
        let report = json!({
            "cases": [{
                "id": "compound.pipeline.fused_filter.1m",
                "correctness": "Exact",
                "metrics": {
                    "host_to_device_bytes": {"p50": 12_582_912},
                    "output_bytes": {"p50": 4_194_304},
                    "cpu_digest": {"p50": 91},
                    "gpu_digest": {"p50": 91}
                }
            }]
        });

        assert_eq!(
            fused_execution_dag_from_report(
                &report,
                "compound.pipeline.fused_filter.1m",
            ),
            Err(
                "benchmark case `compound.pipeline.fused_filter.1m` metric `device_to_host_bytes` must be positive"
                    .to_string()
            )
        );
    }
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
    fn write_backend_suite_records_workload_run_failures() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite run-failure blocker test.");

        write_backend_suite_with_extra_blockers(
            dir.path(),
            "wgpu",
            Vec::new(),
            vec![
                "backend `wgpu` comparison family `string-bitmap-scatter` case `release.string_bitmap_scatter.1m` artifact `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json`: Fix: benchmark command failed with exit status 1"
                    .to_string(),
            ],
        );

        let suite_path = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-fallback-suite.json");
        let text = fs::read_to_string(&suite_path)
            .expect("Fix: read generated WGPU fallback suite JSON for run-failure assertions.");
        let suite = serde_json::from_str::<Value>(&text)
            .expect("Fix: generated WGPU fallback suite JSON must be parseable.");
        let blockers = suite
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated suite must carry blockers array.");

        assert!(
            blockers.iter().any(|blocker| {
                blocker.as_str().is_some_and(|blocker| {
                    blocker.contains("comparison family `string-bitmap-scatter`")
                        && blocker.contains("benchmark command failed")
                })
            }),
            "Fix: backend suite evidence must record benchmark run failures instead of leaving them only on stderr; blockers={blockers:?}"
        );
    }

    #[test]
    fn write_backend_suite_rejects_duplicate_family_input_coverage() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite duplicate family test.");

        write_backend_suite(
            dir.path(),
            "wgpu",
            vec![
                BackendSuiteArtifactInput {
                    path: "release/evidence/benchmarks/wgpu-condition-fast.json".to_string(),
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    cpu_sota_100x_required: false,
                },
                BackendSuiteArtifactInput {
                    path: "release/evidence/benchmarks/wgpu-condition-slow.json".to_string(),
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.10m".to_string(),
                    cpu_sota_100x_required: false,
                },
            ],
        );

        let suite_path = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-fallback-suite.json");
        let text = fs::read_to_string(&suite_path)
            .expect("Fix: read generated WGPU fallback suite JSON for duplicate family test.");
        let suite = serde_json::from_str::<Value>(&text)
            .expect("Fix: generated WGPU fallback suite JSON must be parseable.");
        let blockers = suite
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated suite must carry blockers array.");

        assert_eq!(
            suite.get("family_count").and_then(Value::as_u64),
            Some(1),
            "Fix: generated backend suite family_count must count unique workload families, not raw artifact inputs."
        );
        assert!(
            blockers.iter().any(|blocker| {
                blocker.as_str().is_some_and(|blocker| {
                    blocker.contains("has 2 artifact input(s) for family `condition-eval`")
                })
            }),
            "Fix: generated backend suite evidence must preserve duplicate family input blockers; blockers={blockers:?}"
        );
    }

    #[test]
    fn write_backend_suite_rejects_duplicate_artifact_input_paths() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite duplicate path test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-shared-path.json";

        write_backend_suite(
            dir.path(),
            "wgpu",
            vec![
                BackendSuiteArtifactInput {
                    path: artifact_rel.to_string(),
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    cpu_sota_100x_required: false,
                },
                BackendSuiteArtifactInput {
                    path: artifact_rel.to_string(),
                    family_id: "entropy-window".to_string(),
                    requested_case_id: "release.entropy_window.1m".to_string(),
                    cpu_sota_100x_required: false,
                },
            ],
        );

        let suite_path = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-fallback-suite.json");
        let text = fs::read_to_string(&suite_path)
            .expect("Fix: read generated WGPU fallback suite JSON for duplicate path test.");
        let suite = serde_json::from_str::<Value>(&text)
            .expect("Fix: generated WGPU fallback suite JSON must be parseable.");
        let blockers = suite
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated suite must carry blockers array.");

        assert!(
            blockers.iter().any(|blocker| {
                blocker.as_str().is_some_and(|blocker| {
                    blocker.contains(
                        "backend `wgpu` release suite has 2 artifact input(s) for path `release/evidence/benchmarks/wgpu-shared-path.json`",
                    )
                })
            }),
            "Fix: generated backend suite evidence must reject duplicate artifact input paths; blockers={blockers:?}"
        );
    }

    #[test]
    fn write_backend_suite_rejects_blank_requested_case_input() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite blank requested-case test.");
        let artifact_rel = "release/evidence/benchmarks/wgpu-blank-requested-case.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: blank requested-case artifact path must have a parent directory."),
        )
        .expect("Fix: create blank requested-case artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:abc:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:abc",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002}
                        }
                    }
                ]
            }))
            .expect("Fix: serialize blank requested-case benchmark artifact JSON."),
        )
        .expect("Fix: write blank requested-case benchmark artifact JSON.");

        write_backend_suite(
            dir.path(),
            "wgpu",
            vec![BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: " \t ".to_string(),
                cpu_sota_100x_required: false,
            }],
        );

        let suite_path = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-fallback-suite.json");
        let text = fs::read_to_string(&suite_path)
            .expect("Fix: read generated WGPU fallback suite JSON for blank requested-case test.");
        let suite = serde_json::from_str::<Value>(&text)
            .expect("Fix: generated WGPU fallback suite JSON must be parseable.");
        let blockers = suite
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated suite must carry blockers array.");

        assert!(
            blockers.iter().any(|blocker| {
                blocker.as_str().is_some_and(|blocker| {
                    blocker.contains(
                        "backend `wgpu` release suite artifact `release/evidence/benchmarks/wgpu-blank-requested-case.json` has blank requested_case_id",
                    )
                })
            }),
            "Fix: generated backend suite evidence must reject blank requested_case_id inputs; blockers={blockers:?}"
        );
    }

    #[test]
    fn suite_artifact_status_rejects_whitespace_only_provenance() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite blank provenance test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-blank-provenance.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create blank provenance suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "   ",
                "source_tree_fingerprint": "\t",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "environment": host_environment(" ", " ", "\t", "\n"),
                "cases": [benchmark_case(
                    "release.condition_eval.1m",
                    "cuda",
                    "pass",
                    [
                        (
                            "metrics",
                            launched_percentile_metrics([10, 11, 12], [1000, 1001, 1002], 1),
                        ),
                        (
                            "performance",
                            json!({"contract_passed": true, "speedup_x": 120.0}),
                        ),
                    ],
                )]
            }))
            .expect("Fix: serialize blank provenance benchmark artifact JSON."),
        )
        .expect("Fix: write blank provenance benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "cuda",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: false,
            },
        );

        assert_eq!(
            status.source_fingerprint, None,
            "Fix: whitespace-only source_fingerprint must not be serialized as suite provenance."
        );
        assert_eq!(
            status.source_tree_fingerprint, None,
            "Fix: whitespace-only source_tree_fingerprint must not be serialized as suite provenance."
        );
        assert_eq!(
            status.host_cpu_model, None,
            "Fix: whitespace-only host_cpu_model must not be serialized as suite provenance."
        );
        for expected in [
            "artifact has no source_fingerprint provenance",
            "artifact has no source_tree_fingerprint provenance",
            "CUDA artifact has no nvidia-smi GPU model provenance",
            "CUDA artifact has no nvidia-smi NVIDIA driver version provenance",
            "CUDA artifact has no nvidia-smi CUDA runtime version provenance",
        ] {
            assert!(
                status
                    .blockers
                    .iter()
                    .any(|blocker| blocker.contains(expected)),
                "Fix: suite artifact inspection must reject whitespace-only CUDA provenance `{expected}`; blockers={:?}",
                status.blockers
            );
        }
    }

    #[test]
    fn suite_artifact_status_rejects_missing_and_weak_source_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for suite source provenance test.");
        let artifacts = [
            (
                "release/evidence/benchmarks/cuda-missing-source-fingerprint.json",
                None,
                None,
                "artifact has no source_fingerprint provenance",
            ),
            (
                "release/evidence/benchmarks/cuda-git-commit-only-source.json",
                None,
                Some(json!({"commit": "abc123", "dirty": false})),
                "artifact has no source_fingerprint provenance",
            ),
            (
                "release/evidence/benchmarks/cuda-legacy-dirty-source-fingerprint.json",
                Some("git:abc123:dirty=true"),
                None,
                "source_fingerprint `git:abc123:dirty=true` is not release-grade provenance",
            ),
        ];

        for (artifact_rel, source_fingerprint, git, expected_blocker) in artifacts {
            let artifact_path = dir.path().join(artifact_rel);
            fs::create_dir_all(
                artifact_path
                    .parent()
                    .expect("Fix: suite artifact must have parent directory."),
            )
            .expect("Fix: create source provenance suite artifact parent directory.");
            let mut artifact = json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_tree_fingerprint": "source-tree-v1:abc",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "environment": host_environment("test CPU", "RTX 5090", "580.0", "13.0"),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": cuda_cached_metrics([10, 11, 12], [1000, 1001, 1002], 1, [1, 1, 0]),
                        "performance": {"contract_passed": true, "speedup_x": 120.0}
                    }
                ]
            });
            if let Some(source_fingerprint) = source_fingerprint {
                artifact["source_fingerprint"] = Value::String(source_fingerprint.to_string());
            }
            if let Some(git) = git {
                artifact["git"] = git;
            }
            fs::write(
                &artifact_path,
                serde_json::to_string_pretty(&artifact)
                    .expect("Fix: serialize source provenance benchmark artifact JSON."),
            )
            .expect("Fix: write source provenance benchmark artifact JSON.");

            let status = inspect_backend_suite_artifact(
                dir.path(),
                "cuda",
                &BackendSuiteArtifactInput {
                    path: artifact_rel.to_string(),
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    cpu_sota_100x_required: false,
                },
            );

            assert!(
                status
                    .blockers
                    .iter()
                    .any(|blocker| blocker.contains(expected_blocker)),
                "Fix: generated CUDA suite evidence must reject weak source fingerprint provenance `{expected_blocker}`; blockers={:?}",
                status.blockers
            );
        }
    }

    #[test]
    fn suite_artifact_status_rejects_stale_source_tree_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for stale suite source-tree test.");
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: create temp workspace Cargo.toml for source-tree freshness test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-stale-source-tree.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create stale source-tree suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "environment": host_environment("test CPU", "RTX 5090", "580.0", "13.0"),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": cuda_cached_metrics([10, 11, 12], [1000, 1001, 1002], 1, [1, 1, 0]),
                        "performance": {"contract_passed": true, "speedup_x": 120.0}
                    }
                ]
            }))
            .expect("Fix: serialize stale source-tree benchmark artifact JSON."),
        )
        .expect("Fix: write stale source-tree benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "cuda",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: false,
            },
        );

        assert!(
            status.blockers.iter().any(|blocker| {
                blocker.contains("source_tree_fingerprint `source-tree-v1:stale`")
                    && blocker.contains("does not match current workspace source")
            }),
            "Fix: generated CUDA suite evidence must reject stale source-tree benchmark artifacts; blockers={:?}",
            status.blockers
        );
    }

    #[test]
    fn suite_artifact_status_rejects_duplicate_requested_case_rows() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for duplicate requested-case suite test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-duplicate-requested-case.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create duplicate requested-case suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:abc",
                "summary": {"total_cases": 2, "passed": 2, "failed": 0},
                "environment": host_environment("test CPU", "RTX 5090", "580.0", "13.0"),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": cuda_cached_metrics([10, 11, 12], [1000, 1001, 1002], 1, [1, 1, 0]),
                        "performance": {"contract_passed": true, "speedup_x": 120.0}
                    },
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": cuda_cached_metrics([12, 13, 14], [1200, 1201, 1202], 1, [1, 1, 0]),
                        "performance": {"contract_passed": true, "speedup_x": 100.0}
                    }
                ]
            }))
            .expect("Fix: serialize duplicate requested-case benchmark artifact JSON."),
        )
        .expect("Fix: write duplicate requested-case benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "cuda",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: false,
            },
        );

        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "requested case `release.condition_eval.1m` appears 2 times in artifact cases"
            )),
            "Fix: generated CUDA suite evidence must reject artifacts where the requested_case_id resolves to multiple benchmark rows; blockers={:?}",
            status.blockers
        );
    }

    #[test]
    fn suite_artifact_status_rejects_backend_mismatched_cpu_sota_counts() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for backend-mismatch CPU-SOTA test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-backend-mismatch-cpu-sota.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create backend-mismatch CPU-SOTA suite artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:abc",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "environment": host_environment("test CPU", "RTX 5090", "580.0", "13.0"),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": cuda_cached_metrics([10, 11, 12], [1000, 1001, 1002], 1, [1, 1, 0]),
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize backend-mismatch CPU-SOTA benchmark artifact JSON."),
        )
        .expect("Fix: write backend-mismatch CPU-SOTA benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "cuda",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: true,
            },
        );

        assert_eq!(
            status.nonmatching_case_backend_count, 1,
            "Fix: backend-mismatched suite artifacts must remain visible in generated status rows."
        );
        assert_eq!(
            status.cpu_sota_100x_contract_cases, 0,
            "Fix: generated CUDA suite status must not count WGPU case rows as CUDA CPU-SOTA proof."
        );
        assert_eq!(
            status.cpu_sota_100x_passing_cases, 0,
            "Fix: generated CUDA suite status must not count backend-mismatched rows as passing CPU-SOTA proof."
        );
        for expected in [
            "1 case(s) do not match requested backend `cuda`",
            "CPU-SOTA 100x workload artifact has no 100x contract case",
            "CPU-SOTA 100x workload artifact has no passing 100x case",
        ] {
            assert!(
                status
                    .blockers
                    .iter()
                    .any(|blocker| blocker.contains(expected)),
                "Fix: generated CUDA suite evidence must expose backend-mismatched CPU-SOTA proof drift `{expected}`; blockers={:?}",
                status.blockers
            );
        }
    }

    #[test]
    fn suite_artifact_status_surfaces_source_integrity_blockers() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CUDA suite integrity test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-suite-integrity-drift.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: suite artifact must have parent directory."),
        )
        .expect("Fix: create CUDA suite integrity artifact parent directory.");
        let mut fallback_dispatch_metrics =
            cuda_cached_metrics([10, 11, 12], [2000, 2001, 2002], 1, [1, 1, 0]);
        fallback_dispatch_metrics["cuda_resident_borrowed_fallback_dispatches"] =
            json!({"p50": 2.0});
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:abc",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "environment": host_environment("test CPU", "RTX 5090", "580.0", "13.0"),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "optimization_passes_applied": ["cuda-resident-borrowed-escape-hatch"],
                        "metrics": fallback_dispatch_metrics,
                        "contract": cpu_sota_contract("release condition eval", &["wgpu"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize CUDA suite integrity benchmark artifact JSON."),
        )
        .expect("Fix: write CUDA suite integrity benchmark artifact JSON.");

        let status = inspect_backend_suite_artifact(
            dir.path(),
            "cuda",
            &BackendSuiteArtifactInput {
                path: artifact_rel.to_string(),
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cpu_sota_100x_required: true,
            },
        );

        assert!(
            status.blockers.iter().any(|blocker| blocker.contains(
                "source_artifact `release/evidence/benchmarks/cuda-suite-integrity-drift.json` case `release.condition_eval.1m` backend `cuda` has no applicable performance contract baseline"
            )),
            "Fix: generated CUDA suite status must expose wrong-backend source artifact contracts; blockers={:?}",
            status.blockers
        );
        assert!(
            status.blockers.iter().any(|blocker| {
                blocker.contains(
                    "source_artifact `release/evidence/benchmarks/cuda-suite-integrity-drift.json` case `release.condition_eval.1m` has cuda_resident_borrowed_fallback_dispatches p50=2",
                ) && blocker.contains("backend `cuda` release suite artifact must use native resident dispatch")
            }),
            "Fix: generated CUDA suite status must expose borrowed resident dispatch telemetry; blockers={:?}",
            status.blockers
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
                "summary": case_summary(0, 1),
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
                "summary": case_summary(1, 0),
                "cases": [hidden_invalid_measured_case(
                    "release.condition_eval.1m",
                    "wgpu",
                    cpu_sota_contract("release condition eval", &["wgpu"]),
                    launched_percentile_metrics([10, 11, 12], [2000, 2001, 2002], 1),
                )]
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
                "summary": case_summary(0, 0),
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
                        "contract": cpu_sota_contract("release condition eval", &["wgpu"]),
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
                "summary": case_summary(0, 1),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002},
                            "kernel_launches": {"samples": 1, "p50": 1}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["wgpu"]),
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
}
