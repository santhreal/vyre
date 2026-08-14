use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) =
        first_json_evidence(requirement, base_dir, "conformance-matrix.json", failures)
    else {
        return;
    };
    xtask::release::conformance_evidence_semantics::inspect_conformance_matrix(
        "requirement `conformance-hard-gate` matrix",
        &matrix,
        failures,
    );
    for suffix in [
        "cuda-conformance.json",
        "wgpu-conformance.json",
        "reference-conformance.json",
        "release-gate-log.json",
    ] {
        check_json_evidence_has_no_blockers(requirement, base_dir, suffix, failures);
    }
    for suffix in [
        "cuda-conformance.json",
        "wgpu-conformance.json",
        "reference-conformance.json",
    ] {
        check_backend_conformance_report(requirement, base_dir, suffix, failures);
    }
    if let Some(log) = first_json_evidence(requirement, base_dir, "release-gate-log.json", failures)
    {
        let schema_version = log
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if schema_version < 2 {
            failures.push(format!(
                "requirement `conformance-hard-gate` release log schema_version={schema_version}; expected schema>=2"
            ));
        }
        let requested = log
            .get("requested_backends")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        for backend in ["cuda", "wgpu", "cpu-ref"] {
            if !requested
                .iter()
                .any(|entry| entry.as_str() == Some(backend))
            {
                failures.push(format!(
                    "requirement `conformance-hard-gate` release log requested_backends is missing `{backend}`"
                ));
            }
        }
        let statuses = log
            .get("artifact_statuses")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        for artifact in [
            "cuda-conformance.json",
            "wgpu-conformance.json",
            "reference-conformance.json",
        ] {
            if !statuses.iter().any(|status| {
                status.get("path").and_then(serde_json::Value::as_str) == Some(artifact)
                    && status.get("exists").and_then(serde_json::Value::as_bool) == Some(true)
                    && status
                        .get("bytes")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        > 0
                    && status
                        .get("read_error")
                        .is_some_and(serde_json::Value::is_null)
            }) {
                failures.push(format!(
                    "requirement `conformance-hard-gate` release log does not prove non-empty readable artifact `{artifact}`"
                ));
            }
        }
    }
}
