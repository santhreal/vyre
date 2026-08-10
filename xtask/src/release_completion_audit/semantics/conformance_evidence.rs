fn inspect_conformance_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    crate::conformance_evidence_semantics::inspect_conformance_matrix(
        evidence, value, blockers,
    );
}

fn inspect_release_conformance_log_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}; release conformance log must be schema>=2"
        ));
    }
    inspect_named_blockers(evidence, value, "release conformance log", blockers);
    if !value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| {
            command.contains("cargo_full") && command.contains("release-conformance")
        })
    {
        blockers.push(format!(
            "{evidence}: command must run release-conformance through cargo_full"
        ));
    }
    let requested = value
        .get("requested_backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for backend in ["cuda", "wgpu", "cpu-ref"] {
        if !requested
            .iter()
            .any(|entry| entry.as_str() == Some(backend))
        {
            blockers.push(format!(
                "{evidence}: requested_backends is missing `{backend}`"
            ));
        }
    }
    let statuses = value
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
            blockers.push(format!(
                "{evidence}: does not prove non-empty readable conformance artifact `{artifact}`"
            ));
        }
    }
}

fn inspect_named_blockers(
    evidence: &str,
    value: &serde_json::Value,
    surface: &str,
    blockers: &mut Vec<String>,
) {
    if let Some(surface_blockers) = value.get("blockers").and_then(serde_json::Value::as_array) {
        for blocker in surface_blockers {
            blockers.push(format!(
                "{evidence}: {surface} blocker: {}",
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
}

#[cfg(test)]
mod conformance_evidence_tests {
    use super::*;

    #[test]
    fn completion_audit_rejects_conformance_matrix_blockers() {
        let matrix = serde_json::json!({
            "schema_version": 2,
            "op_count": 49,
            "distinct_op_count": 49,
            "catalog_required_op_count": 49,
            "catalog_covered_op_count": 49,
            "missing_catalog_ops": [],
            "op_matrix_blocked_release_count": 0,
            "op_matrix_errors": [],
            "fixture_input_count": 49,
            "expected_output_count": 49,
            "duplicate_op_ids": [],
            "dispatch_backends": ["cuda", "wgpu", "cpu-ref"],
            "ci_blocking_gate_count": 3,
            "required_ci_statuses": ["conform"],
            "missing_required_ci_statuses": [],
            "ci_status_scan_errors": [],
            "path_filtered_required_workflows": [],
            "missing_required_workflow_triggers": [],
            "missing_fail_closed_fanins": [],
            "ci_gates": [],
            "blockers": ["CUDA conformance artifact failed"]
        });
        let mut blockers = Vec::new();

        inspect_conformance_matrix_semantics("conformance-matrix.json", &matrix, &mut blockers);

        assert!(
            blockers.iter().any(|blocker| blocker.contains(
                "conformance-matrix.json: conformance matrix blocker: CUDA conformance artifact failed"
            )),
            "Fix: completion audit must reject explicit conformance matrix blockers; blockers={blockers:?}"
        );
    }

    #[test]
    fn completion_audit_rejects_release_conformance_log_blockers() {
        let log = serde_json::json!({
            "schema_version": 2,
            "command": "cargo_full run --bin xtask -- release-conformance",
            "requested_backends": ["cuda", "wgpu", "cpu-ref"],
            "artifact_statuses": [
                {"path": "cuda-conformance.json", "exists": true, "bytes": 1, "read_error": null},
                {"path": "wgpu-conformance.json", "exists": true, "bytes": 1, "read_error": null},
                {"path": "reference-conformance.json", "exists": true, "bytes": 1, "read_error": null}
            ],
            "blockers": ["wgpu conformance produced zero op pairs"]
        });
        let mut blockers = Vec::new();

        inspect_release_conformance_log_semantics("release-gate-log.json", &log, &mut blockers);

        assert!(
            blockers.iter().any(|blocker| blocker.contains(
                "release-gate-log.json: release conformance log blocker: wgpu conformance produced zero op pairs"
            )),
            "Fix: completion audit must reject explicit release conformance log blockers; blockers={blockers:?}"
        );
    }
}
