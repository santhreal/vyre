pub(crate) fn check_backend_feature_marker_id(
    requirement_id: &str,
    matrix: &serde_json::Value,
    field: &str,
    required_id: &str,
    failures: &mut Vec<String>,
) {
    let Some(markers) = matrix.get(field).and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing `{field}`"
        ));
        return;
    };
    let Some(marker) = markers
        .iter()
        .find(|marker| marker.get("id").and_then(serde_json::Value::as_str) == Some(required_id))
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` is missing required marker `{required_id}`"
        ));
        return;
    };
    if marker.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` does not exist"
        ));
    }
    let missing_tokens = marker
        .get("missing_tokens")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_tokens != 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` reports {missing_tokens} missing token(s)"
        ));
    }
    let unresolved_markers = marker
        .get("unresolved_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_markers != 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix `{field}` marker `{required_id}` reports {unresolved_markers} unresolved marker(s)"
        ));
    }
}
pub(crate) fn check_backend_conformance_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let schema_version = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` schema_version={schema_version}; expected schema>=2",
            requirement.id
        ));
    }
    let expected_backend = match suffix {
        "cuda-conformance.json" => Some("cuda"),
        "wgpu-conformance.json" => Some("wgpu"),
        "reference-conformance.json" => Some("cpu-ref"),
        _ => None,
    };
    if let Some(expected) = expected_backend {
        let backend_id = report.get("backend_id").and_then(serde_json::Value::as_str);
        if backend_id != Some(expected) {
            failures.push(format!(
                "requirement `{}` backend conformance `{suffix}` reports backend `{:?}`, expected `{expected}`",
                requirement.id,
                backend_id
            ));
        }
    }
    let total_pairs = report
        .get("total_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let failed_pairs = report
        .get("failed_pairs")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let distinct_op_count = report
        .get("distinct_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_required_op_count = report
        .get("catalog_required_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let catalog_covered_op_count = report
        .get("catalog_covered_op_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_catalog_ops = report
        .get("missing_catalog_ops")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_blocked_release_count = report
        .get("op_matrix_blocked_release_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let release_backend_row_count = report
        .get("release_backend_row_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let missing_release_backend_rows = report
        .get("missing_release_backend_rows")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let op_matrix_errors = report
        .get("op_matrix_errors")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if op_matrix_errors != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {op_matrix_errors} OP_MATRIX read/parse error(s)",
            requirement.id
        ));
    }
    if total_pairs == 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports zero op pairs",
            requirement.id
        ));
    }
    if total_pairs < 49 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {total_pairs} op pair(s), below release floor 49",
            requirement.id
        ));
    }
    if distinct_op_count < 49 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {distinct_op_count} distinct op id(s), below release floor 49",
            requirement.id
        ));
    }
    if catalog_required_op_count == 0
        || catalog_covered_op_count != catalog_required_op_count
        || missing_catalog_ops != 0
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` covers {catalog_covered_op_count}/{catalog_required_op_count} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}",
            requirement.id
        ));
    }
    if op_matrix_blocked_release_count != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {op_matrix_blocked_release_count} OP_MATRIX release backend row(s) marked blocked_release",
            requirement.id
        ));
    }
    let expected_release_backend_rows = catalog_required_op_count.saturating_mul(3);
    if release_backend_row_count < expected_release_backend_rows
        || missing_release_backend_rows != 0
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` has release_backend_row_count={release_backend_row_count}, expected {expected_release_backend_rows}, missing_release_backend_rows={missing_release_backend_rows}",
            requirement.id
        ));
    }
    if failed_pairs != 0 {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports {failed_pairs} failed pair(s)",
            requirement.id
        ));
    }
    if report
        .get("duplicate_op_ids")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|duplicates| !duplicates.is_empty())
    {
        failures.push(format!(
            "requirement `{}` backend conformance `{suffix}` reports duplicate op id(s)",
            requirement.id
        ));
    }
    check_duplicate_object_rows(
        &report,
        "pairs",
        "op_id",
        &format!(
            "requirement `{}` backend conformance `{suffix}` has duplicate pair op_id rows",
            requirement.id
        ),
        failures,
    );
    if let (Some(expected), Some(pairs)) = (
        expected_backend,
        report.get("pairs").and_then(serde_json::Value::as_array),
    ) {
        for pair in pairs {
            let op_id = pair
                .get("op_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            let backend_id = pair.get("backend_id").and_then(serde_json::Value::as_str);
            if backend_id != Some(expected) {
                failures.push(format!(
                    "requirement `{}` backend conformance `{suffix}` pair `{op_id}` reports backend `{:?}`, expected `{expected}`",
                    requirement.id,
                    backend_id
                ));
            }
        }
    }
}

#[cfg(test)]
mod backend_conformance_tests {
    use super::*;

    #[test]
    fn backend_conformance_rejects_duplicate_pair_op_ids() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temp workspace for backend conformance duplicate pair test.");
        let report = serde_json::json!({
            "schema_version": 2,
            "backend_id": "cuda",
            "total_pairs": 49,
            "failed_pairs": 0,
            "distinct_op_count": 49,
            "catalog_required_op_count": 49,
            "catalog_covered_op_count": 49,
            "missing_catalog_ops": [],
            "op_matrix_blocked_release_count": 0,
            "release_backend_row_count": 147,
            "missing_release_backend_rows": [],
            "op_matrix_errors": [],
            "duplicate_op_ids": [],
            "pairs": [
                {"op_id": "vyre.add", "backend_id": "cuda"},
                {"op_id": "vyre.add", "backend_id": "cuda"}
            ]
        });
        std::fs::write(dir.path().join("cuda-conformance.json"), report.to_string())
            .expect("Fix: write backend conformance duplicate pair fixture.");
        let requirement = Requirement {
            id: "conformance-hard-gate".to_string(),
            title: "conformance".to_string(),
            status: "required".to_string(),
            evidence: vec!["cuda-conformance.json".to_string()],
            minimum_evidence: 1,
        };
        let mut failures = Vec::new();

        check_backend_conformance_report(
            &requirement,
            dir.path(),
            "cuda-conformance.json",
            &mut failures,
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("duplicate pair op_id rows: vyre.add")),
            "Fix: backend conformance gate must reject duplicate pairs[].op_id even when duplicate_op_ids claims clean evidence; failures={failures:?}"
        );
    }
}
