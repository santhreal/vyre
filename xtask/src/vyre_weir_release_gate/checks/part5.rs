pub(crate) fn required_marker_ids_for_suffix(suffix: &str) -> &'static [&'static str] {
    if suffix == "alias-aware-dse.json" {
        &[
            "alias-aware-dse-entrypoint",
            "reaching-def-dse-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
        ]
    } else if suffix == "alias-aware-stlf.json" {
        &[
            "alias-aware-stlf-entrypoint",
            "reaching-def-stlf-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
            "dataflow-analysis-stlf-firing-test",
        ]
    } else if suffix == "alias-aware-licm.json" {
        &[
            "alias-aware-licm-entrypoint",
            "reaching-def-licm-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if suffix == "alias-aware-fusion-fission.json" {
        &[
            "alias-aware-loop-fusion-entrypoint",
            "reaching-def-loop-fusion-entrypoint",
            "alias-aware-loop-fission-entrypoint",
            "reaching-def-loop-fission-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
        ]
    } else if suffix == "weir-facts-pass-firing.json" {
        &[
            "alias-aware-dse-entrypoint",
            "reaching-def-dse-entrypoint",
            "alias-aware-stlf-entrypoint",
            "reaching-def-stlf-entrypoint",
            "alias-aware-licm-entrypoint",
            "reaching-def-licm-entrypoint",
            "alias-aware-loop-fusion-entrypoint",
            "reaching-def-loop-fusion-entrypoint",
            "alias-aware-loop-fission-entrypoint",
            "reaching-def-loop-fission-entrypoint",
            "weir-alias-analysis-api",
            "weir-reaching-def-analysis-api",
            "dataflow-analysis-pipeline-entrypoint",
        ]
    } else if suffix == "egraph-saturation-matrix.json"
        || suffix == "egraph-semantic-contracts.json"
    {
        &[
            "egraph-saturation",
            "egraph-canonical-pipeline-entrypoint",
            "egraph-algebraic-reassociation",
            "egraph-bitwise-reassociation",
        ]
    } else {
        &[]
    }
}
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
pub(crate) fn check_parser_contract_evidence(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let expected_component = if suffix == "vyrec-cli-contracts.json" {
        "vyrec"
    } else {
        suffix.strip_suffix("-contracts.json").unwrap_or(suffix)
    };
    let component_id = report
        .get("component_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if component_id != expected_component {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has component_id `{component_id}`, expected `{expected_component}`",
            requirement.id
        ));
    }
    if report
        .get("role")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|role| role.is_empty())
    {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has an empty role",
            requirement.id
        ));
    }
    if report
        .get("root")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|root| root.is_empty())
    {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has an empty root",
            requirement.id
        ));
    }
    let required_terms = report
        .get("required_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_terms == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required terms",
            requirement.id
        ));
    }
    let missing_terms = report
        .get("missing_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_terms != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_terms} missing term(s)",
            requirement.id
        ));
    }
    let required_contract_topics = report
        .get("required_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_contract_topics == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required contract topics",
            requirement.id
        ));
    }
    let missing_contract_topics = report
        .get("missing_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_contract_topics != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_contract_topics} missing contract topic(s)",
            requirement.id
        ));
    }
    let required_test_categories = report
        .get("required_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_test_categories == 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required test categories",
            requirement.id
        ));
    }
    let missing_test_categories = report
        .get("missing_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_test_categories != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {missing_test_categories} missing test categor(ies)",
            requirement.id
        ));
    }
    let required_evidence_trees = report
        .get("required_evidence_trees")
        .and_then(serde_json::Value::as_array);
    if required_evidence_trees.is_none_or(|trees| trees.len() < 3) {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` must list tests, benches, and fuzz evidence trees",
            requirement.id
        ));
    }
    if let Some(trees) = required_evidence_trees {
        for tree in trees {
            let tree_name = tree
                .get("tree")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if tree.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` does not exist",
                    requirement.id
                ));
            }
            let source_bytes = tree
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if source_bytes == 0 {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` has zero source bytes",
                    requirement.id
                ));
            }
            let unreadable = tree
                .get("unreadable_file_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if unreadable != 0 {
                failures.push(format!(
                    "requirement `{}` parser contract `{suffix}` evidence tree `{tree_name}` has {unreadable} unreadable source file(s)",
                    requirement.id
                ));
            }
        }
    }
    let unresolved_ownership_markers = report
        .get("unresolved_ownership_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_ownership_markers != 0 {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` reports {unresolved_ownership_markers} unresolved ownership marker(s)",
            requirement.id
        ));
    }
    let Some(required_files) = report
        .get("required_files")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has no required_files array",
            requirement.id
        ));
        return;
    };
    if required_files.is_empty() {
        failures.push(format!(
            "requirement `{}` parser contract `{suffix}` has zero required file(s)",
            requirement.id
        ));
    }
    for file in required_files {
        let path = file
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if file.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` does not exist",
                requirement.id
            ));
        }
        if file
            .get("source_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` is empty",
                requirement.id
            ));
        }
        let read_error = file.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            failures.push(format!(
                "requirement `{}` parser contract `{suffix}` required file `{path}` read_error={}",
                requirement.id,
                read_error
                    .map(serde_json::Value::to_string)
                    .unwrap_or_else(|| "<missing>".to_string())
            ));
        }
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
