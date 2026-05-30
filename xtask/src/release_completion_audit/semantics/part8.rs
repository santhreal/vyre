fn inspect_metadata_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    for (field, label) in [
        ("publishable_package_count", "publishable package"),
        ("vyre_package_count", "Vyre package"),
        ("weir_package_count", "Weir package"),
        (
            "non_publishable_release_surface_count",
            "non-publishable release-surface package",
        ),
    ] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: contains zero {label}(s)"));
        }
    }
    if value
        .get("parser_release_surface_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        < 2
    {
        blockers.push(format!(
            "{evidence}: parser_release_surface_count must cover vyrec and vyre-frontend-c"
        ));
    }
    let missing_required = value
        .get("missing_required_release_surfaces")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        blockers.push(format!(
            "{evidence}: missing_required_release_surfaces has {missing_required} entrie(s), expected zero"
        ));
    }
    if value
        .get("root_patch_section_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX)
        != 0
    {
        blockers.push(format!(
            "{evidence}: root_patch_section_count must be present and zero"
        ));
    }
    let Some(packages) = value.get("packages").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing packages array"));
        return;
    };
    if !packages.iter().any(|package| {
        package.get("name").and_then(serde_json::Value::as_str) == Some("vyrec")
            && package.get("version").and_then(serde_json::Value::as_str) == Some("0.1.0")
            && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
            && package
                .get("release_surface")
                .and_then(serde_json::Value::as_str)
                == Some("parser-cli")
    }) {
        blockers.push(format!(
            "{evidence}: missing vyrec 0.1.0 parser-cli package metadata with README.md"
        ));
    }
    if !packages.iter().any(|package| {
        package.get("name").and_then(serde_json::Value::as_str) == Some("vyre-frontend-c")
            && package.get("version").and_then(serde_json::Value::as_str) == Some("0.6.1")
            && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
            && package
                .get("release_kind")
                .and_then(serde_json::Value::as_str)
                == Some("non-publishable-release-surface")
            && package
                .get("release_surface")
                .and_then(serde_json::Value::as_str)
                == Some("c-frontend")
    }) {
        blockers.push(format!(
            "{evidence}: missing vyre-frontend-c 0.6.1 c-frontend non-publishable release-surface metadata with README.md"
        ));
    }
    for (package_name, backend_surface) in [
        ("vyre-driver-cuda", "cuda-backend"),
        ("vyre-driver-wgpu", "wgpu-backend"),
    ] {
        if !packages.iter().any(|package| {
            package.get("name").and_then(serde_json::Value::as_str) == Some(package_name)
                && package.get("version").and_then(serde_json::Value::as_str) == Some("0.6.1")
                && package.get("readme").and_then(serde_json::Value::as_str) == Some("README.md")
                && package
                    .get("release_kind")
                    .and_then(serde_json::Value::as_str)
                    == Some("publishable-crate")
                && package
                    .get("release_surface")
                    .and_then(serde_json::Value::as_str)
                    == Some(backend_surface)
        }) {
            blockers.push(format!(
                "{evidence}: missing {package_name} 0.6.1 publishable {backend_surface} release-surface metadata with README.md"
            ));
        }
    }
    for package in packages {
        let name = package
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let release_kind = package
            .get("release_kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if release_kind == "internal-tooling" {
            continue;
        }
        let release_group = package
            .get("release_group")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let expected = package
            .get("expected_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let version = package
            .get("version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if expected.is_empty() || version != expected {
            blockers.push(format!(
                "{evidence}: package `{name}` release_group `{release_group}` has version `{version}`, expected `{expected}`"
            ));
        }
        if package
            .get("example_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: release package `{name}` has zero examples or README usage blocks"
            ));
        }
        if release_kind == "publishable-crate"
            && package
                .get("has_runnable_example")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
        {
            blockers.push(format!(
                "{evidence}: publishable release package `{name}` has no runnable examples/*.rs"
            ));
        }
        if release_kind == "publishable-crate"
            && package
                .get("has_api_referencing_example")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
        {
            blockers.push(format!(
                "{evidence}: publishable release package `{name}` has no API-referencing examples/*.rs"
            ));
        }
    }
}

fn inspect_package_readiness_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!(
            "{evidence}: package readiness still reports {blocker_count} blocker(s)"
        ));
    }
    if value
        .get("release_train")
        .and_then(|train| train.get("cuda_release_path"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!("{evidence}: cuda_release_path must be true"));
    }
    let publish_order = value
        .get("publish_order")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if publish_order.len() < 20 {
        blockers.push(format!(
            "{evidence}: publish_order contains {} package(s), expected the full release train",
            publish_order.len()
        ));
    }
    for required in [
        "vyre-macros",
        "vyre-spec",
        "vyre-foundation",
        "vyre-driver-cuda",
        "vyre-driver-wgpu",
        "vyre",
        "vyre-harness",
        "weir",
        "vyre-libs",
    ] {
        if !publish_order
            .iter()
            .any(|entry| entry.get("package").and_then(serde_json::Value::as_str) == Some(required))
        {
            blockers.push(format!("{evidence}: publish_order is missing `{required}`"));
        }
    }
    let missing_metadata = value
        .get("missing_metadata_packages")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let extra_metadata = value
        .get("extra_metadata_packages")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_metadata != 0 || extra_metadata != 0 {
        blockers.push(format!(
            "{evidence}: publish_order and metadata disagree: {missing_metadata} missing, {extra_metadata} extra"
        ));
    }
    if value
        .get("dependency_order_edges")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        blockers.push(format!("{evidence}: dependency_order_edges is empty"));
    }
    if value
        .get("versioned_local_dependencies")
        .and_then(serde_json::Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        blockers.push(format!("{evidence}: versioned_local_dependencies is empty"));
    }
    let verify_passed = value
        .get("package_verify_passed")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["vyre-macros@0.6.1", "vyre-spec@0.6.1", "vyre-lints@0.6.1"] {
        if !verify_passed
            .iter()
            .any(|entry| entry.as_str() == Some(required))
        {
            blockers.push(format!(
                "{evidence}: package_verify_passed is missing `{required}`"
            ));
        }
    }
    let non_publish_surfaces = value
        .get("non_publish_release_surfaces")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in ["vyre-frontend-c", "vyrec"] {
        if !non_publish_surfaces
            .iter()
            .any(|entry| entry.get("package").and_then(serde_json::Value::as_str) == Some(required))
        {
            blockers.push(format!(
                "{evidence}: non_publish_release_surfaces is missing `{required}`"
            ));
        }
    }
}

fn inspect_public_launch_state_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let blocker_count = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != 0 {
        blockers.push(format!(
            "{evidence}: public launch is incomplete with {blocker_count} blocker(s)"
        ));
    }
    if value
        .get("completion_status")
        .and_then(serde_json::Value::as_str)
        != Some("complete")
    {
        blockers.push(format!("{evidence}: completion_status is not `complete`"));
    }
    let external_actions = value
        .get("external_actions")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in [
        "cargo publish approved crates in dependency order",
        "make repositories public",
        "git push release branch and tags",
    ] {
        let Some(action) = external_actions.iter().find(|action| {
            action.get("action").and_then(serde_json::Value::as_str) == Some(required)
        }) else {
            blockers.push(format!(
                "{evidence}: external action `{required}` is missing"
            ));
            continue;
        };
        if action.get("status").and_then(serde_json::Value::as_str) != Some("complete") {
            blockers.push(format!(
                "{evidence}: external action `{required}` is not complete"
            ));
        }
    }
}

fn inspect_docs_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("curated_proof_docs_preserved")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        blockers.push(format!(
            "{evidence}: curated_proof_docs_preserved must be true"
        ));
    }
    let Some(docs) = value.get("docs").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing docs array"));
        return;
    };
    if docs.is_empty() {
        blockers.push(format!("{evidence}: docs array is empty"));
        return;
    }
    if value
        .get("limitation_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: limitation_findings must exist and be empty"
        ));
    }
    for doc in docs {
        let id = doc
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if doc.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: required doc `{id}` does not exist"));
        }
        if doc
            .get("contains_release_evidence_rule")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` does not reference release evidence"
            ));
        }
        if doc
            .get("evidence_artifact_ref_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has zero concrete evidence artifact references"
            ));
        }
        if doc
            .get("missing_evidence_artifact_refs")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|refs| !refs.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` references missing evidence artifacts"
            ));
        }
        if doc
            .get("missing_topics")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|topics| !topics.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has missing topics"
            ));
        }
        if doc
            .get("unresolved_markers")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|markers| !markers.is_empty())
        {
            blockers.push(format!(
                "{evidence}: required doc `{id}` has unresolved markers"
            ));
        }
    }
}

fn inspect_release_axes_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let source_artifacts = value
        .get("source_artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if source_artifacts < 12 {
        blockers.push(format!(
            "{evidence}: source_artifacts has {source_artifacts} entrie(s), needs at least 12"
        ));
    }
    for field in [
        "warm_us_per_file",
        "cold_pipeline_build_ms",
        "gbs_scan_throughput",
        "ulp_drift_max",
        "max_vram_mib",
    ] {
        if value.get(field).is_none_or(serde_json::Value::is_null) {
            blockers.push(format!("{evidence}: missing benchmark axis `{field}`"));
        }
    }
}

