fn inspect_parser_contract_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let component_id = value
        .get("component_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let expected = if evidence.ends_with("vyrec-cli-contracts.json") {
        "vyrec"
    } else {
        evidence
            .rsplit('/')
            .next()
            .and_then(|file| file.strip_suffix("-contracts.json"))
            .unwrap_or("")
    };
    if component_id != expected {
        blockers.push(format!(
            "{evidence}: component_id `{component_id}` does not match expected `{expected}`"
        ));
    }
    if value
        .get("role")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|role| role.is_empty())
    {
        blockers.push(format!("{evidence}: parser contract role is empty"));
    }
    if value
        .get("root")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|root| root.is_empty())
    {
        blockers.push(format!("{evidence}: parser contract root is empty"));
    }
    let required_terms = value
        .get("required_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_terms == 0 {
        blockers.push(format!("{evidence}: parser contract has no required_terms"));
    }
    let missing_terms = value
        .get("missing_terms")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_terms != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_terms} missing term(s)"
        ));
    }
    let required_contract_topics = value
        .get("required_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_contract_topics == 0 {
        blockers.push(format!(
            "{evidence}: parser contract has no required_contract_topics"
        ));
    }
    let missing_contract_topics = value
        .get("missing_contract_topics")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_contract_topics != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_contract_topics} missing contract topic(s)"
        ));
    }
    let required_test_categories = value
        .get("required_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_test_categories == 0 {
        blockers.push(format!(
            "{evidence}: parser contract has no required_test_categories"
        ));
    }
    let missing_test_categories = value
        .get("missing_test_categories")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_test_categories != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {missing_test_categories} missing test categor(ies)"
        ));
    }
    let required_evidence_trees = value
        .get("required_evidence_trees")
        .and_then(serde_json::Value::as_array);
    if required_evidence_trees.is_none_or(|trees| trees.len() < 3) {
        blockers.push(format!(
            "{evidence}: parser contract must list tests, benches, and fuzz evidence trees"
        ));
    }
    if let Some(trees) = required_evidence_trees {
        for tree in trees {
            let tree_name = tree
                .get("tree")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if tree.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` does not exist"
                ));
            }
            let source_bytes = tree
                .get("source_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if source_bytes == 0 {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` has zero source bytes"
                ));
            }
            let unreadable = tree
                .get("unreadable_file_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::MAX);
            if unreadable != 0 {
                blockers.push(format!(
                    "{evidence}: parser contract evidence tree `{tree_name}` has {unreadable} unreadable source file(s)"
                ));
            }
        }
    }
    let unresolved_ownership_markers = value
        .get("unresolved_ownership_markers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if unresolved_ownership_markers != 0 {
        blockers.push(format!(
            "{evidence}: parser contract reports {unresolved_ownership_markers} unresolved ownership marker(s)"
        ));
    }
    let Some(files) = value
        .get("required_files")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!(
            "{evidence}: parser contract missing required_files"
        ));
        return;
    };
    if files.is_empty() {
        blockers.push(format!(
            "{evidence}: parser contract required_files is empty"
        ));
    }
    for file in files {
        let path = file
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if file.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: required file `{path}` does not exist"));
        }
        if file
            .get("source_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: required file `{path}` is empty"));
        }
        let read_error = file.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: required file `{path}` read_error={}",
                read_error
                    .map(serde_json::Value::to_string)
                    .unwrap_or_else(|| "<missing>".to_string())
            ));
        }
    }
}

fn inspect_cpu_100x_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        != Some("cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    for required_case in [
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
    ] {
        if !cases
            .iter()
            .any(|case| case.get("id").and_then(serde_json::Value::as_str) == Some(required_case))
        {
            blockers.push(format!(
                "{evidence}: missing required CPU-SOTA 100x proof case `{required_case}`"
            ));
        }
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda") {
            blockers.push(format!("{evidence}: case `{id}` backend_id must be cuda"));
        }
        if !case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            blockers.push(format!(
                "{evidence}: case `{id}` must pass its performance contract"
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall = metrics.and_then(active_gpu_metric_p50);
        let baseline = metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
        require_benchmark_metric_percentiles(evidence, id, metrics, "baseline_wall_ns", blockers);
        match (wall, baseline) {
            (Some(wall), Some(baseline)) if wall > 0.0 && baseline / wall >= 100.0 => {}
            (Some(wall), Some(baseline)) if wall > 0.0 => blockers.push(format!(
                "{evidence}: case `{id}` end-to-end p50 speedup is {:.2}x, needs 100.00x",
                baseline / wall
            )),
            _ => blockers.push(format!(
                "{evidence}: case `{id}` must include p50 wall_ns and baseline_wall_ns"
            )),
        }
    }
}

fn case_has_cpu_sota_contract(case: &serde_json::Value, required_speedup: f64) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(serde_json::Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
            })
        })
}

fn metric_p50(value: Option<&serde_json::Value>) -> Option<f64> {
    metric_percentile(value, "p50")
}

fn active_gpu_metric_p50(metrics: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
    metric_p50(metrics.get("dispatch_ns"))
        .or_else(|| metric_p50(metrics.get("kernel_execute_ns")))
        .or_else(|| metric_p50(metrics.get("wall_ns")))
}

fn metric_percentile(value: Option<&serde_json::Value>, percentile: &str) -> Option<f64> {
    value
        .and_then(|value| value.get(percentile))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            value
                .and_then(|value| value.get(percentile))
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as f64)
        })
}

fn metric_samples(value: Option<&serde_json::Value>) -> Option<u64> {
    value?.get("samples").and_then(serde_json::Value::as_u64)
}

fn require_benchmark_metric_percentiles(
    evidence: &str,
    case_id: &str,
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    metric_name: &str,
    blockers: &mut Vec<String>,
) {
    for percentile in ["p50", "p95", "p99"] {
        let value =
            metrics.and_then(|metrics| metric_percentile(metrics.get(metric_name), percentile));
        if !value.is_some_and(|value| value > 0.0) {
            blockers.push(format!(
                "{evidence}: case `{case_id}` must include positive {percentile} {metric_name}"
            ));
        }
    }
}

fn inspect_version_matrix_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("requested_vyre_release")
        .and_then(serde_json::Value::as_str)
        != Some("0.6.1")
    {
        blockers.push(format!(
            "{evidence}: requested_vyre_release must be `0.6.1`"
        ));
    }
    if value
        .get("requested_weir_release")
        .and_then(serde_json::Value::as_str)
        != Some("0.1.0")
    {
        blockers.push(format!(
            "{evidence}: requested_weir_release must be `0.1.0`"
        ));
    }
    if value
        .get("release_doc_tag_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: release_doc_tag_findings must exist and be empty"
        ));
    }
    if value
        .get("release_note_token_findings")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|findings| !findings.is_empty())
    {
        blockers.push(format!(
            "{evidence}: release_note_token_findings must exist and be empty"
        ));
    }
    if value
        .get("missing_required_release_packages")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|packages| !packages.is_empty())
    {
        blockers.push(format!(
            "{evidence}: missing_required_release_packages must exist and be empty"
        ));
    }
    let required_release_packages = value
        .get("required_release_packages")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required_package in [
        "vyre@0.6.1",
        "vyre-driver-cuda@0.6.1",
        "vyre-driver-wgpu@0.6.1",
        "weir@0.1.0",
        "vyrec@0.1.0",
        "vyre-frontend-c@0.6.1",
    ] {
        if !required_release_packages
            .iter()
            .any(|package| package.as_str() == Some(required_package))
        {
            blockers.push(format!(
                "{evidence}: required_release_packages must include `{required_package}`"
            ));
        }
    }
    let Some(tag_story) = value
        .get("tag_story")
        .and_then(serde_json::Value::as_object)
    else {
        blockers.push(format!("{evidence}: missing tag_story"));
        return;
    };
    for (field, expected) in [
        ("vyre_rc_tag", "vyre-v0.6.1-rc.1"),
        ("weir_rc_tag", "weir-v0.1.0-rc.1"),
        (
            "combined_release_train_rc_tag",
            "vyre-0.6.1-weir-0.1.0-rc.1",
        ),
        ("vyre_tag", "vyre-v0.6.1"),
        ("weir_tag", "weir-v0.1.0"),
        ("combined_release_train_tag", "vyre-0.6.1-weir-0.1.0"),
    ] {
        if tag_story.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            blockers.push(format!(
                "{evidence}: tag_story.{field} must be `{expected}`"
            ));
        }
    }
    for required in [
        "vyre 0.6.1",
        "weir 0.1.0",
        "vyre-driver-cuda@0.6.1",
        "vyre-driver-wgpu@0.6.1",
        "vyre-v0.6.1-rc.1",
        "weir-v0.1.0-rc.1",
        "vyre-0.6.1-weir-0.1.0-rc.1",
        "vyre-v0.6.1",
        "weir-v0.1.0",
        "vyre-0.6.1-weir-0.1.0",
    ] {
        let present = tag_story
            .get("required_in_release_notes")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|entries| entries.iter().any(|entry| entry.as_str() == Some(required)));
        if !present {
            blockers.push(format!(
                "{evidence}: tag_story.required_in_release_notes is missing `{required}`"
            ));
        }
    }
}

fn inspect_markdown_evidence(evidence: &str, path: &Path, blockers: &mut Vec<String>) {
    let text = match read_text_bounded(path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{evidence}: failed to read markdown evidence: {error}"
            ));
            return;
        }
    };
    if text.trim().is_empty() {
        blockers.push(format!("{evidence}: markdown evidence is empty"));
    }
    for marker in [
        "status: blocked",
        "status: open",
        "status: pending",
        "todo",
        "fixme",
        "placeholder",
        "stub",
        "tbd",
        "to be filled",
    ] {
        for line in text.lines() {
            let lowered = line.to_ascii_lowercase();
            if markdown_line_is_release_rule_text(&lowered) {
                continue;
            }
            if lowered.contains(marker) {
                blockers.push(format!(
                    "{evidence}: markdown evidence contains unresolved marker `{marker}`"
                ));
                break;
            }
        }
    }
    if evidence.starts_with("evidence/docs/") && !text.contains("Evidence sources:") {
        blockers.push(format!(
            "{evidence}: generated docs evidence does not list evidence sources"
        ));
    }
}

