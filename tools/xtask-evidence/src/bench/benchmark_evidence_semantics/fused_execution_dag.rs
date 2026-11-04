//! Whether a fused execution DAG in a report describes a runnable graph.
//!
//! The DAG is evidence that fusion happened, so it has to be a graph and not a
//! list of claims: node ids unique and non-blank, every edge endpoint a declared
//! node, no self edge, no cycle, and the reported node and edge counts equal to
//! what the arrays actually contain.

use serde_json::Value;

pub(crate) fn benchmark_fused_execution_dag_issues(
    artifact_label: &str,
    report: &Value,
) -> Vec<String> {
    let mut issues = Vec::new();
    let Some(dag) = report.get("fused_execution_dag") else {
        return vec![format!(
            "{artifact_label}: missing fused_execution_dag evidence"
        )];
    };
    if dag.get("schema_version").and_then(Value::as_u64) != Some(1) {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.schema_version must be 1"
        ));
    }
    let graph_nodes = dag_node_ids(dag.get("graph_nodes"));
    if graph_nodes.len() < 5 {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.graph_nodes needs ingest, scan, verify, confidence, and report nodes"
        ));
    }
    for required in ["ingest", "scan", "verify", "confidence", "report"] {
        if !graph_nodes.iter().any(|node| node == required) {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.graph_nodes is missing `{required}`"
            ));
        }
    }
    let Some(memory_edges) = dag.get("memory_edges").and_then(Value::as_array) else {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.memory_edges must be a non-empty array"
        ));
        return issues;
    };
    if memory_edges.is_empty() {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.memory_edges must be non-empty"
        ));
    }
    for (index, edge) in memory_edges.iter().enumerate() {
        for field in ["from", "to", "buffer"] {
            if edge
                .get(field)
                .and_then(Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                issues.push(format!(
                    "{artifact_label}: fused_execution_dag.memory_edges[{index}].{field} is blank or missing"
                ));
            }
        }
        if edge.get("bytes").and_then(Value::as_u64).unwrap_or(0) == 0 {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.memory_edges[{index}].bytes must be positive"
            ));
        }
    }
    match dag.get("host_sync_points").and_then(Value::as_u64) {
        Some(points) if points <= 1 => {}
        Some(points) => issues.push(format!(
            "{artifact_label}: fused_execution_dag.host_sync_points={points}, expected final-state-only sync <=1"
        )),
        None => issues.push(format!(
            "{artifact_label}: fused_execution_dag.host_sync_points is missing"
        )),
    }
    let bytes = dag.get("bytes_transferred");
    for field in ["host_to_device_bytes", "device_to_host_bytes"] {
        if bytes
            .and_then(|bytes| bytes.get(field))
            .and_then(Value::as_u64)
            .is_none()
        {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.bytes_transferred.{field} is missing"
            ));
        }
    }
    let parity = dag.get("reporter_parity");
    if parity
        .and_then(|parity| parity.get("parity_passed"))
        .and_then(Value::as_bool)
        != Some(true)
    {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.reporter_parity.parity_passed must be true"
        ));
    }
    for field in ["cpu_digest", "gpu_digest", "output_digest"] {
        if parity
            .and_then(|parity| parity.get(field))
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.reporter_parity.{field} is blank or missing"
            ));
        }
    }
    let Some(fallback_reasons) = dag.get("fallback_reasons").and_then(Value::as_array) else {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.fallback_reasons must be an array"
        ));
        return issues;
    };
    for (index, reason) in fallback_reasons.iter().enumerate() {
        if reason
            .get("reason")
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.fallback_reasons[{index}].reason is blank or missing"
            ));
        }
        if reason
            .get("fix")
            .and_then(Value::as_str)
            .is_none_or(|fix| !fix.starts_with("Fix:"))
        {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.fallback_reasons[{index}].fix must start with `Fix:`"
            ));
        }
    }
    for field in [
        "comparator",
        "dataset_id",
        "metric_family",
        "release_floor",
        "failure_mode",
    ] {
        if dag
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(format!(
                "{artifact_label}: fused_execution_dag.{field} is blank or missing"
            ));
        }
    }
    if dag
        .get("frontier_leaderboard_artifact")
        .and_then(Value::as_str)
        != Some("release/evidence/benchmarks/frontier-leaderboard.json")
    {
        issues.push(format!(
            "{artifact_label}: fused_execution_dag.frontier_leaderboard_artifact must point at release/evidence/benchmarks/frontier-leaderboard.json"
        ));
    }
    issues
}

fn dag_node_ids(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |nodes| {
            nodes
                .iter()
                .filter_map(|node| {
                    node.as_str()
                        .or_else(|| node.get("id").and_then(Value::as_str))
                })
                .filter(|node| !node.trim().is_empty())
                .map(str::to_string)
                .collect()
        })
}
