//! The blockers a benchmark evidence file reports about itself.
//!
//! Evidence may declare its own blockers, at the top level and again per suite
//! status row, and a declared blocker is fatal wherever it appears. Reading both
//! places here is what keeps a suite from passing a gate by burying its blocker
//! one level down.

use serde_json::Value;

use super::backend_identity::expected_backend_for_suite_evidence;

pub(crate) fn benchmark_evidence_blocker_issues(evidence: &str, value: &Value) -> Vec<String> {
    let mut blockers = Vec::new();
    let Some(artifact_blockers) = value.get("blockers").and_then(Value::as_array) else {
        blockers.push(format!("`{evidence}` is missing blockers array"));
        return blockers;
    };
    for (index, blocker) in artifact_blockers.iter().enumerate() {
        let blocker = blocker.as_str().unwrap_or("<non-string blocker>");
        blockers.push(format!("`{evidence}` blocker[{index}]: {blocker}"));
    }
    collect_benchmark_suite_status_blocker_issues(evidence, value, &mut blockers);
    blockers
}

fn collect_benchmark_suite_status_blocker_issues(
    evidence: &str,
    value: &Value,
    blockers: &mut Vec<String>,
) {
    let Some(statuses) = value.get("artifact_statuses") else {
        if expected_backend_for_suite_evidence(evidence).is_some() {
            blockers.push(format!("`{evidence}` is missing artifact_statuses array"));
        }
        return;
    };
    let Some(statuses) = statuses.as_array() else {
        blockers.push(format!("`{evidence}` artifact_statuses must be an array"));
        return;
    };
    for (status_index, status) in statuses.iter().enumerate() {
        let status_path = status
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        let Some(status_blockers) = status.get("blockers").and_then(Value::as_array) else {
            blockers.push(format!(
                "`{evidence}` artifact_statuses[{status_index}] `{status_path}` is missing blockers array"
            ));
            continue;
        };
        for (blocker_index, blocker) in status_blockers.iter().enumerate() {
            let blocker = blocker.as_str().unwrap_or("<non-string blocker>");
            blockers.push(format!(
                "`{evidence}` artifact_statuses[{status_index}] `{status_path}` blocker[{blocker_index}]: {blocker}"
            ));
        }
    }
}
