use std::collections::BTreeSet;
use std::path::Path;

use super::super::checks::*;
use super::super::types::{GateMode, Requirement};

const REQUIREMENT_ID: &str = "public-launch";

pub(super) fn check(
    requirement: &Requirement,
    base_dir: &Path,
    mode: GateMode,
    failures: &mut Vec<String>,
) {
    for command in ["release-evidence", "vyre-release-gate", "launch-state"] {
        if !requirement.evidence.iter().any(|evidence| {
            evidence.contains("cargo_full")
                && evidence.contains("xtask")
                && evidence.contains(command)
        }) {
            failures.push(format!(
                "requirement `{REQUIREMENT_ID}` must include the cargo_full {command} command as concrete evidence"
            ));
        }
    }

    let Some(state) =
        first_json_evidence(requirement, base_dir, "public-launch-state.json", failures)
    else {
        return;
    };
    check_launch_state(&state, mode, failures);

    let Some(run) =
        first_json_evidence(requirement, base_dir, "release-evidence-run.json", failures)
    else {
        return;
    };
    check_release_evidence_run(requirement, &run, failures);
}

fn check_launch_state(state: &serde_json::Value, mode: GateMode, failures: &mut Vec<String>) {
    if state
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(2)
    {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` launch state schema_version must be 2"
        ));
    }

    let (expected_state, expected_completion, expected_action_status, expected_blockers) =
        match mode {
            GateMode::Final => ("public_launch_complete", "complete", "complete", 0),
            GateMode::Prepublish => (
                "prepublish_release_ready",
                "not_complete_until_external_actions_are_approved_and_done",
                "blocked_pending_user_approval",
                crate::release::launch_contract::required_external_actions().len(),
            ),
        };
    if state
        .get("current_state")
        .and_then(serde_json::Value::as_str)
        != Some(expected_state)
    {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` launch state must be `{expected_state}`"
        ));
    }
    if state
        .get("completion_status")
        .and_then(serde_json::Value::as_str)
        != Some(expected_completion)
    {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` completion status must be `{expected_completion}`"
        ));
    }
    if !crate::release::repo_boundary::has_single_public_repository(state) {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` must name only `{}` as public_repository",
            crate::release::repo_boundary::vyre_public_repository()
        ));
    }

    let expected_gates = [
        ("version_matrix", "pass"),
        ("metadata_matrix", "pass"),
        ("feature_matrix", "pass"),
        ("package_readiness", "pass"),
    ];
    let prepublish_gates = state.get("prepublish_gates");
    for (gate, expected) in expected_gates {
        if prepublish_gates
            .and_then(|gates| gates.get(gate))
            .and_then(serde_json::Value::as_str)
            != Some(expected)
        {
            failures.push(format!(
                "requirement `{REQUIREMENT_ID}` prepublication gate `{gate}` must be `{expected}`"
            ));
        }
    }

    let actions = state
        .get("external_actions")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let required = crate::release::launch_contract::required_external_actions();
    if actions.len() != required.len() {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` launch state has {} external action(s), expected {}",
            actions.len(),
            required.len()
        ));
    }
    let mut seen = BTreeSet::new();
    for required_action in required {
        let matching = actions
            .iter()
            .filter(|action| {
                action.get("action").and_then(serde_json::Value::as_str) == Some(required_action)
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 || !seen.insert(required_action) {
            failures.push(format!(
                "requirement `{REQUIREMENT_ID}` must contain external action `{required_action}` exactly once"
            ));
            continue;
        }
        let action = matching[0];
        if action.get("status").and_then(serde_json::Value::as_str) != Some(expected_action_status)
        {
            failures.push(format!(
                "requirement `{REQUIREMENT_ID}` external action `{required_action}` must be `{expected_action_status}`"
            ));
        }
        if action
            .get("evidence")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|evidence| evidence.trim().is_empty())
        {
            failures.push(format!(
                "requirement `{REQUIREMENT_ID}` external action `{required_action}` lacks evidence"
            ));
        }
    }

    let blocker_count = state
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blocker_count != expected_blockers {
        failures.push(format!(
            "requirement `{REQUIREMENT_ID}` launch state has {blocker_count} blocker(s), expected {expected_blockers}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch_state(mode: GateMode) -> serde_json::Value {
        let final_launch = matches!(mode, GateMode::Final);
        let action_status = if final_launch {
            "complete"
        } else {
            "blocked_pending_user_approval"
        };
        serde_json::json!({
            "schema_version": 2,
            "current_state": if final_launch { "public_launch_complete" } else { "prepublish_release_ready" },
            "completion_status": if final_launch { "complete" } else { "not_complete_until_external_actions_are_approved_and_done" },
            "public_repository": crate::release::repo_boundary::vyre_public_repository(),
            "prepublish_gates": {
                "version_matrix": "pass",
                "metadata_matrix": "pass",
                "feature_matrix": "pass",
                "package_readiness": "pass",
                "vyre_release_gate": "prepublish-pass"
            },
            "external_actions": crate::release::launch_contract::required_external_actions()
                .iter()
                .map(|action| serde_json::json!({
                    "action": action,
                    "status": action_status,
                    "evidence": "verified release action"
                }))
                .collect::<Vec<_>>(),
            "blockers": if final_launch {
                Vec::<String>::new()
            } else {
                crate::release::launch_contract::required_external_actions()
                    .iter()
                    .map(|action| format!("pending approval: {action}"))
                    .collect::<Vec<_>>()
            }
        })
    }

    #[test]
    fn accepts_prepublish_and_final_launch_states() {
        for mode in [GateMode::Prepublish, GateMode::Final] {
            let mut failures = Vec::new();
            check_launch_state(&launch_state(mode), mode, &mut failures);
            assert!(failures.is_empty(), "{failures:#?}");
        }
    }

    #[test]
    fn final_launch_rejects_pending_external_actions() {
        let mut failures = Vec::new();
        check_launch_state(
            &launch_state(GateMode::Prepublish),
            GateMode::Final,
            &mut failures,
        );
        assert!(failures
            .iter()
            .any(|failure| failure.contains("external action")));
        assert!(failures.iter().any(|failure| failure.contains("blocker")));
    }
}
