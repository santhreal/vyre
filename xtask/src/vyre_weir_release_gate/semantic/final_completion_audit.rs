use std::collections::BTreeSet;

use std::path::Path;

use super::super::checks::*;
use super::super::types::{GateMode, Requirement};

pub(super) fn check(
    requirement: &Requirement,
    base_dir: &Path,
    mode: GateMode,
    failures: &mut Vec<String>,
) {
    if !requirement.evidence.iter().any(|evidence| {
        evidence.contains("cargo_full")
            && evidence.contains("xtask")
            && evidence.contains("release-evidence")
    }) {
        failures.push(
            "requirement `final-completion-audit` must include the cargo_full release-evidence command as concrete evidence"
                .to_string(),
        );
    }
    if !requirement.evidence.iter().any(|evidence| {
        evidence.contains("cargo_full")
            && evidence.contains("xtask")
            && evidence.contains("release-completion-audit")
    }) {
        failures.push(
            "requirement `final-completion-audit` must include the cargo_full release-completion-audit command as concrete evidence"
                .to_string(),
        );
    }
    if !requirement.evidence.iter().any(|evidence| {
        evidence.contains("cargo_full")
            && evidence.contains("xtask")
            && evidence.contains("vyre-release-gate")
    }) {
        failures.push(
            "requirement `final-completion-audit` must include the cargo_full vyre-release-gate command as concrete evidence"
                .to_string(),
        );
    }

    let Some(audit) = first_json_evidence(requirement, base_dir, "completion-audit.json", failures)
    else {
        return;
    };
    match mode {
        GateMode::Final => check_final_audit(&audit, failures),
        GateMode::Prepublish => {
            let Some(launch_state) =
                first_json_evidence(requirement, base_dir, "public-launch-state.json", failures)
            else {
                return;
            };
            check_prepublish_audit(&audit, &launch_state, failures);
        }
    }

    let Some(run) =
        first_json_evidence(requirement, base_dir, "release-evidence-run.json", failures)
    else {
        return;
    };
    check_release_evidence_run(requirement, &run, failures);
}

fn check_final_audit(audit: &serde_json::Value, failures: &mut Vec<String>) {
    let blockers = audit
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let open = audit
        .get("blocked_or_open_requirements")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if blockers != 0 || open != 0 {
        failures.push(format!(
            "requirement `final-completion-audit` still reports {blockers} blocker(s) and {open} open requirement(s)"
        ));
    }
}

fn check_prepublish_audit(
    audit: &serde_json::Value,
    launch_state: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let open = audit
        .get("blocked_or_open_requirements")
        .and_then(serde_json::Value::as_u64);
    if open != Some(1) {
        failures.push(format!(
            "requirement `final-completion-audit` prepublication audit reports {open:?} open requirement(s), expected exactly one external-launch requirement"
        ));
    }

    let requirement_rows = audit
        .get("requirements")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let incomplete = requirement_rows
        .iter()
        .filter(|row| row.get("complete").and_then(serde_json::Value::as_bool) != Some(true))
        .collect::<Vec<_>>();
    if incomplete.len() != 1
        || incomplete[0].get("id").and_then(serde_json::Value::as_str)
            != Some("final-completion-audit")
    {
        failures.push(
            "requirement `final-completion-audit` prepublication audit must leave only `final-completion-audit` incomplete"
                .to_string(),
        );
    } else {
        let row = incomplete[0];
        if row
            .get("missing_evidence")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|missing| !missing.is_empty())
        {
            failures.push(
                "requirement `final-completion-audit` prepublication row has missing evidence"
                    .to_string(),
            );
        }
        let semantic_blockers = row
            .get("semantic_blockers")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if semantic_blockers.is_empty()
            || semantic_blockers.iter().any(|blocker| {
                blocker
                    .as_str()
                    .is_none_or(|text| !text.contains("public-launch-state.json"))
            })
        {
            failures.push(
                "requirement `final-completion-audit` prepublication semantic blockers must refer only to `public-launch-state.json`"
                    .to_string(),
            );
        }
    }

    let blockers = audit
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if blockers.is_empty()
        || blockers.iter().any(|blocker| {
            blocker.as_str().is_none_or(|text| {
                !text.starts_with("final-completion-audit is status `closed`")
                    && !text.contains("public-launch-state.json")
            })
        })
    {
        failures.push(
            "requirement `final-completion-audit` prepublication blockers include a non-launch blocker"
                .to_string(),
        );
    }

    check_prepublish_launch_state(launch_state, failures);
}

fn check_prepublish_launch_state(state: &serde_json::Value, failures: &mut Vec<String>) {
    if state
        .get("current_state")
        .and_then(serde_json::Value::as_str)
        != Some("prepublish_release_ready")
    {
        failures.push(
            "requirement `final-completion-audit` launch state is not `prepublish_release_ready`"
                .to_string(),
        );
    }
    if state
        .get("completion_status")
        .and_then(serde_json::Value::as_str)
        != Some("not_complete_until_external_actions_are_approved_and_done")
    {
        failures.push(
            "requirement `final-completion-audit` launch completion status does not preserve pending external actions"
                .to_string(),
        );
    }
    if !crate::repo_boundary::has_single_public_repository(state) {
        failures.push(format!(
            "requirement `final-completion-audit` launch state must name only `{}` as public_repository",
            crate::repo_boundary::vyre_public_repository()
        ));
    }

    let expected_gates = [
        ("version_matrix", "pass"),
        ("metadata_matrix", "pass"),
        ("feature_matrix", "pass"),
        ("package_readiness", "pass"),
        ("release_completion_audit", "prepublish-pass"),
        ("vyre_weir_release_gate", "prepublish-pass"),
    ];
    let prepublish_gates = state.get("prepublish_gates");
    for (gate, expected) in expected_gates {
        if prepublish_gates
            .and_then(|gates| gates.get(gate))
            .and_then(serde_json::Value::as_str)
            != Some(expected)
        {
            failures.push(format!(
                "requirement `final-completion-audit` prepublication gate `{gate}` is not `{expected}`"
            ));
        }
    }

    let actions = state
        .get("external_actions")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let required = crate::launch_contract::required_external_actions();
    if actions.len() != required.len() {
        failures.push(format!(
            "requirement `final-completion-audit` launch state has {} external action(s), expected {}",
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
                "requirement `final-completion-audit` launch state must contain external action `{required_action}` exactly once"
            ));
            continue;
        }
        let action = matching[0];
        if action.get("status").and_then(serde_json::Value::as_str)
            != Some("blocked_pending_user_approval")
        {
            failures.push(format!(
                "requirement `final-completion-audit` external action `{required_action}` is not blocked pending user approval"
            ));
        }
        if action
            .get("evidence")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|evidence| evidence.trim().is_empty())
        {
            failures.push(format!(
                "requirement `final-completion-audit` external action `{required_action}` lacks evidence"
            ));
        }
    }

    let launch_blockers = state.get("blockers").and_then(serde_json::Value::as_array);
    if launch_blockers.map(Vec::len) != Some(required.len()) {
        failures.push(format!(
            "requirement `final-completion-audit` launch state must report exactly {} approval-gated blocker(s)",
            required.len()
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepublish_launch_state() -> serde_json::Value {
        serde_json::json!({
            "current_state": "prepublish_release_ready",
            "public_repository": crate::repo_boundary::vyre_public_repository(),
            "prepublish_gates": {
                "version_matrix": "pass",
                "metadata_matrix": "pass",
                "feature_matrix": "pass",
                "package_readiness": "pass",
                "release_completion_audit": "prepublish-pass",
                "vyre_weir_release_gate": "prepublish-pass"
            },
            "external_actions": crate::launch_contract::required_external_actions()
                .into_iter()
                .map(|action| serde_json::json!({
                    "action": action,
                    "status": "blocked_pending_user_approval",
                    "evidence": "release evidence"
                }))
                .collect::<Vec<_>>(),
            "blockers": ["publish", "visibility", "push"],
            "completion_status": "not_complete_until_external_actions_are_approved_and_done"
        })
    }

    fn prepublish_audit() -> serde_json::Value {
        serde_json::json!({
            "blocked_or_open_requirements": 1,
            "requirements": [{
                "id": "final-completion-audit",
                "complete": false,
                "missing_evidence": [],
                "semantic_blockers": [
                    "evidence/final/public-launch-state.json: external action is not complete"
                ]
            }],
            "blockers": [
                "final-completion-audit is status `closed` with one semantic blocker",
                "release/evidence/final/public-launch-state.json: external action is not complete"
            ]
        })
    }

    /// A truthful prepublication audit must pass while all three outward
    /// actions remain explicitly blocked on user approval.
    #[test]
    fn prepublish_gate_accepts_only_external_launch_blockers() {
        let mut failures = Vec::new();

        check_prepublish_audit(
            &prepublish_audit(),
            &prepublish_launch_state(),
            &mut failures,
        );

        assert_eq!(failures, Vec::<String>::new());
    }

    /// An unrelated internal blocker must keep prepublication readiness red
    /// even when the launch-state artifact itself is well formed.
    #[test]
    fn prepublish_gate_rejects_internal_audit_blocker() {
        let mut audit = prepublish_audit();
        audit["blockers"]
            .as_array_mut()
            .expect("blocker array")
            .push(serde_json::json!("metadata matrix is stale"));
        let mut failures = Vec::new();

        check_prepublish_audit(&audit, &prepublish_launch_state(), &mut failures);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("non-launch blocker")));
    }

    /// A supposedly ready prepublication state must not accept an outward
    /// action already marked complete without final launch evidence.
    #[test]
    fn prepublish_gate_rejects_completed_external_action() {
        let mut state = prepublish_launch_state();
        state["external_actions"][0]["status"] = serde_json::json!("complete");
        let mut failures = Vec::new();

        check_prepublish_audit(&prepublish_audit(), &state, &mut failures);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("not blocked pending user approval")));
    }
}
