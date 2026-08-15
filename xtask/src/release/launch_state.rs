//! Hold the public launch state to the completion marker the launch writes.

use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::artifact_gate::Inspection;
use crate::release::launch_contract::{required_external_actions, GIT_PUSH_ACTION, PUBLISH_ACTION};
use crate::release::release_train;
use crate::release::repo_boundary;

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/final/public-launch-state.json";

/// The marker `final-launch` writes when the external actions are done.
const COMPLETION_MARKER: &str = "release/evidence/final/public-launch-completion.json";

#[derive(Debug, Serialize)]
struct LaunchState {
    schema_version: u32,
    objective: &'static str,
    current_state: &'static str,
    public_repository: &'static str,
    prepublish_gates: PrepublishGates,
    external_actions: Vec<ExternalAction>,
    blockers: Vec<&'static str>,
    completion_status: &'static str,
}

#[derive(Debug, Serialize)]
struct PrepublishGates {
    version_matrix: &'static str,
    metadata_matrix: &'static str,
    feature_matrix: &'static str,
    package_readiness: &'static str,
}

/// The four prepublish gates, and the artifact each one has to have left behind.
///
/// These four fields used to be the literal string `pass`, four times, written
/// without consulting anything. The artifact asserted that four gates had
/// passed on every run including runs where all four had never been invoked,
/// which is the worst kind of evidence: a reader saw four passes and there was
/// nothing behind them. Each is now the state of that gate's own artifact.
const PREPUBLISH_GATES: &[(&str, &str)] = &[
    ("version-matrix", "release/evidence/version/version-matrix.json"),
    ("metadata-matrix", "release/evidence/metadata/metadata-matrix.json"),
    ("feature-matrix", "release/evidence/metadata/feature-matrix.json"),
    ("package-readiness", "release/evidence/package/publish-readiness.json"),
];

#[derive(Debug, Serialize)]
struct ExternalAction {
    action: &'static str,
    status: &'static str,
    evidence: Option<&'static str>,
}

crate::artifact_gate! {
    /// Holds the public launch state to the completion marker and the gate artifacts.
    LaunchStateGate,
    name: "launch-state",
    help: "Regenerate release/evidence/final/public-launch-state.json from the launch completion \
       marker and the four prepublish gate artifacts, and report each line the committed \
       artifact disagrees on. Proves the recorded launch state matches the marker on disk, and \
       that each prepublish gate left an artifact carrying no blockers. Proves nothing about \
       whether the external actions were really performed: the marker is written by the launch \
       script and this gate reads it, it does not contact crates.io or the git remote.",
    inspect: |ctx| inspect(&ctx.root),
}

/// What the marker and the gate artifacts say, and the artifact recording it.
fn inspect(root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let complete = completion_marker_complete(&root.join(COMPLETION_MARKER));
    let status = if complete {
        "complete"
    } else {
        "blocked_pending_user_approval"
    };
    let state = LaunchState {
        schema_version: 2,
        objective: "publish the canonical Vyre release artifacts",
        current_state: if complete {
            "public_launch_complete"
        } else {
            "prepublish_release_ready"
        },
        public_repository: repo_boundary::vyre_public_repository(),
        prepublish_gates: prepublish_gates(root, &mut inspection),
        external_actions: vec![
            ExternalAction {
                action: PUBLISH_ACTION,
                status,
                evidence: Some("scripts/final-launch.sh + scripts/publish-release.sh + release/evidence/package/publish-readiness.json"),
            },
            ExternalAction {
                action: repo_boundary::verify_public_repo_action(),
                status,
                evidence: Some(repo_boundary::verify_public_repo_evidence()),
            },
            ExternalAction {
                action: GIT_PUSH_ACTION,
                status,
                evidence: Some("scripts/final-launch.sh"),
            },
        ],
        blockers: if complete {
            Vec::new()
        } else {
            vec![
                "cargo_full publish is not approved or completed",
                "vyre repository public verification is not completed",
                "git push release branch and tags is not approved or completed",
            ]
        },
        completion_status: if complete {
            "complete"
        } else {
            "not_complete_until_external_actions_are_approved_and_done"
        },
    };
    for blocker in &state.blockers {
        inspection.blocked(
            ARTIFACT,
            (*blocker).to_string(),
            format!(
                "Perform the external action, then run `scripts/final-launch.sh` so it records \
                 `{COMPLETION_MARKER}`. Until that marker names this release train with every \
                 required action complete, the launch is not closed."
            ),
        );
    }
    inspection.generates(ARTIFACT, &state);
    inspection
}

/// The state of each prepublish gate, read from the artifact that gate owns.
fn prepublish_gates(root: &Path, inspection: &mut Inspection) -> PrepublishGates {
    let mut status = Vec::new();
    for (gate, artifact) in PREPUBLISH_GATES {
        status.push(prepublish_gate_status(root, gate, artifact, inspection));
    }
    PrepublishGates {
        version_matrix: status[0],
        metadata_matrix: status[1],
        feature_matrix: status[2],
        package_readiness: status[3],
    }
}

/// Whether one prepublish gate left an artifact, and whether it carried blockers.
fn prepublish_gate_status(
    root: &Path,
    gate: &str,
    artifact: &str,
    inspection: &mut Inspection,
) -> &'static str {
    let Ok(text) = fs::read_to_string(root.join(artifact)) else {
        inspection.blocked(
            ARTIFACT,
            format!("prepublish gate `{gate}` has written no `{artifact}`"),
            format!("Run `./cargo_full run --bin xtask -- {gate} --write` and commit the artifact."),
        );
        return "missing";
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        inspection.blocked(
            ARTIFACT,
            format!("prepublish gate artifact `{artifact}` is not valid JSON"),
            format!("Run `./cargo_full run --bin xtask -- {gate} --write` to rewrite it."),
        );
        return "unreadable";
    };
    let blockers = value
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if blockers == 0 {
        return "pass";
    }
    inspection.blocked(
        ARTIFACT,
        format!("prepublish gate `{gate}` recorded {blockers} blocker(s) in `{artifact}`"),
        format!("Run `./cargo_full run --bin xtask -- {gate}` and clear what it reports."),
    );
    "blocked"
}

fn completion_marker_complete(path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(2)
        && value
            .get("release_train")
            .and_then(|train| train.get("vyre"))
            .and_then(serde_json::Value::as_str)
            == Some(release_train::vyre_version())
        && value
            .get("completion_status")
            .and_then(serde_json::Value::as_str)
            == Some("complete")
        && value
            .get("git")
            .and_then(|git| git.get("branch"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|branch| !branch.trim().is_empty())
        && value
            .get("git")
            .and_then(|git| git.get("tags"))
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tags| {
                release_train::tag_creation_order()
                    .iter()
                    .all(|required| tags.iter().any(|tag| tag.as_str() == Some(*required)))
            })
        && repo_boundary::has_single_public_repository(&value)
        && value
            .get("external_actions")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|actions| {
                required_external_actions().iter().all(|required| {
                    actions.iter().any(|action| {
                        action.get("action").and_then(serde_json::Value::as_str) == Some(*required)
                            && action.get("status").and_then(serde_json::Value::as_str)
                                == Some("complete")
                    })
                })
            })
}

#[cfg(test)]
mod tests {
    use super::completion_marker_complete;

    /// Build a completed launch marker for whatever release train is declared.
    ///
    /// The fixture is derived rather than pasted so cutting a release never leaves this
    /// suite asserting against a version that no longer ships. It previously hardcoded
    /// 0.6.3 and started failing the moment the train moved to 0.6.6.
    fn completed_marker_json(vyre: &str) -> String {
        let tags = crate::release::release_train::tag_creation_order()
            .iter()
            .map(|tag| format!("      \"{tag}\""))
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            r#"{{
  "schema_version": 2,
  "release_train": {{
    "vyre": "{vyre}"
  }},
  "git": {{
    "branch": "main",
    "tags": [
{tags}
    ]
  }},
  "public_repository": "santhreal/vyre",
  "external_actions": [
    {{
      "action": "cargo_full publish approved crates in dependency order",
      "status": "complete"
    }},
    {{
      "action": "verify vyre repository is public",
      "status": "complete"
    }},
    {{
      "action": "git push release branch and tags",
      "status": "complete"
    }}
  ],
  "completion_status": "complete"
}}"#
        )
    }

    /// A marker matching the declared release train, with every external action complete,
    /// is what `final-launch` writes and `launch-state` must accept.
    #[test]
    fn completion_marker_accepts_the_declared_release_train_with_required_actions() {
        let dir = tempfile::tempdir().expect("Fix: create launch-state test directory.");
        let marker = dir.path().join("public-launch-completion.json");
        std::fs::write(
            &marker,
            completed_marker_json(crate::release::release_train::vyre_version()),
        )
        .expect("Fix: write launch completion marker fixture.");

        assert!(
            completion_marker_complete(&marker),
            "Fix: launch-state must accept the completed {} marker that final-launch writes.",
            crate::release::release_train::vyre_version()
        );
    }

    /// Launch evidence from an earlier release train must not satisfy the current one.
    ///
    /// Accepting a stale marker would let a release claim its public-launch actions were
    /// done when they were done for a previous version.
    #[test]
    fn completion_marker_rejects_a_previous_release_train() {
        let dir = tempfile::tempdir().expect("Fix: create launch-state test directory.");
        let marker = dir.path().join("public-launch-completion.json");
        std::fs::write(&marker, completed_marker_json("0.4.2"))
            .expect("Fix: write stale launch completion marker fixture.");

        assert!(
            !completion_marker_complete(&marker),
            "Fix: launch-state must reject launch evidence from a previous release train."
        );
    }

    /// Final-only tag evidence must fail because the documented ceremony requires every RC tag before final promotion.
    #[test]
    fn completion_marker_rejects_a_missing_release_candidate_tag() {
        let dir = tempfile::tempdir().expect("Fix: create missing RC tag fixture directory.");
        let marker = dir.path().join("public-launch-completion.json");
        let rc_line = format!(
            "      \"{}\",\n",
            crate::release::release_train::vyre_rc_tag()
        );
        let marker_json = completed_marker_json(crate::release::release_train::vyre_version())
            .replacen(&rc_line, "", 1);
        std::fs::write(&marker, marker_json)
            .expect("Fix: write launch completion marker without the Vyre RC tag.");

        assert!(
            !completion_marker_complete(&marker),
            "Fix: launch-state must reject evidence that omits any release candidate tag."
        );
    }

    /// Legacy plural repository evidence must not bypass the singular public-repository boundary.
    #[test]
    fn completion_marker_rejects_legacy_single_public_repository_array() {
        let dir = tempfile::tempdir().expect("Fix: create launch-state test directory.");
        let marker = dir.path().join("public-launch-completion.json");
        std::fs::write(
            &marker,
            r#"{
  "schema_version": 2,
  "release_train": {
    "vyre": "0.6.3"
  },
  "git": {
    "branch": "main",
    "tags": [
      "vyre-v0.6.3"
    ]
  },
  "repositories_public": ["santhreal/vyre"],
  "external_actions": [
    {
      "action": "cargo_full publish approved crates in dependency order",
      "status": "complete"
    },
    {
      "action": "verify vyre repository is public",
      "status": "complete"
    },
    {
      "action": "git push release branch and tags",
      "status": "complete"
    }
  ],
  "completion_status": "complete"
}"#,
        )
        .expect("Fix: write launch completion marker fixture.");

        assert!(
            !completion_marker_complete(&marker),
            "Fix: launch-state must reject legacy repositories_public evidence and require singular public_repository."
        );
    }
}
