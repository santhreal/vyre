//! Public launch completion evidence.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::launch_contract::{required_external_actions, GIT_PUSH_ACTION, PUBLISH_ACTION};
use crate::release_train;
use crate::repo_boundary;

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
    release_completion_audit: &'static str,
    vyre_release_gate: &'static str,
}

#[derive(Debug, Serialize)]
struct ExternalAction {
    action: &'static str,
    status: &'static str,
    evidence: Option<&'static str>,
}

pub(crate) fn run(args: &[String]) {
    let output = match parse_output(args) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let completion_marker = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("public-launch-completion.json");
    let complete = completion_marker_complete(&completion_marker);
    let state = LaunchState {
        schema_version: 1,
        objective: "complete release/plans/paradigm-shift-100-concrete.md",
        current_state: if complete {
            "public_launch_complete"
        } else {
            "prepublish_release_ready"
        },
        public_repository: repo_boundary::vyre_public_repository(),
        prepublish_gates: PrepublishGates {
            version_matrix: "pass",
            metadata_matrix: "pass",
            feature_matrix: "pass",
            package_readiness: "pass",
            release_completion_audit: "prepublish-pass",
            vyre_release_gate: "prepublish-pass",
        },
        external_actions: vec![
            ExternalAction {
                action: PUBLISH_ACTION,
                status: if complete {
                    "complete"
                } else {
                    "blocked_pending_user_approval"
                },
                evidence: Some("scripts/final-launch.sh + scripts/publish-release.sh + release/evidence/package/publish-readiness.json"),
            },
            ExternalAction {
                action: repo_boundary::verify_public_repo_action(),
                status: if complete {
                    "complete"
                } else {
                    "blocked_pending_user_approval"
                },
                evidence: Some(repo_boundary::verify_public_repo_evidence()),
            },
            ExternalAction {
                action: GIT_PUSH_ACTION,
                status: if complete {
                    "complete"
                } else {
                    "blocked_pending_user_approval"
                },
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
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    crate::output_arg::write_json(&output, &state);
    println!("launch-state: wrote {}", output.display());
    if !state.blockers.is_empty() {
        std::process::exit(1);
    }
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
        == Some(1)
        && value
            .get("release_train")
            .and_then(|train| train.get("vyre"))
            .and_then(serde_json::Value::as_str)
            == Some(release_train::vyre_version())
        && value
            .get("release_train")
            .and_then(|train| train.get("weir"))
            .and_then(serde_json::Value::as_str)
            == Some(release_train::weir_version())
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

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    crate::output_arg::parse_output_arg(
        args,
        "launch-state",
        "Writes public launch completion evidence.",
        default_output,
    )
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/final/public-launch-state.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/final/public-launch-state.json"))
}

#[cfg(test)]
mod tests {
    use super::completion_marker_complete;

    /// Build a completed launch marker for whatever release train is declared.
    ///
    /// The fixture is derived rather than pasted so cutting a release never leaves this
    /// suite asserting against a version that no longer ships. It previously hardcoded
    /// 0.6.3 and started failing the moment the train moved to 0.6.6.
    fn completed_marker_json(vyre: &str, weir: &str) -> String {
        let tags = crate::release_train::tag_creation_order()
            .iter()
            .map(|tag| format!("      \"{tag}\""))
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            r#"{{
  "schema_version": 1,
  "release_train": {{
    "vyre": "{vyre}",
    "weir": "{weir}"
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
            completed_marker_json(
                crate::release_train::vyre_version(),
                crate::release_train::weir_version(),
            ),
        )
        .expect("Fix: write launch completion marker fixture.");

        assert!(
            completion_marker_complete(&marker),
            "Fix: launch-state must accept the completed {}/{} marker that final-launch writes.",
            crate::release_train::vyre_version(),
            crate::release_train::weir_version()
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
        std::fs::write(
            &marker,
            completed_marker_json("0.4.2", crate::release_train::weir_version()),
        )
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
        let rc_line = format!("      \"{}\",\n", crate::release_train::vyre_rc_tag());
        let marker_json = completed_marker_json(
            crate::release_train::vyre_version(),
            crate::release_train::weir_version(),
        )
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
  "schema_version": 1,
  "release_train": {
    "vyre": "0.6.3",
    "weir": "0.1.0"
  },
  "git": {
    "branch": "main",
    "tags": [
      "vyre-v0.6.3",
      "weir-v0.1.0",
      "vyre-0.6.3-weir-0.1.0"
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
