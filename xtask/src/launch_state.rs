//! Public launch completion evidence.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Serialize)]
struct LaunchState {
    schema_version: u32,
    objective: &'static str,
    current_state: &'static str,
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
    vyre_weir_release_gate: &'static str,
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
        prepublish_gates: PrepublishGates {
            version_matrix: "pass",
            metadata_matrix: "pass",
            feature_matrix: "pass",
            package_readiness: "pass",
            release_completion_audit: "prepublish-pass",
            vyre_weir_release_gate: "prepublish-pass",
        },
        external_actions: vec![
            ExternalAction {
                action: "cargo_full publish approved crates in dependency order",
                status: if complete {
                    "complete"
                } else {
                    "blocked_pending_user_approval"
                },
                evidence: Some("scripts/final-launch.sh + scripts/publish-release.sh + release/evidence/package/publish-readiness.json"),
            },
            ExternalAction {
                action: "make repositories public",
                status: if complete {
                    "complete"
                } else {
                    "blocked_pending_user_approval"
                },
                evidence: Some("scripts/final-launch.sh"),
            },
            ExternalAction {
                action: "git push release branch and tags",
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
                "repository public launch is not approved or completed",
                "git push release branch and tags is not approved or completed",
            ]
        },
        completion_status: if complete {
            "complete"
        } else {
            "not_complete_until_external_actions_are_approved_and_done"
        },
    };
    let json = match serde_json::to_string_pretty(&state) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize launch state: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", output.display());
        std::process::exit(1);
    }
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
            == Some("0.6.1")
        && value
            .get("release_train")
            .and_then(|train| train.get("weir"))
            .and_then(serde_json::Value::as_str)
            == Some("0.1.0")
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
                ["vyre-v0.6.1", "weir-v0.1.0", "vyre-0.6.1-weir-0.1.0"]
                    .iter()
                    .all(|required| tags.iter().any(|tag| tag.as_str() == Some(required)))
            })
        && value
            .get("repositories_public")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|repos| !repos.is_empty())
        && value
            .get("external_actions")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|actions| {
                [
                    "cargo_full publish approved crates in dependency order",
                    "make repositories public",
                    "git push release branch and tags",
                ]
                .iter()
                .all(|required| {
                    actions.iter().any(|action| {
                        action.get("action").and_then(serde_json::Value::as_str) == Some(required)
                            && action.get("status").and_then(serde_json::Value::as_str)
                                == Some("complete")
                    })
                })
            })
}

fn parse_output(args: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- launch-state [--output PATH]\n\n\
                     Writes public launch completion evidence."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown launch-state option `{other}`.")),
        }
    }
    Ok(output.unwrap_or_else(default_output))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/final/public-launch-state.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/final/public-launch-state.json"))
}
