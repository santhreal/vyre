//! The gate runner and the wiring meta-check.
//!
//! 41 subcommands were registered and 9 were ever invoked. The other 32 gates
//! compiled, passed review, and judged nothing, so every rule they encoded was
//! decorative. Nothing failed when that happened, which is the defect this
//! module closes: the registry, the pinned baseline, and the workflows must now
//! agree, and disagreement is a hard failure.
//!
//! A gate that currently reports findings is pinned at its present result
//! rather than excused. The pin ratchets: more findings than the pin fails,
//! fewer is reported so the pin can be lowered, and a red gate that starts
//! passing fails until its pin is flipped to green. Progress therefore lands in
//! this file instead of quietly widening the tolerance.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use serde::Deserialize;

use crate::subcommands::{self, Kind, Subcommand};

/// Workspace root, resolved from the xtask manifest directory.
#[must_use]
pub(crate) fn workspace_root() -> PathBuf {
    crate::checkout::checkout_root()
}

/// Pinned result for one gate.
#[derive(Debug, Deserialize)]
struct Baseline {
    name: String,
    status: String,
    output_lines: usize,
    /// PR that is expected to clear a red gate. Required while it is red.
    #[serde(default)]
    owner: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BaselineFile {
    #[serde(default)]
    gate: Vec<Baseline>,
}

fn baseline_path(root: &Path) -> PathBuf {
    root.join("xtask/gate-baselines.toml")
}

fn load_baselines(root: &Path) -> Vec<Baseline> {
    let path = baseline_path(root);
    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!(
            "Fix: cannot read {}: {error}. Regenerate it with `xtask gates --write-baseline`.",
            path.display()
        );
        process::exit(1);
    });
    let parsed: BaselineFile = toml::from_str(&text).unwrap_or_else(|error| {
        eprintln!("Fix: cannot parse {}: {error}", path.display());
        process::exit(1);
    });
    parsed.gate
}

/// Observed result of running one gate.
struct Observed {
    status: &'static str,
    output_lines: usize,
}

fn execute(entry: &Subcommand) -> Observed {
    let exe = std::env::current_exe().unwrap_or_else(|error| {
        eprintln!("Fix: cannot resolve the running xtask binary: {error}");
        process::exit(1);
    });
    let output = Command::new(exe)
        .arg(entry.name)
        .args(entry.ci_args)
        .output()
        .unwrap_or_else(|error| {
            eprintln!("Fix: cannot run `xtask {}`: {error}", entry.name);
            process::exit(1);
        });
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Observed {
        status: if output.status.success() {
            "green"
        } else {
            "red"
        },
        output_lines: text.lines().count(),
    }
}

/// Every `-- <name>` invocation appearing in an in-repo workflow.
fn workflow_invocations(root: &Path) -> Vec<String> {
    let dir = root.join(".github/workflows");
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(&dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "yml" && ext != "yaml") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            let mut rest = line;
            while let Some(at) = rest.find("-- ") {
                rest = &rest[at + 3..];
                let name: String = rest
                    .chars()
                    .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
                    .collect();
                if !name.is_empty() {
                    found.push(name);
                }
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

/// An owner has to name something a reader can go and look at. The baseline
/// writer emits `unassigned` for a newly red gate, which satisfies "a value is
/// present" while carrying none of the meaning, so the placeholders it and its
/// predecessors produce are rejected by name.
fn is_real_owner(owner: &str) -> bool {
    const PLACEHOLDERS: [&str; 5] = ["unassigned", "unknown", "none", "tbd", "todo"];
    !owner.is_empty() && !PLACEHOLDERS.contains(&owner.to_ascii_lowercase().as_str())
}

/// Reject a registry, baseline and workflow set that do not agree.
///
/// This is the meta-check. It is what makes an unwired gate impossible rather
/// than merely discouraged.
fn wiring_failures(root: &Path, baselines: &[Baseline]) -> Vec<String> {
    let mut failures = Vec::new();

    for entry in subcommands::gates() {
        if !baselines.iter().any(|pin| pin.name == entry.name) {
            failures.push(format!(
                "gate `{}` has no row in xtask/gate-baselines.toml; add one or reclassify it",
                entry.name
            ));
        }
    }
    for pin in baselines {
        match subcommands::find(&pin.name) {
            None => failures.push(format!(
                "xtask/gate-baselines.toml pins `{}`, which is not a registered subcommand",
                pin.name
            )),
            Some(entry) if entry.kind != Kind::Gate => failures.push(format!(
                "xtask/gate-baselines.toml pins `{}`, which is not a gate",
                pin.name
            )),
            Some(_) => {}
        }
        let owner = pin.owner.as_deref().map(str::trim).unwrap_or("");
        if pin.status == "red" && !is_real_owner(owner) {
            failures.push(format!(
                "gate `{}` is pinned red with no owner; name the PR that clears it",
                pin.name
            ));
        }
        if pin.status != "red" && pin.status != "green" {
            failures.push(format!(
                "gate `{}` has status `{}`; use `green` or `red`",
                pin.name, pin.status
            ));
        }
    }

    let invoked = workflow_invocations(root);
    for entry in subcommands::SUBCOMMANDS {
        let owed_a_workflow = entry.kind == Kind::Evidence || entry.kind == Kind::Composite;
        if owed_a_workflow && !invoked.iter().any(|name| name == entry.name) {
            failures.push(format!(
                "`{}` is not invoked by any workflow in .github/workflows",
                entry.name
            ));
        }
    }
    for name in &invoked {
        if subcommands::find(name).is_none() && name.starts_with("vyre") {
            failures.push(format!(
                "a workflow invokes `xtask {name}`, which is not a registered subcommand"
            ));
        }
    }

    failures
}

fn render_baseline(rows: &[(String, Observed, Option<String>)]) -> String {
    let mut text = String::from(
        "# Pinned result of every registered gate, written by `xtask gates --write-baseline`.\n\
         #\n\
         # `status` is the exit result and `output_lines` is the combined stdout and\n\
         # stderr line count, used as the finding count. More lines than the pin\n\
         # fails; fewer is reported so the pin can be lowered here. A red gate that\n\
         # starts passing fails until its status is flipped, and must name the PR\n\
         # that clears it while it is still red.\n",
    );
    for (name, observed, owner) in rows {
        text.push_str("\n[[gate]]\n");
        text.push_str(&format!("name = \"{name}\"\n"));
        text.push_str(&format!("status = \"{}\"\n", observed.status));
        text.push_str(&format!("output_lines = {}\n", observed.output_lines));
        if let Some(owner) = owner {
            text.push_str(&format!("owner = \"{owner}\"\n"));
        }
    }
    text
}

/// Run the gate sweep.
pub(crate) fn run(args: &[String]) {
    let root = workspace_root();

    if args.iter().any(|argument| argument == "--list") {
        for entry in subcommands::gates() {
            println!("{} {}", entry.name, entry.ci_args.join(" "));
        }
        return;
    }

    if args.iter().any(|argument| argument == "--write-baseline") {
        let previous = fs::read_to_string(baseline_path(&root))
            .ok()
            .and_then(|text| toml::from_str::<BaselineFile>(&text).ok())
            .map(|file| file.gate)
            .unwrap_or_default();
        let mut rows = Vec::new();
        for entry in subcommands::gates() {
            let observed = execute(entry);
            // A green gate carries no owner. Keeping one from a previous run
            // left stale assignments on gates that had already been cleared.
            let owner = (observed.status == "red")
                .then(|| {
                    previous
                        .iter()
                        .find(|pin| pin.name == entry.name)
                        .and_then(|pin| pin.owner.clone())
                })
                .flatten()
                .or_else(|| (observed.status == "red").then(|| "unassigned".to_string()));
            println!(
                "{}: {} ({} lines)",
                entry.name, observed.status, observed.output_lines
            );
            rows.push((entry.name.to_string(), observed, owner));
        }
        let path = baseline_path(&root);
        fs::write(&path, render_baseline(&rows)).unwrap_or_else(|error| {
            eprintln!("Fix: cannot write {}: {error}", path.display());
            process::exit(1);
        });
        println!("wrote {}", path.display());
        return;
    }

    let baselines = load_baselines(&root);
    let mut failures = wiring_failures(&root, &baselines);

    for entry in subcommands::gates() {
        let Some(pin) = baselines.iter().find(|pin| pin.name == entry.name) else {
            continue;
        };
        let observed = execute(entry);
        if observed.status != pin.status {
            let detail = if observed.status == "red" {
                format!(
                    "gate `{}` regressed from {} to red; fix the finding it reports",
                    entry.name, pin.status
                )
            } else {
                format!(
                    "gate `{}` now passes but is pinned red; set status = \"green\" and output_lines = {} in xtask/gate-baselines.toml",
                    entry.name, observed.output_lines
                )
            };
            failures.push(detail);
            continue;
        }
        if observed.output_lines > pin.output_lines {
            failures.push(format!(
                "gate `{}` reported {} output lines against a pinned {}; fix the new finding",
                entry.name, observed.output_lines, pin.output_lines
            ));
            continue;
        }
        if observed.output_lines < pin.output_lines {
            println!(
                "{}: {} ({} lines, improved from {}); lower the pin in xtask/gate-baselines.toml",
                entry.name, pin.status, observed.output_lines, pin.output_lines
            );
            continue;
        }
        println!("{}: {} ({} lines)", entry.name, pin.status, pin.output_lines);
    }

    if !failures.is_empty() {
        eprintln!("gates: {} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        process::exit(1);
    }
    println!("gates: {} registered gate(s) hold their baseline", subcommands::gates().len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(name: &str, status: &str, owner: Option<&str>) -> Baseline {
        Baseline {
            name: name.to_string(),
            status: status.to_string(),
            output_lines: 0,
            owner: owner.map(str::to_string),
        }
    }

    /// WHY: this is the exact defect. A registered gate with no baseline row is
    /// a gate CI never runs, and that must not be expressible.
    #[test]
    fn a_gate_with_no_baseline_row_is_a_failure() {
        let failures = wiring_failures(&workspace_root(), &[]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("has no row in xtask/gate-baselines.toml")),
            "an empty baseline must fail every gate, got {failures:?}"
        );
    }

    /// WHY: a red pin with no owner is an excuse rather than a plan, and it is
    /// how a temporary exemption becomes permanent.
    #[test]
    fn a_red_pin_without_an_owner_is_a_failure() {
        let failures = wiring_failures(&workspace_root(), &[pin("gate1", "red", None)]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("pinned red with no owner")),
            "got {failures:?}"
        );
    }

    /// WHY: the baseline writer stamps `unassigned` on every newly red gate, so
    /// a check that only tests for presence is satisfied by the placeholder it
    /// just wrote and never asks anyone to own anything. Every placeholder the
    /// writer or a human reaches for has to be rejected by name, in any case.
    #[test]
    fn a_red_pin_owned_by_a_placeholder_is_a_failure() {
        for placeholder in ["unassigned", "UNASSIGNED", "unknown", "none", "tbd", "todo", "  "] {
            let failures =
                wiring_failures(&workspace_root(), &[pin("gate1", "red", Some(placeholder))]);
            assert!(
                failures
                    .iter()
                    .any(|failure| failure.contains("pinned red with no owner")),
                "placeholder owner {placeholder:?} must not satisfy the meta-check, got {failures:?}"
            );
        }

        let failures = wiring_failures(&workspace_root(), &[pin("gate1", "red", Some("PR-26"))]);
        assert!(
            !failures
                .iter()
                .any(|failure| failure.contains("pinned red with no owner")),
            "a named owner must satisfy the meta-check, got {failures:?}"
        );
    }

    /// WHY: a pin naming a deleted or renamed subcommand silently stops
    /// covering anything.
    #[test]
    fn a_pin_for_an_unregistered_subcommand_is_a_failure() {
        let failures = wiring_failures(&workspace_root(), &[pin("no-such-gate", "green", None)]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("not a registered subcommand")),
            "got {failures:?}"
        );
    }

    /// WHY: pinning a dev tool or evidence generator as a gate would make the
    /// sweep run something CI cannot judge.
    #[test]
    fn a_pin_for_a_non_gate_is_a_failure() {
        let failures = wiring_failures(&workspace_root(), &[pin("shrink", "green", None)]);
        assert!(
            failures.iter().any(|failure| failure.contains("not a gate")),
            "got {failures:?}"
        );
    }

    /// WHY: the sweep runs the real binary, so the gate list must be non-empty
    /// or the whole check passes vacuously.
    #[test]
    fn the_gate_list_is_not_empty() {
        assert!(subcommands::gates().len() > 10);
    }
}
