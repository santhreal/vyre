//! The gate runner and the wiring meta-check.
//!
//! 41 subcommands were registered and 9 were ever invoked. The other 32 gates
//! compiled, passed review, and judged nothing, so every rule they encoded was
//! decorative. Nothing failed when that happened, which is the defect this
//! module closes: the registry, the pinned baseline, and the workflows must now
//! agree, and disagreement is a hard failure.
//!
//! A gate that reports findings is pinned at its present finding count rather
//! than excused. The pin ratchets: more findings than the pin fails, fewer is
//! reported so the pin can be lowered. No pin makes a failing gate legal and no
//! owner sentence buys it time: a gate that exits nonzero is a failure, so
//! progress lands in the code the gate judges instead of in this file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use serde::Deserialize;

use crate::subcommands::{self, Kind, Subcommand};

/// Pinned finding count for one gate.
///
/// `deny_unknown_fields` is load-bearing. This file used to carry `status` and
/// `owner` per row, which together let a failing gate stay legal indefinitely
/// behind a prose excuse; three gates sat red that way for a fortnight while
/// the sweep reported that every gate held its baseline. A row that still
/// carries either field now fails to load instead of being ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    name: String,
    output_lines: usize,
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
    passed: bool,
    output_lines: usize,
}

fn execute(entry: &Subcommand) -> Observed {
    let output = Command::new(crate::delegate::dispatcher())
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
        passed: output.status.success(),
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
        if path
            .extension()
            .is_none_or(|ext| ext != "yml" && ext != "yaml")
        {
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

fn render_baseline(rows: &[(String, Observed)]) -> String {
    let mut text = String::from(
        "# Pinned finding count of every registered gate, written by\n\
         # `xtask gates --write-baseline`.\n\
         #\n\
         # `output_lines` is the combined stdout and stderr line count, used as the\n\
         # finding count. More lines than the pin fails; fewer is reported so the pin\n\
         # can be lowered here. A gate that exits nonzero fails outright: no row makes\n\
         # a failing gate legal, and the writer refuses to record one.\n",
    );
    for (name, observed) in rows {
        text.push_str("\n[[gate]]\n");
        text.push_str(&format!("name = \"{name}\"\n"));
        text.push_str(&format!("output_lines = {}\n", observed.output_lines));
    }
    text
}

/// Run the gate sweep.
pub(crate) fn run(args: &[String]) {
    let root = crate::checkout::checkout_root();

    if args.iter().any(|argument| argument == "--list") {
        for entry in subcommands::gates() {
            println!("{} {}", entry.name, entry.ci_args.join(" "));
        }
        return;
    }

    if args.iter().any(|argument| argument == "--write-baseline") {
        let mut rows = Vec::new();
        let mut failing = Vec::new();
        for entry in subcommands::gates() {
            let observed = execute(entry);
            println!(
                "{}: {} ({} lines)",
                entry.name,
                if observed.passed { "green" } else { "red" },
                observed.output_lines
            );
            if !observed.passed {
                failing.push(entry.name);
            }
            rows.push((entry.name.to_string(), observed));
        }
        if !failing.is_empty() {
            eprintln!(
                "Fix: {} gate(s) fail and a baseline may not record a failure: {}. Fix what they report, then write the baseline.",
                failing.len(),
                failing.join(", ")
            );
            process::exit(1);
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
        if !observed.passed {
            failures.push(format!(
                "gate `{}` failed with {} output line(s); fix what it reports",
                entry.name, observed.output_lines
            ));
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
                "{}: green ({} lines, improved from {}); lower the pin in xtask/gate-baselines.toml",
                entry.name, observed.output_lines, pin.output_lines
            );
            continue;
        }
        println!("{}: green ({} lines)", entry.name, pin.output_lines);
    }

    if !failures.is_empty() {
        eprintln!("gates: {} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        process::exit(1);
    }
    println!(
        "gates: {} registered gate(s) hold their baseline",
        subcommands::gates().len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(name: &str) -> Baseline {
        Baseline {
            name: name.to_string(),
            output_lines: 0,
        }
    }

    /// WHY: this is the exact defect. A registered gate with no baseline row is
    /// a gate CI never runs, and that must not be expressible.
    #[test]
    fn a_gate_with_no_baseline_row_is_a_failure() {
        let failures = wiring_failures(&crate::checkout::checkout_root(), &[]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("has no row in xtask/gate-baselines.toml")),
            "an empty baseline must fail every gate, got {failures:?}"
        );
    }

    /// WHY: `status = "red"` beside an `owner` sentence is exactly how three
    /// gates stayed failing for a fortnight while this sweep printed that every
    /// gate held its baseline. Both fields are gone, and a file that still
    /// carries either has to fail to load rather than have it silently ignored,
    /// which is what a default serde derive would do. Each field is offered on
    /// its own: a row carrying both proves only that one of them was caught.
    #[test]
    fn a_baseline_row_that_pins_a_retired_field_is_rejected() {
        for (field, row) in [
            ("status", "status = \"red\"\n"),
            ("owner", "owner = \"PR-26\"\n"),
        ] {
            let text = format!("[[gate]]\nname = \"gate1\"\noutput_lines = 0\n{row}");

            let parsed = toml::from_str::<BaselineFile>(&text);
            assert!(parsed.is_err(), "a row carrying `{field}` must not parse");
            let error = parsed.unwrap_err().to_string();

            assert!(
                error.contains(field),
                "the diagnostic must name the rejected field `{field}`, got {error}"
            );
        }
    }

    /// WHY: a pin naming a deleted or renamed subcommand silently stops
    /// covering anything.
    #[test]
    fn a_pin_for_an_unregistered_subcommand_is_a_failure() {
        let failures =
            wiring_failures(&crate::checkout::checkout_root(), &[pin("no-such-gate")]);
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
        let failures = wiring_failures(&crate::checkout::checkout_root(), &[pin("shrink")]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("not a gate")),
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
