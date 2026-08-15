//! The gate runner and the wiring meta-check.
//!
//! 41 subcommands were registered and 9 were ever invoked. The other 32 gates
//! compiled, passed review, and judged nothing, so every rule they encoded was
//! decorative. Nothing failed when that happened, which is the defect this
//! module closes: the registry, the pinned baseline, and the workflows must
//! agree, and disagreement is a hard failure.
//!
//! The sweep enumerates the registry at run time. A gate registered without a
//! baseline row fails, and a baseline row naming no registered gate fails, so
//! neither list is maintained by hand and neither can quietly fall behind.
//!
//! Registration is not wiring. A gate no workflow selects by name, directly or
//! through a subset a workflow runs, fails the wiring check: `file-size` was
//! red on fourteen source files while nothing in CI named it, so its judgement
//! reached nobody who could act on it. Wiring failures are not pinnable and no
//! baseline row excuses one.
//!
//! A gate that reports findings is pinned at its present finding count rather
//! than excused. The pin ratchets: more findings than the pin fails, fewer is
//! reported so the pin can be lowered. No pin makes a failing gate legal and no
//! owner sentence buys it time, so progress lands in the code the gate judges
//! instead of in this file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use serde::Deserialize;

use crate::gate::{self, GateCtx, GateError, Report};
use crate::subcommands::{self, SUBSETS};

/// The name the dispatcher answers to with this runner.
///
/// The dispatcher, the generated help and the workflow-reference check each
/// needed to know the one accepted name that is not a gate. Three literals meant
/// the check that every subcommand a workflow names is dispatchable read only
/// the gate half of the table and reported the runner itself as unregistered.
pub const RUNNER: &str = "gates";

/// Pinned finding count for one gate.
///
/// `deny_unknown_fields` is load-bearing. This file used to carry `status` and
/// `owner` per row, which together let a failing gate stay legal indefinitely
/// behind a prose excuse; three gates sat red that way for a fortnight while
/// the sweep reported that every gate held its baseline. A row that still
/// carries either field, or the `output_lines` this file pinned before findings
/// were countable, now fails to load instead of being ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    name: String,
    findings: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Every `-- <name>` invocation appearing in an in-repo workflow.
///
/// Only lines that mention `xtask` are read, and a token beginning with `-` is
/// not a subcommand, so `cargo test -- --nocapture` is not mistaken for one.
/// The old scan kept only names beginning with `vyre`, which made the check
/// near-vacuous: a workflow could invoke any misspelled gate and pass.
fn workflow_invocations(root: &Path) -> (Vec<String>, Vec<String>) {
    let dir = root.join(".github/workflows");
    let mut invoked = Vec::new();
    let mut subsets = Vec::new();
    let Ok(entries) = fs::read_dir(&dir) else {
        return (invoked, subsets);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_none_or(|extension| extension != "yml" && extension != "yaml")
        {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            if !line.contains("xtask") {
                continue;
            }
            let mut rest = line;
            while let Some(at) = rest.find("-- ") {
                rest = &rest[at + 3..];
                let name = token(rest);
                if !name.is_empty() && !name.starts_with('-') {
                    invoked.push(name);
                }
            }
            let mut rest = line;
            while let Some(at) = rest.find("--subset ") {
                rest = &rest[at + 9..];
                let name = token(rest);
                if !name.is_empty() {
                    subsets.push(name);
                }
            }
        }
    }
    for list in [&mut invoked, &mut subsets] {
        list.sort();
        list.dedup();
    }
    (invoked, subsets)
}

/// The leading subcommand-shaped token of `text`.
fn token(text: &str) -> String {
    text.chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
        .collect()
}

/// Every disagreement between the registry and the baseline file.
///
/// Both directions are failures. A gate with no row would run unpinned, so a
/// new finding in it would pass; a row with no gate is a pin nobody enforces,
/// which is what a retired gate leaves behind.
fn baseline_failures(gate_names: &[&str], baselines: &[Baseline]) -> Vec<String> {
    let mut failures = Vec::new();
    for name in gate_names {
        if !baselines.iter().any(|pin| pin.name == *name) {
            failures.push(format!(
                "gate `{name}` has no row in xtask/gate-baselines.toml; add one with its present finding count"
            ));
        }
    }
    for pin in baselines {
        if !gate_names.iter().any(|name| *name == pin.name) {
            failures.push(format!(
                "xtask/gate-baselines.toml pins `{}`, which is not a registered gate; delete the row or register the gate",
                pin.name
            ));
        }
    }
    failures
}

/// Every disagreement between the registry, the subsets and the workflows.
fn workflow_failures(gate_names: &[&str], invoked: &[String], subsets: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    for subset in SUBSETS {
        for name in subset.gates {
            if !gate_names.contains(name) {
                failures.push(format!(
                    "subset `{}` names `{name}`, which is not a registered gate",
                    subset.name
                ));
            }
        }
        if !subsets.iter().any(|named| named == subset.name) {
            failures.push(format!(
                "subset `{}` is not run by any workflow in .github/workflows; invoke `xtask gates --subset {}` or delete the subset",
                subset.name, subset.name
            ));
        }
    }
    if !invoked.iter().any(|name| name == "gates") {
        failures
            .push("no workflow runs `xtask gates`, so no workflow runs the registry".to_string());
    }
    for name in subsets {
        if !SUBSETS.iter().any(|subset| subset.name == name) {
            failures.push(format!(
                "a workflow runs `xtask gates --subset {name}`, which is not a registered subset"
            ));
        }
    }
    for name in invoked {
        if name != "gates" && !gate_names.contains(&name.as_str()) {
            failures.push(format!(
                "a workflow invokes `xtask {name}`, which is not a registered gate"
            ));
        }
    }
    let mut reachable: Vec<&str> = invoked.iter().map(String::as_str).collect();
    for subset in SUBSETS {
        if subsets.iter().any(|named| named == subset.name) {
            reachable.extend(subset.gates.iter().copied());
        }
    }
    for name in gate_names {
        if !reachable.contains(name) {
            failures.push(format!(
                "no workflow names `{name}` and no subset a workflow runs contains it, so nothing in CI selects it by name; invoke `xtask {name}` from the workflow that owns it, add it to a subset a workflow runs, or delete the gate"
            ));
        }
    }
    failures
}

/// Reject a registry, baseline and workflow set that do not agree.
///
/// This is the meta-check. It is what makes an unwired gate impossible rather
/// than merely discouraged.
fn wiring_failures(root: &Path, gate_names: &[&str], baselines: &[Baseline]) -> Vec<String> {
    let (invoked, subsets) = workflow_invocations(root);
    let mut failures = baseline_failures(gate_names, baselines);
    failures.extend(workflow_failures(gate_names, &invoked, &subsets));
    failures
}

fn render_baseline(rows: &[(&str, usize)]) -> String {
    let mut text = String::from(
        "# Pinned finding count of every registered gate, written by\n\
         # `xtask gates --write-baseline`.\n\
         #\n\
         # One row per registered gate, and the sweep fails on a gate with no row\n\
         # or a row with no gate. More findings than the pin fails; fewer is\n\
         # reported so the pin can be lowered here, and the pin only ever moves\n\
         # down. A gate that cannot run at all fails outright: no row makes a\n\
         # failing gate legal, and the writer refuses to record one.\n",
    );
    for (name, findings) in rows {
        text.push_str("\n[[gate]]\n");
        text.push_str(&format!("name = \"{name}\"\n"));
        text.push_str(&format!("findings = {findings}\n"));
    }
    text
}

/// The gates this invocation runs, and what to call the selection.
fn selection(args: &[String]) -> Result<(Vec<&'static dyn crate::gate::Gate>, String), GateError> {
    let registry = subcommands::registry();
    let Some(at) = args.iter().position(|argument| argument == "--subset") else {
        return Ok((registry, "registry".to_string()));
    };
    let Some(name) = args.get(at + 1) else {
        return Err(GateError::new(
            "`--subset` was passed without a name",
            "name one of the registered subsets, or drop the flag to run the whole registry",
        ));
    };
    let Some(subset) = subcommands::subset(name) else {
        return Err(GateError::new(
            format!("`{name}` is not a registered subset"),
            format!(
                "pass one of: {}",
                SUBSETS
                    .iter()
                    .map(|subset| subset.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    };
    let selected = registry
        .into_iter()
        .filter(|gate| subset.gates.contains(&gate.name()))
        .collect();
    Ok((selected, format!("subset `{name}`")))
}

/// Run one gate and render what it reported.
fn execute(gate: &dyn crate::gate::Gate, root: &Path) -> Result<Report, GateError> {
    gate.run(&GateCtx::new(root.to_path_buf(), Vec::new()))
}

/// Run the gate sweep.
pub fn run(args: &[String]) {
    let root = crate::checkout::checkout_root();

    if args.iter().any(|argument| argument == "--list") {
        for gate in subcommands::registry() {
            println!("{}", gate.name());
        }
        for subset in SUBSETS {
            println!("subset {} {}", subset.name, subset.gates.join(" "));
        }
        return;
    }

    let (selected, what) = selection(args).unwrap_or_else(|error| {
        eprintln!("{error}");
        process::exit(2);
    });

    if args.iter().any(|argument| argument == "--write-baseline") {
        let mut rows = Vec::new();
        let mut failing = Vec::new();
        for gate in &selected {
            match execute(*gate, &root) {
                Ok(report) => {
                    println!("{}: {} finding(s)", gate.name(), report.count());
                    rows.push((gate.name(), report.count()));
                }
                Err(error) => {
                    println!("{}: could not run: {error}", gate.name());
                    failing.push(gate.name());
                }
            }
        }
        if !failing.is_empty() {
            eprintln!(
                "Fix: {} gate(s) could not run and a baseline may not record a failure: {}. Fix what they report, then write the baseline.",
                failing.len(),
                failing.join(", ")
            );
            process::exit(1);
        }
        if args.iter().any(|argument| argument == "--subset") {
            eprintln!(
                "Fix: a baseline covers the whole registry, so it cannot be written from a subset. Drop `--subset`."
            );
            process::exit(2);
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
    let registry = subcommands::registry();
    let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
    let mut failures = wiring_failures(&root, &gate_names, &baselines);

    for gate in &selected {
        let name = gate.name();
        let Some(pin) = baselines.iter().find(|pin| pin.name == name) else {
            // The wiring check already reported this, and running a gate with no
            // pin would report a number nothing holds it to.
            continue;
        };
        match execute(*gate, &root) {
            Err(error) => failures.push(format!("gate `{name}` could not run: {error}")),
            Ok(report) => {
                print!("{}", gate::render(name, &report));
                let found = report.count();
                if found > pin.findings {
                    failures.push(format!(
                        "gate `{name}` reported {found} finding(s) against a pinned {}; fix the new finding",
                        pin.findings
                    ));
                } else if found < pin.findings {
                    println!(
                        "{name}: {found} finding(s), improved from {}; lower the pin in xtask/gate-baselines.toml",
                        pin.findings
                    );
                }
            }
        }
    }

    if !failures.is_empty() {
        eprintln!("gates: {} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        process::exit(1);
    }
    println!(
        "gates: {} gate(s) in the {what} hold their baseline",
        selected.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(name: &str, findings: usize) -> Baseline {
        Baseline {
            name: name.to_string(),
            findings,
        }
    }

    /// WHY: this is the defect the whole registry exists to close. A gate
    /// registered with no baseline row runs unpinned, so a new finding in it
    /// passes; both injections have to go red, and the clean case has to be
    /// silent or the failure means nothing.
    #[test]
    fn a_registry_and_a_baseline_that_disagree_both_fail() {
        let names = ["dep-drift", "op-names"];
        assert_eq!(
            baseline_failures(&names, &[pin("dep-drift", 0), pin("op-names", 3)]),
            Vec::<String>::new()
        );

        let missing_row = baseline_failures(&names, &[pin("dep-drift", 0)]);
        assert_eq!(missing_row.len(), 1);
        assert!(missing_row[0].contains("`op-names` has no row"));

        let extra_row = baseline_failures(
            &names,
            &[pin("dep-drift", 0), pin("op-names", 3), pin("retired", 9)],
        );
        assert_eq!(extra_row.len(), 1);
        assert!(extra_row[0].contains("pins `retired`"));
    }

    /// WHY: the baseline row shape is the exemption surface. `status` and
    /// `owner` are what kept three gates red for a fortnight, and `output_lines`
    /// is the pin that counted output instead of findings, so a file carrying
    /// any of them must fail to load rather than be read with the field ignored.
    #[test]
    fn a_row_carrying_a_retired_field_fails_to_load() {
        let good: BaselineFile =
            toml::from_str("[[gate]]\nname = \"dep-drift\"\nfindings = 0\n").expect("loads");
        assert_eq!(good.gate.len(), 1);
        for row in [
            "[[gate]]\nname = \"dep-drift\"\nfindings = 0\nstatus = \"red\"\n",
            "[[gate]]\nname = \"dep-drift\"\nfindings = 0\nowner = \"someone\"\n",
            "[[gate]]\nname = \"dep-drift\"\noutput_lines = 32\n",
        ] {
            assert!(
                toml::from_str::<BaselineFile>(row).is_err(),
                "a row carrying a retired field must not load: {row}"
            );
        }
    }

    /// WHY: a workflow that invokes a name nobody registered runs nothing under
    /// a name that reads as coverage, and a subset nobody runs is a set of gates
    /// that only the whole-registry sweep reaches. The old scan kept only names
    /// starting with `vyre`, so it could not see either. The third direction is
    /// gate to workflow: `file-size` was red on fourteen files while no workflow
    /// named it, so its redness reached nobody who could act on it. A registered
    /// gate no workflow selects by name, directly or through a subset, is
    /// decoration until someone wires it.
    #[test]
    fn workflow_and_registry_disagreement_fails_in_both_directions() {
        // The registry under test must hold every subset member, because a
        // subset naming a gate nobody registered is itself a failure and would
        // otherwise drown the direction this test exercises.
        let mut names: Vec<&str> = SUBSETS
            .iter()
            .flat_map(|subset| subset.gates.iter().copied())
            .collect();
        names.push("dep-drift");
        names.sort_unstable();
        names.dedup();
        let every_subset: Vec<String> = SUBSETS
            .iter()
            .map(|subset| subset.name.to_string())
            .collect();
        let clean = workflow_failures(
            &names,
            &["gates".to_string(), "dep-drift".to_string()],
            &every_subset,
        );
        assert_eq!(clean, Vec::<String>::new());

        let unregistered = workflow_failures(
            &names,
            &["gates".to_string(), "dep-drfit".to_string()],
            &every_subset,
        );
        assert_eq!(unregistered.len(), 1);
        assert!(unregistered[0].contains("`xtask dep-drfit`"));

        let unswept = workflow_failures(&names, &["dep-drift".to_string()], &every_subset);
        assert!(unswept
            .iter()
            .any(|failure| failure.contains("no workflow runs `xtask gates`")));

        let unrun_subset = workflow_failures(&names, &["gates".to_string()], &[]);
        assert_eq!(unrun_subset.len(), SUBSETS.len() + names.len());
        assert_eq!(
            unrun_subset
                .iter()
                .filter(|failure| failure.contains("no workflow names"))
                .count(),
            names.len(),
            "a subset nobody runs leaves every gate it holds named by nobody: {unrun_subset:?}"
        );

        // A gate no subset holds and no workflow names. `file-size` used to
        // stand here and was wired into `source-rules` since, which made this
        // direction assert nothing: the name has to be one the subsets do not
        // carry for the check to be about wiring at all.
        let with_unnamed: Vec<&str> = [names.clone(), vec!["unwired-gate"]].concat();
        let unnamed = workflow_failures(
            &with_unnamed,
            &["gates".to_string(), "dep-drift".to_string()],
            &every_subset,
        );
        assert_eq!(unnamed.len(), 1);
        assert!(
            unnamed[0].contains("no workflow names `unwired-gate`"),
            "got {unnamed:?}"
        );

        let named_directly = workflow_failures(
            &with_unnamed,
            &[
                "gates".to_string(),
                "dep-drift".to_string(),
                "unwired-gate".to_string(),
            ],
            &every_subset,
        );
        assert_eq!(named_directly, Vec::<String>::new());

        let unknown_subset = workflow_failures(
            &names,
            &["gates".to_string()],
            &[every_subset.clone(), vec!["not-a-subset".to_string()]].concat(),
        );
        assert_eq!(unknown_subset.len(), 1);
        assert!(unknown_subset[0].contains("not-a-subset"));
    }

    /// WHY: the token reader decides what counts as an invocation, so a flag
    /// must never be read as a subcommand name.
    #[test]
    fn a_flag_is_not_a_subcommand_name() {
        assert_eq!(token("dep-drift --strict"), "dep-drift");
        assert_eq!(token("--nocapture"), "--nocapture");
        assert_eq!(token(""), "");
    }

    /// WHY: the sweep runs what the registry holds at run time, so a gate that
    /// exists is a gate that is swept. A hardcoded count here would go stale in
    /// silence, which is the same failure as having no sweep.
    #[test]
    fn the_sweep_enumerates_the_registry() {
        assert_eq!(
            selection(&[]).expect("the whole registry").0.len(),
            subcommands::registry().len()
        );
        let (selected, what) =
            selection(&["--subset".to_string(), "cat-a".to_string()]).expect("a registered subset");
        assert_eq!(what, "subset `cat-a`");
        assert!(!selected.is_empty());
        assert!(selected.len() < subcommands::registry().len());
        assert!(selection(&["--subset".to_string()]).is_err());
        assert!(selection(&["--subset".to_string(), "nope".to_string()]).is_err());
    }
}
