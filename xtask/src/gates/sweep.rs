//! The gate runner.
//!
//! 41 subcommands were registered and 9 were ever invoked. The other 32 gates
//! compiled, passed review, and judged nothing, so every rule they encoded was
//! decorative. Nothing failed when that happened.
//!
//! The sweep enumerates the registry at run time, so a gate that exists is a
//! gate that is swept and neither list is maintained by hand. Which subsets hold
//! a gate and which workflows run it are declared once in
//! `xtask/ci-registry.toml`, and the `ci-registry` gate holds that declaration
//! to the registry, the subsets and the workflow steps in both directions.
//! Registration is not wiring: a gate no workflow selects by name is a failure
//! there, not a pinnable finding here.
//!
//! A gate that reports findings is pinned in `xtask/gate-baselines.toml` at its
//! present finding count rather than excused. The pin ratchets: more findings
//! than the pin fails, fewer is reported so the pin can be lowered. A selected
//! gate with no row does not run, because a number nothing holds it to is not a
//! result. No pin makes a failing gate legal and no owner sentence buys it time,
//! so progress lands in the code the gate judges instead of in that file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use serde::Deserialize;

use crate::gate::{self, GateCtx, GateError, Report};
use crate::subcommands::{self, SUBSETS};

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
                "gate `{name}` has no row in xtask/gate-baselines.toml, so it does not run; add one with its present finding count"
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

/// Render the baseline file from measured finding counts.
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

/// The name the dispatcher answers to with this runner.
///
/// The dispatcher, the generated help and the workflow-reference check each
/// needed to know the one accepted name that is not a gate. Three literals meant
/// the check that every subcommand a workflow names is dispatchable read only
/// the gate half of the table and reported the runner itself as unregistered.
pub const RUNNER: &str = "gates";

/// Reject a registry whose gates cannot report a finding.
///
/// Agreement between the declaration, the registry, the subsets and the
/// workflows is the `ci-registry` gate's, so it is checked once there. What
/// stays here is the property the runner itself depends on: a gate that can
/// only ever return a clean report pins at zero and holds forever.
fn wiring_failures(root: &Path, gate_names: &[&str]) -> Vec<String> {
    match crate::gates::finding_capability::failures(root, gate_names) {
        Ok(found) => found,
        Err(error) => vec![format!(
            "the finding-capability check could not read the gate sources: {}. Fix: {}",
            error.message, error.fix
        )],
    }
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

/// Write the pins from measured findings.
///
/// The pin only ever moves down. A writer that recorded whatever it measured
/// would turn every red gate green by running it, which is the exemption the
/// `status` and `owner` fields used to grant in prose, so a gate measuring above
/// its recorded pin fails here and nothing is written. The wiring declaration is
/// a different fact with a different owner: `xtask ci-registry --write`.
fn write_declaration(root: &Path, args: &[String], selected: &[&'static dyn crate::gate::Gate]) {
    if args.iter().any(|argument| argument == "--subset") {
        eprintln!(
            "Fix: the pins cover the whole registry, so they cannot be written from a subset. Drop `--subset`."
        );
        process::exit(2);
    }
    let recorded = load_baselines(root);
    let mut rows = Vec::new();
    let mut failing = Vec::new();
    let mut raised = Vec::new();
    for gate in selected {
        let name = gate.name();
        match execute(*gate, root) {
            Ok(report) => {
                let found = report.count();
                println!("{name}: {found} finding(s)");
                match recorded.iter().find(|pin| pin.name == name) {
                    Some(pin) if found > pin.findings => {
                        raised.push(format!("{name} measured {found} against a pinned {}", pin.findings));
                    }
                    Some(pin) => rows.push((name, found.min(pin.findings))),
                    None => rows.push((name, found)),
                }
            }
            Err(error) => {
                println!("{name}: could not run: {error}");
                failing.push(name);
            }
        }
    }
    if !failing.is_empty() {
        eprintln!(
            "Fix: {} gate(s) could not run and a pin may not record a failure: {}. Fix what they report, then write it.",
            failing.len(),
            failing.join(", ")
        );
        process::exit(1);
    }
    if !raised.is_empty() {
        eprintln!(
            "Fix: {} gate(s) report more than they are pinned at, and a pin never rises: {}. Fix the findings.",
            raised.len(),
            raised.join("; ")
        );
        process::exit(1);
    }
    let path = baseline_path(root);
    fs::write(&path, render_baseline(&rows)).unwrap_or_else(|error| {
        eprintln!("Fix: cannot write {}: {error}", path.display());
        process::exit(1);
    });
    println!("wrote {}", path.display());
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
        write_declaration(&root, args, &selected);
        return;
    }

    let baselines = load_baselines(&root);
    let registry = subcommands::registry();
    let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
    let mut failures = wiring_failures(&root, &gate_names);
    failures.extend(baseline_failures(&gate_names, &baselines));

    for gate in &selected {
        let name = gate.name();
        let Some(pin) = baselines.iter().find(|pin| pin.name == name) else {
            // Running a gate with no pin would report a number nothing holds it
            // to, which reads as a result and is not one. The gap is already a
            // failure, reported once by the registry comparison above.
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

    /// WHY: the pin and the wiring are two facts in two files, and a writer that
    /// mixes them recreates the exemption surface the pin file closed. A
    /// `findings` key in the wiring declaration must fail to load.
    #[test]
    fn the_wiring_declaration_does_not_carry_a_pin() {
        assert!(toml::from_str::<crate::gates::ci_registry::Registry>(
            "schema_version = 1\n[[gate]]\nname = \"dep-drift\"\nsubsets = []\nworkflows = []\n"
        )
        .is_ok());
        assert!(toml::from_str::<crate::gates::ci_registry::Registry>(
            "schema_version = 1\n[[gate]]\nname = \"dep-drift\"\nfindings = 0\n"
        )
        .is_err());
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

    /// WHY: every gate the runner sweeps has to be able to report a finding. A
    /// gate that can only return a clean report pins at zero and holds forever,
    /// which reads as coverage and is none.
    #[test]
    fn every_registered_gate_can_report_a_finding() {
        let root = crate::checkout::checkout_root();
        let registry = subcommands::registry();
        let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
        let failures = wiring_failures(&root, &gate_names);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
