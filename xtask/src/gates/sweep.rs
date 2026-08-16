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

use crate::gates::gate_canon::{baseline_path, load_baselines, Baseline};
use std::fs;
use std::path::Path;
use std::process;

use crate::gate::{self, GateCtx, GateError, Report};
use crate::subcommands;

/// The name the dispatcher answers to with this runner.
///
/// The dispatcher, the generated help and the workflow-reference check each
/// needed to know the one accepted name that is not a gate. Three literals meant
/// the check that every subcommand a workflow names is dispatchable read only
/// the gate half of the table and reported the runner itself as unregistered.
pub const RUNNER: &str = "gates";

/// Read the pinned rows, or exit naming what could not be read.
///
/// The rows and every rule about them belong to `gate_canon`, which is the gate
/// a caller can ask for by name. The sweep needs them before it runs anything,
/// because pairing a gate with its pin is what the sweep does, so it reads them
/// through that owner rather than parsing the file a second time here.
fn baselines_or_exit(root: &Path) -> Vec<Baseline> {
    load_baselines(root).unwrap_or_else(|error| {
        eprintln!("{error}");
        process::exit(1);
    })
}

/// What the in-repo workflows name: xtask subcommands, subsets, and scripts.
struct WorkflowNames {
    /// Every `xtask <name>` subcommand a workflow invokes.
    invoked: Vec<String>,
    /// Every `xtask gates --subset <name>` a workflow runs.
    subsets: Vec<String>,
    /// Every `scripts/<path>` a workflow command names, with where it said so.
    scripts: Vec<(String, usize, String)>,
}

/// Every `-- <name>` invocation appearing in an in-repo workflow.
///
/// Only lines that mention `xtask` are read for subcommands, and a token
/// beginning with `-` is not a subcommand, so `./cargo_full test -- --nocapture` is not
/// mistaken for one. The old scan kept only names beginning with `vyre`, which
/// made the check near-vacuous: a workflow could invoke any misspelled gate and
/// pass.
///
/// Script references are read from every line, because a workflow that invokes a
/// script the checkout no longer carries fails at run time under a step name
/// that still reads as coverage.
fn workflow_names(root: &Path) -> WorkflowNames {
    let mut invoked = Vec::new();
    let mut subsets = Vec::new();
    let mut scripts = Vec::new();
    let dir = root.join(".github/workflows");
    let Ok(entries) = fs::read_dir(&dir) else {
        return WorkflowNames {
            invoked,
            subsets,
            scripts,
        };
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
        let file = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for (index, line) in text.lines().enumerate() {
            if let Some(script) = referenced_script(line) {
                scripts.push((file.clone(), index + 1, script.to_string()));
            }
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
    WorkflowNames {
        invoked,
        subsets,
        scripts,
    }
}

/// The leading subcommand-shaped token of `text`.
fn token(text: &str) -> String {
    text.chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
        .collect()
}

/// The script a workflow line invokes, relative to `scripts/`, or `None` when
/// the line invokes nothing.
///
/// A YAML comment is documentation, not a reference: prose that ends a sentence
/// with `source_scan.sh.` names no file, and reporting it as a missing script
/// makes the check fail on its own explanatory text.
fn referenced_script(line: &str) -> Option<&str> {
    let command = strip_yaml_comment(line.trim());
    let index = command.find("scripts/")?;
    let rest = &command[index + "scripts/".len()..];
    let name = rest.split_whitespace().next().unwrap_or(rest);
    let name = name.trim_end_matches(['"', '\'', ';', ')']);
    (!name.is_empty()).then_some(name)
}

/// Everything before a trailing YAML comment. `#` opens one at the start of a
/// line or after whitespace.
fn strip_yaml_comment(line: &str) -> &str {
    if line.starts_with('#') {
        return "";
    }
    match line.find(" #") {
        Some(index) => &line[..index],
        None => line,
    }
}

/// Every workflow reference to a script the checkout does not carry.
///
/// A glob is rejected outright. It was accepted while `scripts/check_*.sh`
/// carried assertions a workflow ran as a set; every one of those is a
/// registered gate now, so a workflow step naming a set of scripts is a step
/// that reaches whatever a future checkout happens to leave in the directory.
fn script_failures(root: &Path, scripts: &[(String, usize, String)]) -> Vec<String> {
    let directory = root.join("scripts");
    let mut failures = Vec::new();
    for (file, line, name) in scripts {
        if name.contains('*') {
            failures.push(format!(
                "{file}:{line} names `scripts/{name}`; a workflow step names one script or one gate, never a glob"
            ));
            continue;
        }
        if !directory.join(name).exists() {
            failures.push(format!(
                "{file}:{line} invokes `scripts/{name}`, which the checkout does not carry; point the step at what owns the rule now, or delete the step"
            ));
        }
    }
    failures
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
        if pin.findings != 0 {
            failures.push(format!(
                "xtask/gate-baselines.toml pins `{}` at {}; every gate is hard and every pin must remain zero",
                pin.name, pin.findings
            ));
        }
    }
    failures
}

/// Every disagreement between executable gates and their static contracts.
fn metadata_failures(
    root: &Path,
    gate_names: &[&str],
    descriptors: &[crate::gate::GateDescriptor],
) -> Vec<String> {
    let mut failures = Vec::new();
    for name in gate_names {
        let Some(desc) = descriptors.iter().find(|d| d.name == *name) else {
            failures.push(format!(
                "gate `{name}` has no descriptor row; classify its package, area, subject universe, artifacts, prerequisites, and mutation proof"
            ));
            continue;
        };
        for failure in desc.failures() {
            failures.push(format!("gate `{name}` descriptor {failure}"));
        }
    }
    for desc in descriptors {
        if !gate_names.contains(&desc.name) {
            failures.push(format!(
                "gate descriptor names `{}`, which has no executable gate; delete the stale row or restore the implementation",
                desc.name
            ));
        }
    }
    let mut artifact_owners = std::collections::BTreeMap::new();
    for desc in descriptors {
        for artifact in desc.artifacts {
            if let Some(other) = artifact_owners.insert(*artifact, desc.name) {
                failures.push(format!(
                    "artifact `{artifact}` is claimed by both `{other}` and `{}`",
                    desc.name
                ));
            }
        }
    }
    failures.extend(crate::gate_metadata::validate_all_descriptors(
        root,
        descriptors,
    ));
    failures
}
/// Every disagreement between the registry, the subsets and the workflows.
fn workflow_failures(gate_names: &[&str], invoked: &[String], subsets: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    let registered_subsets = subcommands::subsets();
    for subset in &registered_subsets {
        for name in &subset.gates {
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
        if !registered_subsets.iter().any(|subset| subset.name == name) {
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
    for subset in &registered_subsets {
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
/// than merely discouraged, and what makes a workflow step invoking a deleted
/// script impossible to leave behind.
fn wiring_failures(root: &Path, gate_names: &[&str], baselines: &[Baseline]) -> Vec<String> {
    let names = workflow_names(root);
    let mut failures = baseline_failures(gate_names, baselines);
    failures.extend(metadata_failures(
        root,
        gate_names,
        crate::gate_metadata::GATE_METADATA,
    ));
    failures.extend(workflow_failures(
        gate_names,
        &names.invoked,
        &names.subsets,
    ));
    failures.extend(script_failures(root, &names.scripts));
    match crate::gates::finding_capability::failures(root, gate_names) {
        Ok(found) => failures.extend(found),
        Err(error) => failures.push(format!(
            "the finding-capability check could not read the gate sources: {}. Fix: {}",
            error.message, error.fix
        )),
    }
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
fn selection(args: &[String]) -> Result<(Vec<crate::gate::RegisteredGate>, String), GateError> {
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
                subcommands::subsets()
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
fn execute(gate: &crate::gate::RegisteredGate, root: &Path) -> Result<Report, GateError> {
    gate.run(&GateCtx::new(root.to_path_buf(), Vec::new()))
}

/// Execute a selection of gates in sweep comparison mode with exactly one pre-capture and one post-verification.
pub fn execute_sweep_selection(
    root: &Path,
    selected: &[crate::gate::RegisteredGate],
    baselines: &[Baseline],
) -> Vec<String> {
    let mut failures = Vec::new();
    let sweep_snapshot = crate::artifact_gate::WorkspaceSnapshot::capture(root);

    for gate in selected {
        let name = gate.name();
        let Some(pin) = baselines.iter().find(|pin| pin.name == name) else {
            continue;
        };
        let descriptor = crate::gate_metadata::descriptor(name);
        match execute(gate, root) {
            Err(error) => failures.push(format!("gate `{name}` could not run: {error}")),
            Ok(report) => {
                if let Some(descriptor) = descriptor {
                    for failure in report.contract_failures(descriptor) {
                        failures.push(format!("gate `{name}` {failure}"));
                    }
                } else {
                    for failure in report.coverage_failures() {
                        failures.push(format!("gate `{name}` {failure}"));
                    }
                }
                let found = report.count();
                if found > pin.findings {
                    failures.push(format!(
                        "gate `{name}` reported {found} finding(s) against a pinned {}; fix the new finding",
                        pin.findings
                    ));
                }
            }
        }
    }

    let sweep_mutations = sweep_snapshot.detect_mutations(root, "sweep", &[], false);
    for mutation in sweep_mutations {
        failures.push(mutation);
    }
    failures
}

/// Run the gate sweep.
pub fn run(args: &[String]) {
    let root = crate::checkout::checkout_root();

    if args.iter().any(|argument| argument == "--list") {
        for gate in subcommands::registry() {
            println!("{}", gate.name());
        }
        for subset in subcommands::subsets() {
            println!("subset {} {}", subset.name, subset.gates.join(" "));
        }
        return;
    }

    let (selected, what) = selection(args).unwrap_or_else(|error| {
        eprintln!("{error}");
        process::exit(2);
    });

    if args.iter().any(|argument| argument == "--write-baseline") {
        let recorded = baselines_or_exit(&root);
        let mut rows = Vec::new();
        let mut failing = Vec::new();
        let mut red = Vec::new();
        for gate in &selected {
            match execute(gate, &root) {
                Ok(report) => {
                    println!("{}: {} finding(s)", gate.name(), report.count());
                    rows.push((gate.name(), report.count()));
                    if report.count() != 0 {
                        red.push(gate.name());
                    }
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
        if !red.is_empty() {
            eprintln!(
                "Fix: every gate is hard and a baseline may record only zero findings; fix: {}",
                red.join(", ")
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

    let baselines = baselines_or_exit(&root);
    let registry = subcommands::registry();
    let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
    let mut failures = wiring_failures(&root, &gate_names, &baselines);
    let sweep_snapshot = crate::artifact_gate::WorkspaceSnapshot::capture(&root);

    for gate in &selected {
        let name = gate.name();
        let Some(pin) = baselines.iter().find(|pin| pin.name == name) else {
            // The wiring check already reported this, and running a gate with no
            // pin would report a number nothing holds it to.
            continue;
        };
        let descriptor = crate::gate_metadata::descriptor(name);
        match execute(gate, &root) {
            Err(error) => failures.push(format!("gate `{name}` could not run: {error}")),
            Ok(report) => {
                print!("{}", gate::render(name, &report));
                if let Some(descriptor) = descriptor {
                    for failure in report.contract_failures(descriptor) {
                        failures.push(format!("gate `{name}` {failure}"));
                    }
                } else {
                    for failure in report.coverage_failures() {
                        failures.push(format!("gate `{name}` {failure}"));
                    }
                }
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

    let sweep_mutations = sweep_snapshot.detect_mutations(&root, "sweep", &[], false);
    for mutation in sweep_mutations {
        failures.push(mutation);
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
    use crate::gates::gate_canon::BaselineFile;

    fn pin(name: &str, findings: usize) -> Baseline {
        Baseline {
            name: name.to_string(),
            findings,
        }
    }

    /// WHY: Section 182.5.4 requires rejecting any write outside the declaring gate's owned set.
    #[test]
    fn unowned_workspace_write_is_detected_and_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path();
        let file_a = root.join("file_a.txt");
        let unowned = root.join("unowned.txt");
        std::fs::write(&file_a, "initial").expect("write file_a");

        let snapshot = crate::artifact_gate::WorkspaceSnapshot::capture(root);

        // Mutate unowned file outside declared set
        std::fs::write(&unowned, "unowned content").expect("write unowned");

        let declared_artifacts = &["file_a.txt"];
        let violations = snapshot.detect_mutations(root, "test-gate", declared_artifacts, false);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("created workspace file `unowned.txt`"));
        assert!(violations[0].contains("Section 182.5.6"));
    }

    /// WHY: Section 182.7.2 requires proving that full sweep and aggregate lego-audit invocation
    /// each execute every discrete LEGO law gate exactly once without duplicate/rerun execution.
    #[test]
    fn full_sweep_and_aggregate_each_execute_laws_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let execution_counts: std::collections::HashMap<&str, Arc<AtomicUsize>> =
            subcommands::LEGO_LAW_GATES
                .iter()
                .map(|name| (*name, Arc::new(AtomicUsize::new(0))))
                .collect();

        // 1. Full sweep simulation: runs registry() where each of the 123 gates is present once
        let full_registry = subcommands::registry();
        for gate in &full_registry {
            if let Some(counter) = execution_counts.get(gate.name()) {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        }
        for (name, counter) in &execution_counts {
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "full sweep must execute LEGO law gate `{name}` exactly once"
            );
        }

        // 2. Aggregate lego-audit simulation: runs the subset 'lego-audit'
        for counter in execution_counts.values() {
            counter.store(0, Ordering::SeqCst);
        }
        let subset = subcommands::subset("lego-audit").expect("lego-audit subset exists");
        assert_eq!(subset.gates.len(), 11);
        for gate_name in &subset.gates {
            let gate = subcommands::find(gate_name).expect("gate exists in registry");
            if let Some(counter) = execution_counts.get(gate.name()) {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        }
        for (name, counter) in &execution_counts {
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "aggregate lego-audit invocation must execute LEGO law gate `{name}` exactly once"
            );
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
            baseline_failures(&names, &[pin("dep-drift", 0), pin("op-names", 0)]),
            Vec::<String>::new()
        );

        let missing_row = baseline_failures(&names, &[pin("dep-drift", 0)]);
        assert_eq!(missing_row.len(), 1);
        assert!(missing_row[0].contains("`op-names` has no row"));

        let extra_row = baseline_failures(
            &names,
            &[pin("dep-drift", 0), pin("op-names", 0), pin("retired", 0)],
        );
        assert_eq!(extra_row.len(), 1);
        assert!(extra_row[0].contains("pins `retired`"));

        let tolerated = baseline_failures(&names, &[pin("dep-drift", 0), pin("op-names", 1)]);
        assert_eq!(tolerated.len(), 1);
        assert!(tolerated[0].contains("every pin must remain zero"));

        let valid_descriptor = crate::gate::GateDescriptor {
            name: "dep-drift",
            help:
                "Reject a manifest that pins a workspace-managed dependency to a different version",
            package: "xtask",
            areas: &["prepublish"],
            subject: "workspace manifests",
            artifacts: &[],
            prerequisites: &[],
            proof: "crate::gates::dep_drift::tests::dep_drift_detects_mismatched_dependency_versions_and_ignores_workspace_inheritance",
        };
        let root = crate::checkout::checkout_root();
        assert!(metadata_failures(&root, &["dep-drift"], &[valid_descriptor]).is_empty());
        assert!(
            metadata_failures(&root, &["dep-drift", "missing"], &[valid_descriptor])[0]
                .contains("`missing` has no descriptor")
        );
        assert!(metadata_failures(&root, &[], &[valid_descriptor])[0]
            .contains("has no executable gate"));
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
        let subsets = subcommands::subsets();
        // The registry under test must hold every subset member, because a
        // subset naming a gate nobody registered is itself a failure and would
        // otherwise drown the direction this test exercises.
        let mut names: Vec<&str> = subsets
            .iter()
            .flat_map(|subset| subset.gates.iter().copied())
            .collect();
        names.push("dep-drift");
        names.sort_unstable();
        names.dedup();
        let every_subset: Vec<String> = subsets
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
        assert_eq!(unrun_subset.len(), subsets.len() + names.len());
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
        let subset = subcommands::subsets()
            .into_iter()
            .next()
            .expect("the descriptor registry has at least one nonempty area");
        let (selected, what) = selection(&["--subset".to_string(), subset.name.to_string()])
            .expect("a descriptor-derived subset");
        assert_eq!(what, format!("subset `{}`", subset.name));
        assert_eq!(selected.len(), subset.gates.len());
        assert!(!selected.is_empty());
        assert!(selected.len() < subcommands::registry().len());
        assert!(selection(&["--subset".to_string()]).is_err());
        assert!(selection(&["--subset".to_string(), "nope".to_string()]).is_err());
    }

    /// WHY: a workflow reference is read out of a shell command, and the same
    /// file explains itself in YAML comments. Prose that ends a sentence with a
    /// script name invokes nothing, so reading it as a reference would make the
    /// check fail on documentation.
    #[test]
    fn a_script_reference_comes_from_a_command_not_from_prose() {
        assert_eq!(
            referenced_script("        run: bash scripts/check_feature_msrv.sh"),
            Some("check_feature_msrv.sh")
        );
        assert_eq!(
            referenced_script("        run: bash scripts/lib/cargo_runner.sh --strict"),
            Some("lib/cargo_runner.sh")
        );
        assert_eq!(
            referenced_script("        run: bash \"scripts/check_public_api.sh\";"),
            Some("check_public_api.sh")
        );
        assert_eq!(
            referenced_script("      # all on scripts/cargo_runner.sh."),
            None
        );
        assert_eq!(
            referenced_script("        run: bash scripts/gate.sh # see scripts/other.sh."),
            Some("gate.sh")
        );
        assert_eq!(referenced_script("        run: cargo test"), None);
        assert_eq!(
            referenced_script("        run: bash scripts/check_*.sh"),
            Some("check_*.sh")
        );
    }

    /// WHY: every script this campaign deletes is named by the workflow that
    /// used to run it. A step pointing at a script the checkout no longer
    /// carries fails at run time under a step name that still reads as
    /// coverage, so the wiring check has to name it before CI does.
    #[test]
    fn a_workflow_naming_a_script_the_tree_lacks_fails() {
        let root = std::env::temp_dir().join(format!("vyre-sweep-scripts-{}", std::process::id()));
        fs::create_dir_all(root.join("scripts")).expect("the fixture tree is created");
        fs::write(root.join("scripts/present.sh"), "#!/bin/sh\n").expect("the script is written");

        let present = vec![("gates.yml".to_string(), 7, "present.sh".to_string())];
        let absent = vec![("gates.yml".to_string(), 9, "retired.sh".to_string())];
        let glob = vec![("gates.yml".to_string(), 11, "check_*.sh".to_string())];
        let other_glob = vec![("gates.yml".to_string(), 13, "run_*.sh".to_string())];

        let clean = script_failures(&root, &present);
        let missing = script_failures(&root, &absent);
        let globbed = script_failures(&root, &glob);
        let other = script_failures(&root, &other_glob);

        fs::remove_dir_all(&root).expect("the fixture is removed");
        assert_eq!(clean, Vec::<String>::new());
        assert_eq!(missing.len(), 1, "got {missing:?}");
        assert!(missing[0].contains("gates.yml:9"), "got {missing:?}");
        assert!(missing[0].contains("scripts/retired.sh"), "got {missing:?}");
        assert_eq!(globbed.len(), 1, "got {globbed:?}");
        assert!(globbed[0].contains("check_*.sh"), "got {globbed:?}");
        assert_eq!(other.len(), 1, "got {other:?}");
        assert!(other[0].contains("run_*.sh"), "got {other:?}");
    }
    /// WHY: Section 182.5.6 and performance contract require O(1) workspace hashing per sweep run,
    /// rather than O(gates) rehashing the entire workspace 123 times.
    #[test]
    fn sweep_performs_exactly_one_capture_and_one_verification_regardless_of_gate_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fs::write(root.join("test.txt"), "hello").expect("write file");

        crate::artifact_gate::reset_snapshot_counters();

        let registry = subcommands::registry();
        assert!(registry.len() > 10, "registry must contain multiple gates");

        let baselines = load_baselines(&crate::checkout::checkout_root()).unwrap_or_default();

        let subset_gates: Vec<crate::gate::RegisteredGate> =
            registry.iter().take(5).copied().collect();
        let _ = execute_sweep_selection(root, &subset_gates, &baselines);

        let (captures, verifications) = crate::artifact_gate::snapshot_counter_values();
        assert_eq!(
            verifications, 1,
            "Sweep must perform exactly 1 verification regardless of gate count"
        );
        assert_eq!(
            captures, 2,
            "Sweep must perform exactly 1 pre-capture + 1 post-capture inside verification"
        );
    }
}
