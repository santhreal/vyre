//! The gate runner and the wiring meta-check.
//!
//! The registry and the workflows must agree, and disagreement is a hard
//! failure: a registered gate no workflow selects by name, directly or through
//! a subset a workflow runs, judges nothing anyone reads. `file-size` was red on
//! fourteen source files while nothing in CI named it.
//!
//! Every gate holds zero findings. There is no per-gate pin: a count above zero
//! fails the sweep, and the only way to make a red gate green is to change what
//! it judges.

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
use crate::gates::ci_registry::referenced_script;
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

/// Reject a registry and workflow set that do not agree.
///
/// This is the meta-check. It is what makes an unwired gate impossible rather
/// than merely discouraged, and what makes a workflow step invoking a deleted
/// script impossible to leave behind.
fn wiring_failures(root: &Path, gate_names: &[&str]) -> Vec<String> {
    let names = workflow_names(root);
    let mut failures = crate::gates::gate_canon::subset_failures(gate_names);
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

/// The gates this invocation runs, and what to call the selection.
fn selection(args: &[String]) -> Result<(Vec<crate::gate::RegisteredGate>, String), GateError> {
    let registry = subcommands::registry();
    let Some(at) = args
        .iter()
        .position(|argument| argument == "--subset" || argument == "--area")
    else {
        return Ok((registry, "registry".to_string()));
    };
    let flag = &args[at];
    let Some(name) = args.get(at + 1) else {
        return Err(GateError::new(
            format!("`{flag}` was passed without a name"),
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

/// Run each selected gate once, print what it reported, and return every
/// failure: a gate that could not run, a report that breaks its descriptor
/// contract, any finding, and any workspace write. The workspace is hashed once
/// before the first gate and once after the last.
pub fn sweep(root: &Path, selected: &[crate::gate::RegisteredGate]) -> Vec<String> {
    let mut failures = Vec::new();
    let snapshot = crate::artifact_gate::WorkspaceSnapshot::capture(root);
    for gate in selected {
        let name = gate.name();
        match execute(gate, root) {
            Err(error) => failures.push(format!("gate `{name}` could not run: {error}")),
            Ok(report) => {
                print!("{}", gate::render(name, &report));
                let contract = match crate::gate_metadata::descriptor(name) {
                    Some(descriptor) => report.contract_failures(descriptor),
                    None => report.coverage_failures(),
                };
                failures.extend(
                    contract
                        .into_iter()
                        .map(|failure| format!("gate `{name}` {failure}")),
                );
                if report.count() > 0 {
                    failures.push(format!(
                        "gate `{name}` reported {} finding(s); every gate holds zero, so fix what it reports",
                        report.count()
                    ));
                }
            }
        }
    }
    failures.extend(snapshot.detect_mutations(root, "sweep", &[], false));
    failures
}

/// Whether a runner command line asks for usage.
///
/// `--subset` and `--area` take the next word as their value, so a subset
/// literally named `--help` is a name and not a request.
fn help_requested_of_the_runner(args: &[String]) -> bool {
    let mut words = args.iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--subset" | "--area" => {
                words.next();
            }
            "--help" | "-h" => return true,
            _ => {}
        }
    }
    false
}

/// The word the runner does not accept, if the caller passed one.
///
/// A silently ignored flag is worse than a rejected one here: `--gate file-size`
/// looks like it ran one gate and actually ran the whole registry, so several
/// runs were read as single-gate evidence when they were full sweeps. The
/// runner takes flags only, so anything that is not a known flag or the value
/// of `--subset`/`--area` is a typo.
fn unknown_argument(args: &[String]) -> Option<String> {
    const KNOWN: &[&str] = &["--list", "--help", "-h"];
    let mut words = args.iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--subset" | "--area" => {
                words.next();
            }
            other if KNOWN.contains(&other) => {}
            other => return Some(other.to_string()),
        }
    }
    None
}

/// What the runner accepts, rendered from the registry it would run.
fn usage_text() -> String {
    let mut text = String::from(
        "usage: xtask gates [--subset <name>] [--list]\n\n\
         With no flags every registered gate runs, and any finding fails.\n\n\
         --subset <name>   run one subset; --area is accepted for it\n\
         --list            print every gate name and every subset\n\n\
         subsets:\n",
    );
    for subset in subcommands::subsets() {
        text.push_str(&format!("  {} {}\n", subset.name, subset.gates.join(" ")));
    }
    text
}

/// Run the gate sweep.
pub fn run(args: &[String]) {
    let root = crate::checkout::checkout_root();

    // Usage is an answer, not a run. `gates --help` executed the whole registry,
    // which is the check running instead of describing itself, the same defect
    // `bench-crossback --help` had. The runner takes flags, so the request is
    // honored anywhere it is not the value of the subset flag, which is how
    // `lego-audit --help` and `gates --subset lego-audit --help` reach it.
    if help_requested_of_the_runner(args) {
        print!("{}", usage_text());
        return;
    }

    if let Some(word) = unknown_argument(args) {
        eprintln!(
            "Fix: `xtask gates` does not accept `{word}`. Run `xtask gates --help` for what it \
             takes; there is no per-gate flag, `--subset <name>` is how you run less than the \
             whole registry."
        );
        process::exit(2);
    }

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

    let registry = subcommands::registry();
    let gate_names: Vec<&str> = registry.iter().map(|gate| gate.name()).collect();
    let mut failures = wiring_failures(&root, &gate_names);
    failures.extend(sweep(&root, &selected));
    if !failures.is_empty() {
        eprintln!("gates: {} failure(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        process::exit(1);
    }
    println!(
        "gates: {} gate(s) in the {what} report zero findings",
        selected.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
    ///
    /// The roster is derived from the metadata area, not restated here. A
    /// hardcoded copy plus a hardcoded count is what went stale when
    /// `lego-tier-claims` joined: the list grew to twelve and the count kept
    /// asserting eleven, so the proof failed on its own bookkeeping.
    #[test]
    fn full_sweep_and_aggregate_each_execute_laws_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let subset = subcommands::subset("lego-audit").expect("lego-audit subset exists");
        assert!(
            !subset.gates.is_empty(),
            "the lego-audit area must name at least one gate or this proof is vacuous"
        );
        let execution_counts: std::collections::HashMap<&str, Arc<AtomicUsize>> = subset
            .gates
            .iter()
            .map(|name| (*name, Arc::new(AtomicUsize::new(0))))
            .collect();

        // 1. Full sweep simulation: the registry holds each gate once.
        for gate in &subcommands::registry() {
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

    /// WHY: a gate registered with no descriptor row has no declared subject,
    /// artifacts or proof, and a descriptor naming no gate is a contract
    /// nobody runs. Both directions fail, and the agreeing case is silent.
    #[test]
    fn a_registry_and_its_descriptors_that_disagree_both_fail() {
        let valid_descriptor = crate::gate::GateDescriptor {
            name: "dep-drift",
            help:
                "Reject a manifest that pins a workspace-managed dependency to a different version",
            package: "xtask",
            areas: &["prepublish"],
            subject: "workspace manifests",
            inputs: &[],
            artifacts: &[],
            prerequisites: &[],
            resource_class: crate::gate::ResourceClass::Cpu,
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

    /// A gate reporting whatever count it was built with.
    struct Counted(usize);

    impl crate::gate::GateBehavior for Counted {
        fn run(&self, _: &GateCtx) -> Result<Report, GateError> {
            let mut report = Report::clean();
            report.cover_complete("fixture subjects", 1);
            for index in 0..self.0 {
                report.find(crate::gate::Finding::new(
                    format!("finding {index}"),
                    "fix it",
                ));
            }
            Ok(report)
        }
    }

    /// WHY: every gate holds zero findings, so the sweep fails on the first
    /// one. A sweep that compared against a tolerated count would pass a gate
    /// whose count was raised along with its findings; the clean gate has to
    /// stay silent or the failure means nothing.
    #[test]
    fn the_sweep_fails_on_any_finding_and_passes_zero() {
        static DESCRIPTOR: crate::gate::GateDescriptor = crate::gate::GateDescriptor {
            name: "sweep-fixture",
            help: "Report the fixture's count",
            package: "xtask",
            areas: &[],
            subject: "fixture subjects",
            inputs: &[],
            artifacts: &[],
            prerequisites: &[],
            resource_class: crate::gate::ResourceClass::Cpu,
            proof: "crate::gates::sweep::tests::the_sweep_fails_on_any_finding_and_passes_zero",
        };
        static CLEAN: Counted = Counted(0);
        static ONE: Counted = Counted(1);
        let root = tempfile::tempdir().expect("tempdir");
        let clean = crate::gate::RegisteredGate::new(&DESCRIPTOR, &CLEAN);
        let red = crate::gate::RegisteredGate::new(&DESCRIPTOR, &ONE);

        assert_eq!(sweep(root.path(), &[clean]), Vec::<String>::new());
        let failures = sweep(root.path(), &[clean, red]);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("`sweep-fixture` reported 1 finding(s)"),
            "{failures:?}"
        );
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

        // Subsets that together hold every gate are the whole registry; a bare
        // `xtask gates` step on top of them runs every gate twice.
        let partitioned = workflow_failures(&names, &["dep-drift".to_string()], &every_subset);
        assert_eq!(partitioned, Vec::<String>::new());

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
        let (area_selected, area_what) =
            selection(&["--area".to_string(), subset.name.to_string()])
                .expect("a descriptor-derived area");
        assert_eq!(area_what, format!("subset `{}`", subset.name));
        assert_eq!(area_selected.len(), subset.gates.len());
        assert!(selection(&["--area".to_string()]).is_err());
        assert!(selection(&["--area".to_string(), "nope".to_string()]).is_err());
        assert!(!selected.is_empty());
        assert!(selected.len() < subcommands::registry().len());
        assert!(selection(&["--subset".to_string()]).is_err());
        assert!(selection(&["--subset".to_string(), "nope".to_string()]).is_err());
    }

    /// WHY: an ignored flag certifies a run that never happened.
    ///
    /// `--gate file-size` was read as "one gate ran" several times while the
    /// whole registry ran instead, because the runner dropped the word. Every
    /// accepted word is derived from the flags the runner actually reads, and a
    /// subset value is not mistaken for a flag.
    #[test]
    fn a_word_the_runner_does_not_read_is_rejected_rather_than_dropped() {
        assert_eq!(
            unknown_argument(&["--gate".to_string(), "file-size".to_string()]),
            Some("--gate".to_string())
        );
        assert_eq!(
            unknown_argument(&["--subset".to_string(), "ci-rules".to_string()]),
            None
        );
        assert_eq!(
            unknown_argument(&["--write-baseline".to_string()]),
            Some("--write-baseline".to_string())
        );
        assert_eq!(unknown_argument(&["--list".to_string()]), None);
        assert_eq!(unknown_argument(&[]), None);
        // A subset literally named like a flag is a value, not a second flag.
        assert_eq!(
            unknown_argument(&["--area".to_string(), "--gate".to_string()]),
            None
        );
        assert_eq!(
            unknown_argument(&["file-size".to_string()]),
            Some("file-size".to_string())
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

        let subset_gates: Vec<crate::gate::RegisteredGate> =
            registry.iter().take(5).copied().collect();
        let _ = sweep(root, &subset_gates);

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
