//! The gate runner.
//!
//! Before any gate runs, the registry is checked against itself: descriptors,
//! artifact ownership, proof symbols and finding capability. Workflow wiring is
//! the `ci-registry` gate's rule.
//!
//! Every gate holds zero findings. There is no per-gate pin: a count above zero
//! fails the sweep, and the only way to make a red gate green is to change what
//! it judges.

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
/// Every defect in the registry itself, checked before any gate runs.
///
/// A gate with no descriptor row, a descriptor with no gate, an artifact two
/// gates claim, a proof symbol that does not resolve, and a gate that cannot
/// report a finding all make every verdict the sweep prints unreliable, so they
/// fail every run. Workflow wiring is the `ci-registry` gate's rule and subset
/// membership is `gate-canon`'s; each runs once, in its own subset.
fn registry_failures(root: &Path, gate_names: &[&str]) -> Vec<String> {
    let mut failures = metadata_failures(root, gate_names, crate::gate_metadata::GATE_METADATA);
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
    let mut failures = registry_failures(&root, &gate_names);
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

    /// WHY: Section 182.5.6 and performance contract require O(1) workspace hashing per sweep run,
    /// rather than O(gates) rehashing the entire workspace 123 times.
    #[test]
    fn sweep_performs_exactly_one_capture_and_one_verification_regardless_of_gate_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join("test.txt"), "hello").expect("write file");

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
