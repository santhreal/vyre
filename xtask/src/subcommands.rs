//! The gate registry.
//!
//! Dispatch, help text, and the sweep all read from the authoritative descriptor
//! registry in `crate::gate_metadata::GATE_METADATA`. Every gate has a single
//! authoritative descriptor defining its stable name, help text, owner package,
//! enforcement areas, authoritative subject class, generated artifact paths,
//! prerequisites, and mutation-proof test identity.
//!
//! There is no separate kind or secondary descriptor table. Every entry is a gate,
//! every gate has a baseline row, and the sweep runs all of them. Local gates are
//! executed in process, while delegated gates (implemented in `xtask-registry` and
//! `xtask-evidence`) are dispatched to child processes based on descriptor package
//! metadata. Named subsets are derived from descriptor area membership.

use crate::gate::RegisteredGate;

/// A named set of gates, so a caller can ask for part of the registry without
/// any gate being exempt from the whole of it.
pub struct Subset {
    /// Name passed to `xtask gates --subset`.
    pub name: &'static str,
    /// What the set is for, shown in help.
    pub help: &'static str,
    /// Gates in the set, derived from gate metadata.
    pub gates: Vec<&'static str>,
}

const AREA_HELP: &[(&str, &str)] = &[
    (
        "ci-rules",
        "Whether CI workflows and their declared gate registry cover the required checks",
    ),
    (
        "contract-rules",
        "Whether source, manifest, API, parity, and workspace contracts hold",
    ),
    (
        "docs",
        "Whether generated documentation artifacts match the tree",
    ),
    (
        "hot-path",
        "Allocation, blocking, synchronous reads, and bounded growth on dispatch paths",
    ),
    (
        "lego-audit",
        "Whether registered building blocks compose under their semantic ownership laws",
    ),
    (
        "prepublish",
        "What must hold before publishing, beyond what a dry run catches",
    ),
    (
        "release-evidence",
        "Whether committed release evidence matches the manifests, lockfile, and recorded runs",
    ),
];

/// Named subsets derived from each gate metadata row's areas.
///
/// # Panics
///
/// Panics when an area declared in gate metadata is missing from `AREA_HELP`.
#[must_use]
pub fn subsets() -> Vec<Subset> {
    crate::gate_metadata::areas()
        .into_iter()
        .map(|name| Subset {
            name,
            help: AREA_HELP
                .iter()
                .find_map(|(area, help)| (*area == name).then_some(*help))
                .expect("Fix: add help text for every gate area in AREA_HELP"),
            gates: crate::gate_metadata::gates_in_area(name),
        })
        .collect()
}

/// Every registered gate, in name order.
#[must_use]
pub fn registry() -> Vec<RegisteredGate> {
    let local = crate::gates::GATES
        .iter()
        .chain(crate::docs::GATES)
        .chain(crate::release::GATES)
        .map(|(name, behavior)| {
            let descriptor = crate::gate_metadata::descriptor_by_name(name);
            assert_eq!(
                descriptor.package, "xtask",
                "local gate `{name}` must be owned by xtask"
            );
            RegisteredGate::new(descriptor, *behavior)
        });
    let delegated = crate::gate_metadata::GATE_METADATA
        .iter()
        .filter(|descriptor| descriptor.package != "xtask")
        .map(RegisteredGate::delegated);
    let mut gates: Vec<RegisteredGate> = local.chain(delegated).collect();
    gates.sort_unstable_by_key(RegisteredGate::name);
    gates
}

/// Look one gate up by name.
#[must_use]
pub fn find(name: &str) -> Option<RegisteredGate> {
    if name == "primitive-admission-gate" {
        return find("lego-primitive-coverage");
    }
    registry().into_iter().find(|gate| gate.name() == name)
}

/// Look one subset up by name.
#[must_use]
pub fn subset(name: &str) -> Option<Subset> {
    subsets().into_iter().find(|subset| subset.name == name)
}

/// Every gate `package` is responsible for, in registry order.
#[must_use]
pub fn owned_by(package: &str) -> Vec<&'static str> {
    crate::gate_metadata::owned_by(package)
}

/// Every disagreement between the gates this registry assigns to `package` and
/// the gates `package` actually implements.
#[must_use]
pub fn delegate_table_problems(
    package: &str,
    implemented: &[(&'static str, &'static dyn crate::gate::GateBehavior)],
) -> Vec<String> {
    let mut problems = Vec::new();
    let assigned = owned_by(package);
    let mut names: Vec<&str> = implemented.iter().map(|(name, _)| *name).collect();
    for name in &assigned {
        if !names.contains(name) {
            problems.push(format!(
                "`{package}` is assigned `{name}` but does not implement it"
            ));
        }
    }
    for name in &names {
        if !assigned.contains(name) {
            problems.push(format!(
                "`{package}` implements `{name}` but is not assigned it"
            ));
        }
    }
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    if before != names.len() {
        problems.push(format!("`{package}` lists a gate more than once"));
    }
    problems.sort_unstable();
    problems
}

/// Render the help text from the registry.
#[must_use]
pub fn help_text() -> String {
    let gates = registry();
    let subsets = subsets();
    let width = gates
        .iter()
        .map(|gate| gate.name().len())
        .chain(subsets.iter().map(|subset| subset.name.len()))
        .max()
        .unwrap_or(0);
    let mut text = String::from(
        "vyre xtask runner\n\nUSAGE:\n  cargo run --bin xtask -- <subcommand> [options]\n\nSUBCOMMANDS:\n",
    );
    for gate in &gates {
        let name = gate.name();
        let help = crate::gate_metadata::descriptor(name).map_or("", |d| d.help);
        text.push_str(&format!("  {name:width$}  {help}\n"));
    }
    text.push_str(&format!(
        "  {:width$}  Run every registered gate and hold each to its pinned finding count\n",
        crate::gates::sweep::RUNNER
    ));
    text.push_str(&format!("  {:width$}  Print this message\n", "--help"));
    text.push_str("\nSUBSETS:\n");
    for subset in &subsets {
        text.push_str(&format!("  {:width$}  {}\n", subset.name, subset.help));
    }
    text.push_str("\nEvery subcommand is a gate. A gate that owns a generated artifact\n");
    text.push_str("checks it against the tree and rewrites it when passed --write.\n");
    text.push_str("Run a subcommand with --help for the options it reads.\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// WHY: dispatch resolves by name, so a duplicate registration makes the
    /// second one unreachable and its gate silently stops running.
    #[test]
    fn every_gate_name_is_unique() {
        let mut names: Vec<&str> = registry().iter().map(|gate| gate.name()).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate gate name in the registry");
    }

    /// WHY: `--help` is a question about a gate, and a gate that reads the tree
    /// to answer it has run the check the caller asked it not to run. Every
    /// option a gate names in its help line is answered from what the gate
    /// declares, and the roster is the registry, so a gate registered later is
    /// judged without being listed here. A gate implemented in another package
    /// declares its usage there and is judged by that package's own table.
    #[test]
    fn every_gate_answers_help_with_the_options_it_names() {
        assert_eq!(crate::gate::usage_gaps(&registry()), Vec::<String>::new());
    }

    /// WHY: a declared write argument exempts one invocation from the workspace
    /// mutation guard, which is the guard that refuses a comparison run that
    /// changed a tracked file. An undocumented exemption is unfindable, and a
    /// gate with no declared artifact has nothing it is allowed to write, so the
    /// roster is the registry and a gate that grows a write argument is judged
    /// without being listed here.
    #[test]
    fn every_write_argument_is_documented_and_writes_a_declared_artifact() {
        assert_eq!(
            crate::gate::write_argument_gaps(&registry()),
            Vec::<String>::new()
        );
    }

    /// WHY: a subset that names an unregistered gate silently runs fewer gates
    /// than its name promises, which is the exemption this whole registry
    /// exists to remove.
    #[test]
    fn every_subset_names_registered_gates() {
        for subset in subsets() {
            assert!(!subset.gates.is_empty(), "`{}` is empty", subset.name);
            for name in &subset.gates {
                assert!(
                    find(name).is_some(),
                    "subset `{}` names `{name}`, which is not registered",
                    subset.name
                );
            }
        }
    }
    /// WHY: help text is the only second table keyed by area. A retired area
    /// left here is a stale public selector, while an area missing here makes
    /// registry construction panic instead of producing actionable help.
    #[test]
    fn area_help_exactly_matches_descriptor_areas() {
        let declared = crate::gate_metadata::areas()
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let documented = AREA_HELP
            .iter()
            .map(|(name, _)| *name)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(documented, declared);
    }

    /// WHY: `gates` is the runner, not a gate. Registering it would let the
    /// sweep recurse into itself, and a subset named `gates` would do the same.
    #[test]
    fn the_runner_is_not_a_gate() {
        assert!(find(crate::gates::sweep::RUNNER).is_none());
        assert!(subset(crate::gates::sweep::RUNNER).is_none());
    }

    /// WHY: help is generated from the registry, so it cannot drift from
    /// dispatch again.
    #[test]
    fn help_lists_every_registered_gate_and_subset() {
        let text = help_text();
        for gate in registry() {
            assert!(
                text.contains(gate.name()),
                "`{}` is registered but absent from help",
                gate.name()
            );
        }
        for subset in subsets() {
            assert!(
                text.contains(subset.name),
                "subset `{}` is absent from help",
                subset.name
            );
        }
    }

    /// WHY: a delegated gate names the crate that has to be built to run it. A
    /// name that is not a workspace member would only fail at the moment an
    /// operator invoked the gate, which is the worst place to learn it.
    #[test]
    fn every_delegated_gate_names_a_workspace_member() {
        let members = std::fs::read_to_string(crate::checkout::checkout_root().join("Cargo.toml"))
            .expect("Fix: the workspace manifest must be readable from xtask");
        for gate in registry() {
            let Some(desc) = crate::gate_metadata::descriptor(gate.name()) else {
                continue;
            };
            if desc.package == "xtask" {
                continue;
            }
            let package = desc.package;
            assert!(
                members.contains(&format!("\"{package}\"")),
                "`{}` delegates to `{package}`, which is not a workspace member",
                gate.name()
            );
        }
    }

    /// WHY: `owned_by` is what each delegated crate checks its own table
    /// against, so the partition has to be total. A gate that belongs to no
    /// package and is not local would be dispatched by nobody.
    #[test]
    fn every_gate_belongs_to_exactly_one_home() {
        let delegated: usize = ["xtask-registry", "xtask-evidence"]
            .iter()
            .map(|package| owned_by(package).len())
            .sum();
        let local = crate::gate_metadata::owned_by("xtask").len();
        assert_eq!(local + delegated, registry().len());
    }

    /// WHY: delegated registry entries are derived directly from descriptor ownership.
    /// A local behavior accidentally attached to an external descriptor would bypass the
    /// child package that owns its implementation.
    #[test]
    fn delegated_registry_entries_derive_from_metadata() {
        let expected: Vec<(&'static str, &'static str)> = crate::gate_metadata::GATE_METADATA
            .iter()
            .filter(|descriptor| descriptor.package != "xtask")
            .map(|descriptor| (descriptor.name, descriptor.package))
            .collect();
        let actual: Vec<(&'static str, &'static str)> = registry()
            .iter()
            .filter(|gate| gate.package() != "xtask")
            .map(|gate| {
                assert!(gate.is_delegated(), "delegated gate has local behavior");
                (gate.name(), gate.package())
            })
            .collect();
        assert_eq!(actual, expected);
    }

    /// WHY: Section 182.2.2 requires every gate in metadata to have an executable implementation.
    #[test]
    fn every_gate_in_metadata_has_an_executable_gate_in_registry() {
        let reg = registry();
        let reg_names: BTreeSet<&str> = reg.iter().map(|g| g.name()).collect();
        let meta_names: BTreeSet<&str> = crate::gate_metadata::GATE_METADATA
            .iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(
            reg_names, meta_names,
            "Registry and GATE_METADATA must match 1:1"
        );
    }

    /// A dummy behavior that exists only to be named in a table under test.
    struct DummyBehavior;
    static DUMMY_BEHAVIOR: DummyBehavior = DummyBehavior;

    impl crate::gate::GateBehavior for DummyBehavior {
        fn run(
            &self,
            _ctx: &crate::gate::GateCtx,
        ) -> Result<crate::gate::Report, crate::gate::GateError> {
            panic!("a gate under table test must never run")
        }
    }

    /// WHY: the delegate crates check their own tables against this registry
    /// through `delegate_table_problems`, so a checker that reported nothing
    /// would let every kind of drift through while reading as coverage. Each way
    /// the two declarations can disagree must be named, and the live assignment
    /// is read at run time so a new delegated gate cannot escape the check.
    #[test]
    fn the_delegate_checker_names_every_kind_of_drift() {
        let package = "xtask-registry";
        let assigned = owned_by(package);
        let first = *assigned.first().expect("the registry owns gates");

        let unassigned: [(&'static str, &'static dyn crate::gate::GateBehavior); 1] =
            [("dep-drift", &DUMMY_BEHAVIOR)];
        let mut expected = assigned
            .iter()
            .map(|name| format!("`{package}` is assigned `{name}` but does not implement it"))
            .collect::<Vec<_>>();
        expected.push(format!(
            "`{package}` implements `dep-drift` but is not assigned it"
        ));
        expected.sort_unstable();
        assert_eq!(delegate_table_problems(package, &unassigned), expected);

        let owned: Vec<(&'static str, &'static dyn crate::gate::GateBehavior)> = assigned
            .iter()
            .map(|name| {
                (
                    *name,
                    &DUMMY_BEHAVIOR as &'static dyn crate::gate::GateBehavior,
                )
            })
            .collect();
        assert_eq!(
            delegate_table_problems(package, &owned),
            Vec::<String>::new()
        );

        let mut repeated = owned.clone();
        repeated.push((first, &DUMMY_BEHAVIOR));
        assert_eq!(
            delegate_table_problems(package, &repeated),
            vec![format!("`{package}` lists a gate more than once")]
        );
    }
}
