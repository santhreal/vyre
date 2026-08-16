//! The gate registry.
//!
//! Dispatch, help text and the sweep all read this one registry. They used to
//! read a table whose rows carried a `Kind`, and the kind decided whether a
//! check could fail a build, whether it had a baseline, whether the sweep saw
//! it, and whether anyone ran it at all: 41 subcommands were registered and 9
//! were ever invoked, so 32 checks existed, compiled, and judged nothing.
//!
//! There is no kind now. Every entry is a gate, every gate has a baseline row,
//! and the sweep runs all of them. A generated artifact is not a category, it is
//! a gate that checks the artifact against the tree and rewrites it under
//! `--write`. A check that needs caller input is not a category, it is a gate
//! that runs over a registered corpus and takes the caller's input as a flag.
//! Delegation to a crate that links vyre is a property of one gate, carried by
//! `Delegated`, not a class of its own.
//!
//! Adding a gate means adding it to the slice its area owns. The registry is
//! assembled from those slices at run time, so no list here is maintained by
//! hand.

use crate::gate::{Delegated, Gate};

/// A named set of gates, so a caller can ask for part of the registry without
/// any gate being exempt from the whole of it.
///
/// This is what the two composites became. `check-cat-a` and `release-gate`
/// were registered subcommands that ran other subcommands, which gave them
/// their own control flow, their own pass summary, no baseline and no place in
/// the sweep. A subset names gates and nothing else, so the runner holds every
/// member to the same contract however it was selected.
pub struct Subset {
    /// Name passed to `xtask gates --subset`.
    pub name: &'static str,
    /// What the set is for, shown in help.
    pub help: &'static str,
    /// Gates in the set, by registered name.
    pub gates: &'static [&'static str],
}

/// Every named subset.
pub static SUBSETS: &[Subset] = &[
    Subset {
        name: "cat-a",
        help: "What a Cat-A author runs before opening a pull request",
        gates: &[
            "workspace-check",
            "workspace-clippy",
            "workspace-tests",
            "workspace-docs",
            "op-names",
            "parity-testing-isolated",
            "platform-boundary",
        ],
    },
    Subset {
        name: "prepublish",
        help: "What must hold before publishing, beyond what a dry run catches",
        gates: &[
            "operation-schema",
            "list-ops",
            "catalog",
            "gate1",
            "abstraction-gate",
            "cross-target",
            "dep-drift",
            "platform-boundary",
            "vyre-release-gate",
            "lockfile-clean",
        ],
    },
    Subset {
        name: "composition",
        help: "Whether the registered building blocks still compose the way the rules say",
        gates: &[
            "lego-audit",
            "lego-quick",
            "primitive-admission-gate",
            "whats-similar",
        ],
    },
    Subset {
        name: "structure",
        help: "Whether the tree still has the shape the layering and hygiene rules require",
        gates: &[
            "bench-baselines",
            "check-tier-deps",
            "dup-scan",
            "example-capability",
            "hot-path-scan",
            "hygiene-matrix",
            "heuristic-audit",
        ],
    },
    Subset {
        name: "docs",
        help: "Whether the generated documentation artifacts still match the tree",
        gates: &[
            "architecture-contract",
            "cli-docs",
            "crate-ownership",
            "crate-readmes",
            "docs-check",
            "docs-coupling",
            "docs-references",
            "op-matrix",
            "docs-register",
            "optimization-docs",
            "release-docs",
            "testing-guides",
        ],
    },
    Subset {
        name: "benchmarks",
        help: "Whether the measured benchmark surface is registered, covered and inside its declared budget",
        gates: &[
            "bench-coverage",
            "bench-release",
            "bench-smoke-runtime",
            "release-benchmarks",
            "release-workload-matrix",
        ],
    },
    Subset {
        name: "release-evidence",
        help: "Whether the committed release evidence still matches the manifests, the lockfile and the recorded runs",
        gates: &[
            "conformance-matrix",
            "launch-state",
            "metadata-matrix",
            "optimization-corpus",
            "optimization-matrix",
            "package-readiness",
            "release-conformance",
            "release-evidence",
            "version-matrix",
        ],
    },
    Subset {
        name: "ir",
        help: "Whether the IR still compiles, reduces, traces and proves what it claims",
        gates: &[
            "compile",
            "shrink",
            "print-composition",
            "trace-f32",
            "verify-rewrite-proofs",
            "bench-crossback",
        ],
    },
    Subset {
        name: "manifest-rules",
        help: "What the manifests must say about each other and about the layering, read without cargo",
        gates: &[
            "workspace-membership",
            "path-deps-resolve",
            "internal-dep-versions",
            "layering",
            "neutral-crates",
            "feature-matrix",
            "feature-isolation",
        ],
    },
    Subset {
        name: "source-rules",
        help: "What every tracked source file must be: compiled by a target, parseable, and inside its size cap",
        gates: &[
            "source-reachability",
            "source-include-module",
            "source-parses",
            "file-size",
        ],
    },
    Subset {
        name: "hot-path-rules",
        help: "Allocation, blocking and unbounded growth on the dispatch path",
        gates: &[
            "hot-path-nested-rows",
            "hot-path-blocking-wait",
            "hot-path-unbounded-cache",
            "hot-path-owned-dispatch",
            "hot-path-unbounded-read",
            "hot-path-inventory",
            "hot-path-reserve",
        ],
    },
    Subset {
        name: "lint-rules",
        help: "Lint hygiene, unsafe justification and property-test coverage",
        gates: &[
            "lint-expect-fix",
            "lint-one-policy",
            "lint-unsafe-budget",
            "lint-unsafe-justification",
            "proptest-coverage",
        ],
    },
    Subset {
        name: "contract-rules",
        help: "Frozen public surfaces, wire field parity, device loudness and the unification ratchets",
        gates: &[
            "frozen-contracts",
            "backend-extension",
            "backend-matrix",
            "program-wire-fields",
            "public-api-paths",
            "public-api-snapshot",
            "readback-ring",
            "unification",
            "gpu-loudness",
            "shader-source",
        ],
    },
    Subset {
        name: "repo-rules",
        help: "What the checkout carries, what the release evidence cites, what the documents claim, and whether the registry itself was softened",
        gates: &[
            "repo-hygiene",
            "single-backlog",
            "platform-consumer-docs",
            "doc-claims",
            "contract-in-source",
            "evidence-paths",
            "invariant-paths",
            "ci-matrix",
            "ci-required",
            "gate-canon",
        ],
    },
];

/// Every gate implemented in a crate that links vyre.
///
/// These read the live operation registry, a linked backend driver or a
/// measured benchmark probe, none of which exist in source text. `xtask` links
/// no vyre crate, so it builds the owning package on demand and reads back the
/// report the child serialises.
static DELEGATED: &[Delegated] = &[
    Delegated {
        name: "abstraction-gate",
        help: "Enforce registered building-block boundaries",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "bench-crossback",
        help: "Derive the cross-backend comparison from the committed release benchmark evidence and hold the recorded table to it; --write records it",
        package: "xtask-evidence",
        generates: true,
    },
    Delegated {
        name: "catalog",
        help: "Hold docs/generated/catalog.toml to the live operation inventory; --write regenerates it",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "compile",
        help: "Compile the registered release corpus; --program ID narrows to one case, --input PATH compiles one wire file, --to ID also compiles that registered target, --out DIR writes the payloads",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "conformance-matrix",
        help: "Hold release op and backend conformance coverage to the recorded matrix; --write regenerates it",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "cross-target",
        help: "Compile the product crates for every target_os the source declares a cfg arm for",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "gate1",
        help: "Enforce the Gate 1 complexity budget",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "heuristic-audit",
        help: "Report hand-rolled heuristics that should be self-consumer calls",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "lego-audit",
        help: "Hold registered composition to the ten composition laws; --write records the composition baseline",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "list-ops",
        help: "Hold docs/generated/op-inventory.toml to the live operation registry; --write regenerates it",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "operation-schema",
        help: "Hold the canonical live operation contract schema to the registry; --write regenerates it, --validate PATH judges one document",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "optimization-docs",
        help: "Hold docs/generated/optimizer-passes.toml to the passes the source declares; --write regenerates it",
        package: "xtask-registry",
        generates: true,
    },
    Delegated {
        name: "primitive-admission-gate",
        help: "Enforce canonical primitive adoption and its recorded exceptions",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "print-composition",
        help: "Walk the decomposition chain of every registered operation; --op-id ID narrows to one",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "release-benchmarks",
        help: "Measure the release benchmark surface; --write records the evidence",
        package: "xtask-evidence",
        generates: true,
    },
    Delegated {
        name: "release-evidence",
        help: "Hold the cheap structural release evidence to the tree; --write regenerates it",
        package: "xtask-evidence",
        generates: true,
    },
    Delegated {
        name: "shrink",
        help: "Delta-debug every registered corpus case that fails its oracle down to a minimal reproducer; --program ID narrows to one, --oracle PATH replaces the oracle",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "trace-f32",
        help: "Run the recorded test inputs of every registered operation through the reference; --op-id ID narrows to one",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "verify-rewrite-proofs",
        help: "Verify every optimizer rewrite proof fixture",
        package: "xtask-registry",
        generates: false,
    },
    Delegated {
        name: "vyre-release-gate",
        help: "Enforce release evidence closure; the default judges the prepublication set, --launch-complete judges the post-ship set, --manifest PATH names another manifest",
        package: "xtask-evidence",
        generates: false,
    },
    Delegated {
        name: "whats-similar",
        help: "Report duplicate operations by IR shape across the whole registry; --op-id ID narrows to one",
        package: "xtask-registry",
        generates: false,
    },
];

/// Every registered gate, in name order.
///
/// Assembled from one slice per area at run time. The sweep enumerates this, so
/// registering a gate is what wires it: there is no second list to update and
/// no way to register a gate the sweep does not run.
#[must_use]
pub fn registry() -> Vec<&'static dyn Gate> {
    let mut gates: Vec<&'static dyn Gate> = crate::gates::GATES
        .iter()
        .copied()
        .chain(crate::docs::GATES.iter().copied())
        .chain(crate::release::GATES.iter().copied())
        .chain(DELEGATED.iter().map(|gate| gate as &'static dyn Gate))
        .collect();
    gates.sort_unstable_by_key(|gate| gate.name());
    gates
}

/// Look one gate up by name.
#[must_use]
pub fn find(name: &str) -> Option<&'static dyn Gate> {
    registry().into_iter().find(|gate| gate.name() == name)
}

/// Look one subset up by name.
#[must_use]
pub fn subset(name: &str) -> Option<&'static Subset> {
    SUBSETS.iter().find(|subset| subset.name == name)
}

/// Every gate `package` is responsible for, in registry order.
#[must_use]
pub fn owned_by(package: &str) -> Vec<&'static str> {
    registry()
        .into_iter()
        .filter(|gate| gate.package() == Some(package))
        .map(Gate::name)
        .collect()
}

/// Every disagreement between the gates this registry assigns to `package` and
/// the gates `package` actually implements.
///
/// The two are separate declarations that have to agree. A gate assigned here
/// with no entry in the delegate table fails as an unknown name after the build
/// has already been paid for, and an entry in the delegate table that this
/// registry assigns elsewhere is unreachable. The child resolves by linear
/// search, so a repeated name would shadow its second entry while both lists
/// still compared equal. Both sides are derived at call time, so a gate added to
/// one and not the other is reported here.
#[must_use]
pub fn delegate_table_problems(package: &str, implemented: &[&dyn Gate]) -> Vec<String> {
    let mut problems = Vec::new();
    let assigned = owned_by(package);
    let mut names: Vec<&str> = implemented.iter().map(|gate| gate.name()).collect();
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
    let width = gates
        .iter()
        .map(|gate| gate.name().len())
        .chain(SUBSETS.iter().map(|subset| subset.name.len()))
        .max()
        .unwrap_or(0);
    let mut text = String::from(
        "vyre xtask runner\n\nUSAGE:\n  cargo run --bin xtask -- <subcommand> [options]\n\nSUBCOMMANDS:\n",
    );
    for gate in &gates {
        let name = gate.name();
        text.push_str(&format!("  {name:width$}  {}\n", gate.help()));
    }
    text.push_str(&format!(
        "  {:width$}  Run every registered gate and hold each to its pinned finding count\n",
        crate::gates::sweep::RUNNER
    ));
    text.push_str(&format!("  {:width$}  Print this message\n", "--help"));
    text.push_str("\nSUBSETS:\n");
    for subset in SUBSETS {
        text.push_str(&format!("  {:width$}  {}\n", subset.name, subset.help));
    }
    text.push_str("\nEvery subcommand is a gate. A gate that owns a generated artifact\n");
    text.push_str("checks it against the tree and rewrites it when passed --write.\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// WHY: a subset that names an unregistered gate silently runs fewer gates
    /// than its name promises, which is the exemption this whole registry
    /// exists to remove.
    #[test]
    fn every_subset_names_registered_gates() {
        for subset in SUBSETS {
            assert!(!subset.gates.is_empty(), "`{}` is empty", subset.name);
            for name in subset.gates {
                assert!(
                    find(name).is_some(),
                    "subset `{}` names `{name}`, which is not registered",
                    subset.name
                );
            }
        }
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
        for subset in SUBSETS {
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
            let Some(package) = gate.package() else {
                continue;
            };
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
        let local = registry()
            .iter()
            .filter(|gate| gate.package().is_none())
            .count();
        assert_eq!(local + delegated, registry().len());
    }

    /// A gate that exists only to be named in a table under test.
    struct Named(&'static str);

    impl Gate for Named {
        fn name(&self) -> &'static str {
            self.0
        }

        fn help(&self) -> &'static str {
            "a gate that exists only under test"
        }

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

        let unassigned = Named("dep-drift");
        let mut expected = assigned
            .iter()
            .map(|name| format!("`{package}` is assigned `{name}` but does not implement it"))
            .collect::<Vec<_>>();
        expected.push(format!(
            "`{package}` implements `dep-drift` but is not assigned it"
        ));
        expected.sort_unstable();
        assert_eq!(
            delegate_table_problems(package, &[&unassigned as &dyn Gate]),
            expected
        );

        let owned: Vec<Named> = assigned.iter().map(|name| Named(*name)).collect();
        let repeated = Named(first);
        let mut complete: Vec<&dyn Gate> = owned.iter().map(|gate| gate as &dyn Gate).collect();
        assert_eq!(
            delegate_table_problems(package, &complete),
            Vec::<String>::new()
        );

        complete.push(&repeated);
        assert_eq!(
            delegate_table_problems(package, &complete),
            vec![format!("`{package}` lists a gate more than once")]
        );
    }
}
