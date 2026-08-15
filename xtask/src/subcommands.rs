//! The registered subcommand table.
//!
//! Dispatch, help text and CI wiring all read this one table. They used to be
//! three hand-maintained lists, and they had stopped agreeing: 41 subcommands
//! were registered and 9 were ever invoked, so 32 gates existed, compiled, and
//! judged nothing. A gate nobody runs is worse than no gate, because it reads
//! as coverage.
//!
//! Adding a subcommand means adding a row here. `Kind` decides what CI owes it,
//! and `xtask gates` fails when that obligation is unmet, so a new row cannot
//! be quietly left unwired.

use crate::docs::docs_check;
use crate::gates::{
    check_cat_a, check_tier_deps, dep_drift, dup_scan, feature_isolation, hot_path_scan,
    hygiene_matrix, platform_boundary, sweep,
};
use crate::release::{
    feature_matrix, launch_state, metadata_matrix, package_readiness, release_conformance,
    release_gate, release_workload_matrix, version_matrix,
};
use crate::shrink;

/// Which crate implements a subcommand, and therefore what running it costs.
///
/// A `Local` subcommand reads source text, manifests, workflows or evidence
/// files, so it runs in this process against no vyre crate. The other two are
/// implemented in a crate that links vyre because it has to observe the live
/// operation registry or a measured benchmark probe; `xtask` builds and runs
/// that crate on demand instead of linking it.
#[derive(Clone, Copy)]
pub enum Home {
    /// Implemented here. The function is the entry point.
    Local(fn(&[String])),
    /// Implemented in `xtask-registry`, which links the operation registry.
    Registry,
    /// Implemented in `xtask-evidence`, which links the benchmark harness.
    Evidence,
}

impl Home {
    /// Package that implements the subcommand, or `None` when `xtask` does.
    #[must_use]
    pub fn package(self) -> Option<&'static str> {
        match self {
            Self::Local(_) => None,
            Self::Registry => Some("xtask-registry"),
            Self::Evidence => Some("xtask-evidence"),
        }
    }
}

/// Every subcommand `package` is responsible for, in table order.
#[must_use]
pub fn owned_by(package: &str) -> Vec<&'static str> {
    SUBCOMMANDS
        .iter()
        .filter(|entry| entry.home.package() == Some(package))
        .map(|entry| entry.name)
        .collect()
}

/// One row of a delegate crate's table: the name typed on the command line,
/// paired with the function that runs it.
pub type Implemented<'a> = &'a [(&'a str, fn(&[String]))];

/// Run `name` out of a delegate crate's table, reporting whether that table owns
/// it. A name it does not own runs nothing, so `xtask` stays the only place an
/// unknown subcommand is reported.
#[must_use]
pub fn dispatch(implemented: Implemented<'_>, name: &str, args: &[String]) -> bool {
    match implemented.iter().find(|(row, _)| *row == name) {
        Some((_, run)) => {
            run(args);
            true
        }
        None => false,
    }
}

/// Every disagreement between the subcommands this table assigns to `package`
/// and the table `package` actually implements.
///
/// The two are separate declarations that have to agree. A row assigned here
/// with no entry in the delegate table fails as an unknown subcommand after the
/// build has already been paid for, and an entry in the delegate table that this
/// table assigns elsewhere is unreachable. Dispatch resolves by linear search,
/// so a repeated name would shadow its second entry while both lists still
/// compared equal. Both sides are derived at call time, so a subcommand added
/// to one and not the other is reported here.
#[must_use]
pub fn delegate_table_problems(package: &str, implemented: Implemented<'_>) -> Vec<String> {
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
        problems.push(format!("`{package}` lists a subcommand more than once"));
    }
    problems.sort_unstable();
    problems
}

/// What a subcommand is for, and therefore what CI owes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Judges the tree and must run on every change. `xtask gates` executes it
    /// and holds it to its pinned baseline.
    Gate,
    /// Writes release evidence artifacts. The release pipeline owns it, so it
    /// must be named by a workflow but is not run by the sweep.
    Evidence,
    /// Re-runs other registered gates or drives full cargo builds. Wiring it
    /// into the sweep would rebuild the workspace inside a gate, so it must be
    /// named by a workflow instead and is not swept.
    Composite,
    /// Needs inputs only a caller can supply, so CI cannot run it. The string
    /// is the reason, and it is required: an empty one is a violation.
    Tool(&'static str),
    /// The sweep itself. Excluded from the gate list so it cannot recurse.
    Runner,
}

/// One registered subcommand.
pub struct Subcommand {
    /// Name as typed on the command line.
    pub name: &'static str,
    /// Argument spec shown in help.
    pub usage: &'static str,
    /// One-line description shown in help.
    pub help: &'static str,
    /// What CI owes this subcommand.
    pub kind: Kind,
    /// Arguments the sweep runs a `Gate` with. Empty means run it bare.
    pub ci_args: &'static [&'static str],
    /// Which crate implements it, and the entry point when that is this one.
    pub home: Home,
}

/// Every registered subcommand.
pub const SUBCOMMANDS: &[Subcommand] = &[
    Subcommand {
        name: "abstraction-gate",
        usage: "",
        help: "Enforce registered building-block boundaries",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "backend-matrix",
        usage: "[--output PATH]",
        help: "Probe linked backend release policy",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "bench-crossback",
        usage: "[program]",
        help: "Cross-backend perf table",
        kind: Kind::Tool(
            "prints a cross-backend performance table for a human; measured release evidence is owned by release-benchmarks",
        ),
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "bench-release",
        usage: "[--backend all]",
        help: "Run the cross-backend release benchmark coordinator",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "catalog",
        usage: "[--out DIR] [--check]",
        help: "Emit one markdown table per subsystem under docs/catalog; --check gates drift",
        kind: Kind::Gate,
        ci_args: &["--check"],
        home: Home::Registry,
    },
    Subcommand {
        name: "check-cat-a",
        usage: "",
        help: "Run every Cat-A pre-merge gate",
        kind: Kind::Composite,
        ci_args: &[],
        home: Home::Local(check_cat_a::run),
    },
    Subcommand {
        name: "check-tier-deps",
        usage: "",
        help: "Reject upward tier path dependencies",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(check_tier_deps::run),
    },
    Subcommand {
        name: "compile",
        usage: "<program.vir> --to TARGET",
        help: "Emit authenticated payloads through linked target compiler facets",
        kind: Kind::Tool("compiles a caller-supplied wire program to a caller-chosen target"),
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "conformance-matrix",
        usage: "[--check] [--output PATH]",
        help: "Enumerate or check release op/backend conformance coverage",
        kind: Kind::Gate,
        ci_args: &["--check"],
        home: Home::Registry,
    },
    Subcommand {
        name: "dep-drift",
        usage: "",
        help: "Fail if a manifest pins a workspace-managed dependency to a different version",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(dep_drift::run),
    },
    Subcommand {
        name: "docs-check",
        usage: "",
        help: "Validate manifest-backed documentation lifecycle and generated navigation",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(docs_check::run),
    },
    Subcommand {
        name: "dup-scan",
        usage: "[--write-baseline] [--lower-pin CRATE] [--report [CRATE]]",
        help: "Measure cross-file duplicate source blocks against the pinned per-crate baseline",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(dup_scan::run),
    },
    Subcommand {
        name: "feature-isolation",
        usage: "[--list] [--sweep [--write] [--only-unrecorded]] [--member NAME]",
        help: "Hold every feature selection the manifests declare to its recorded compile outcome",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(feature_isolation::run),
    },
    Subcommand {
        name: "feature-matrix",
        usage: "[--output PATH]",
        help: "Generate crate feature evidence matrix",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(feature_matrix::run),
    },
    Subcommand {
        name: "gate1",
        usage: "",
        help: "Enforce Gate 1 complexity budget",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "gates",
        usage: "[--list]",
        help: "Run every registered gate and hold it to the pinned baseline",
        kind: Kind::Runner,
        ci_args: &[],
        home: Home::Local(sweep::run),
    },
    Subcommand {
        name: "heuristic-audit",
        usage: "[--strict]",
        help: "Surface hand-rolled heuristics that should be self-consumer calls",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "hot-path-scan",
        usage: "[--strict]",
        help: "Scan files in HOT_PATHS.toml for clone/alloc/lock patterns",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(hot_path_scan::run),
    },
    Subcommand {
        name: "hygiene-matrix",
        usage: "[--output PATH]",
        help: "Scan source hygiene release blockers",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(hygiene_matrix::run),
    },
    Subcommand {
        name: "launch-state",
        usage: "[--output PATH]",
        help: "Generate public launch completion state evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(launch_state::run),
    },
    Subcommand {
        name: "lego-audit",
        usage: "[--report-only|--with-repo|--write-baseline] [--duplicate-report-json PATH]",
        help: "Deeper LEGO-block enforcement and composition baseline management",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "lego-quick",
        usage: "[--all]",
        help: "Fast pre-commit boundary checks",
        kind: Kind::Gate,
        ci_args: &["--all"],
        home: Home::Registry,
    },
    Subcommand {
        name: "list-ops",
        usage: "[--write PATH|--check]",
        help: "Render or check the schema-derived operation inventory",
        kind: Kind::Gate,
        ci_args: &["--check"],
        home: Home::Registry,
    },
    Subcommand {
        name: "metadata-matrix",
        usage: "[--output PATH]",
        help: "Generate crate metadata evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(metadata_matrix::run),
    },
    Subcommand {
        name: "op-matrix",
        usage: "[--output PATH]",
        help: "Generate operation/backend coverage evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "operation-schema",
        usage: "[--output PATH] [--check] [--validate PATH]",
        help: "Generate or verify the canonical live operation contract schema",
        kind: Kind::Gate,
        ci_args: &["--check"],
        home: Home::Registry,
    },
    Subcommand {
        name: "optimization-corpus",
        usage: "[--output PATH]",
        help: "Generate release optimization corpus manifest",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "optimization-docs",
        usage: "[--output PATH] [--check]",
        help: "Generate or check the source-owned optimizer pass reference",
        kind: Kind::Gate,
        ci_args: &["--check"],
        home: Home::Registry,
    },
    Subcommand {
        name: "optimization-matrix",
        usage: "[--output PATH]",
        help: "Generate release optimization integration evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "package-readiness",
        usage: "[--output PATH]",
        help: "Generate pre-publish package order evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(package_readiness::run),
    },
    Subcommand {
        name: "platform-boundary",
        usage: "",
        help: "Fail on consumer names in platform crate docs and comments",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Local(platform_boundary::run),
    },
    Subcommand {
        name: "primitive-admission-gate",
        usage: "",
        help: "Enforce canonical LEGO primitive adoption and exceptions",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "print-composition",
        usage: "<op_id>",
        help: "Walk an op's Region tree and print its decomposition chain",
        kind: Kind::Tool("walks one operation id supplied by the caller"),
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "release-benchmarks",
        usage: "[--backend cuda]",
        help: "Generate long-running release benchmark artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "release-conformance",
        usage: "[--backend all]",
        help: "Generate real backend conformance artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(release_conformance::run),
    },
    Subcommand {
        name: "release-evidence",
        usage: "",
        help: "Generate cheap structural release evidence artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "release-gate",
        usage: "",
        help: "Pre-publish sanity checks",
        kind: Kind::Composite,
        ci_args: &[],
        home: Home::Local(release_gate::run),
    },
    Subcommand {
        name: "release-workload-matrix",
        usage: "[--output PATH]",
        help: "Generate cheap release workload family evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(release_workload_matrix::run),
    },
    Subcommand {
        name: "shrink",
        usage: "<file.vir> <oracle.sh>",
        help: "Delta-debug a crashing wire formulation down to a minimal reproducer",
        kind: Kind::Tool(
            "delta-debugs a caller-supplied crashing wire file against a caller-supplied oracle",
        ),
        ci_args: &[],
        home: Home::Local(shrink::run),
    },
    Subcommand {
        name: "trace-f32",
        usage: "<op_id>",
        help: "Run an op's test inputs through the reference and dump the expected output",
        kind: Kind::Tool("dumps reference output for one operation id supplied by the caller"),
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "verify-rewrite-proofs",
        usage: "",
        help: "Verify optimizer rewrite proof fixtures",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Registry,
    },
    Subcommand {
        name: "version-matrix",
        usage: "[--output PATH]",
        help: "Generate manifest version matrix",
        kind: Kind::Evidence,
        ci_args: &[],
        home: Home::Local(version_matrix::run),
    },
    Subcommand {
        name: "vyre-release-gate",
        usage: "[--launch-complete] [--manifest PATH]",
        help: "Enforce prepublication or launch-complete evidence closure",
        kind: Kind::Gate,
        ci_args: &[],
        home: Home::Evidence,
    },
    Subcommand {
        name: "whats-similar",
        usage: "(--op-id <id>|--all) [--duplicate-report-json PATH]",
        help: "Duplicate query by IR shape",
        kind: Kind::Gate,
        ci_args: &["--all"],
        home: Home::Registry,
    },
];

/// Look one subcommand up by name.
#[must_use]
pub fn find(name: &str) -> Option<&'static Subcommand> {
    SUBCOMMANDS.iter().find(|entry| entry.name == name)
}

/// Every subcommand the sweep must execute.
#[must_use]
pub fn gates() -> Vec<&'static Subcommand> {
    SUBCOMMANDS
        .iter()
        .filter(|entry| entry.kind == Kind::Gate)
        .collect()
}

/// Render the help text from the table.
#[must_use]
pub fn help_text() -> String {
    let width = SUBCOMMANDS
        .iter()
        .map(|entry| entry.name.len() + entry.usage.len() + 1)
        .max()
        .unwrap_or(0);
    let mut text = String::from(
        "vyre xtask runner\n\nUSAGE:\n  cargo run --bin xtask -- <subcommand> [options]\n\nSUBCOMMANDS:\n",
    );
    for entry in SUBCOMMANDS {
        let invocation = if entry.usage.is_empty() {
            entry.name.to_string()
        } else {
            format!("{} {}", entry.name, entry.usage)
        };
        text.push_str(&format!("  {invocation:width$}  {}\n", entry.help));
    }
    text.push_str(&format!("  {:width$}  Print this message\n", "--help"));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: dispatch resolves by name, so a duplicate row makes the second one
    /// unreachable and its gate silently stops running.
    #[test]
    fn every_subcommand_name_is_unique() {
        let mut names: Vec<&str> = SUBCOMMANDS.iter().map(|entry| entry.name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            before,
            names.len(),
            "duplicate subcommand name in the table"
        );
    }

    /// WHY: the exemption is the whole escape hatch. An empty reason turns it
    /// into an unexplained hole, which is how the 32 unwired gates happened.
    #[test]
    fn every_exempt_tool_states_why_ci_cannot_run_it() {
        for entry in SUBCOMMANDS {
            if let Kind::Tool(reason) = entry.kind {
                assert!(
                    reason.len() > 20,
                    "`{}` is exempt from CI without a real reason",
                    entry.name
                );
            }
        }
    }

    /// WHY: a gate whose CI arguments do not actually judge anything passes
    /// vacuously. These four only report when asked to check.
    #[test]
    fn check_only_gates_are_run_with_check() {
        for name in [
            "catalog",
            "conformance-matrix",
            "list-ops",
            "operation-schema",
        ] {
            let entry = find(name).expect("registered");
            assert!(
                entry.ci_args.contains(&"--check"),
                "`{name}` must run with --check or it only regenerates output"
            );
        }
    }

    /// WHY: exactly one runner may exist, or the sweep could recurse into a
    /// second copy of itself.
    #[test]
    fn there_is_exactly_one_runner() {
        let runners = SUBCOMMANDS
            .iter()
            .filter(|entry| entry.kind == Kind::Runner)
            .count();
        assert_eq!(runners, 1);
    }

    /// WHY: help is generated now, so it cannot drift from dispatch again.
    #[test]
    fn help_lists_every_registered_subcommand() {
        let text = help_text();
        for entry in SUBCOMMANDS {
            assert!(
                text.contains(entry.name),
                "`{}` is registered but absent from help",
                entry.name
            );
        }
    }

    /// WHY: a delegated row names the crate that has to be built to run it. A
    /// name that is not a workspace member would only fail at the moment an
    /// operator invoked the gate, which is the worst place to learn it.
    #[test]
    fn every_delegated_subcommand_names_a_workspace_member() {
        let members = std::fs::read_to_string(crate::checkout::checkout_root().join("Cargo.toml"))
            .expect("Fix: the workspace manifest must be readable from xtask");
        for entry in SUBCOMMANDS {
            let Some(package) = entry.home.package() else {
                continue;
            };
            assert!(
                members.contains(&format!("\"{package}\"")),
                "`{}` delegates to `{package}`, which is not a workspace member",
                entry.name
            );
        }
    }

    /// WHY: `owned_by` is what each delegated crate checks its own dispatch
    /// against, so the partition has to be total. A row that belongs to no
    /// package and is not local would be dispatched by nobody.
    #[test]
    fn every_subcommand_belongs_to_exactly_one_home() {
        let delegated: usize = ["xtask-registry", "xtask-evidence"]
            .iter()
            .map(|package| owned_by(package).len())
            .sum();
        let local = SUBCOMMANDS
            .iter()
            .filter(|entry| entry.home.package().is_none())
            .count();
        assert_eq!(local + delegated, SUBCOMMANDS.len());
    }

    fn unreachable_run(_args: &[String]) {
        panic!("a subcommand outside the table must never run");
    }

    fn noop(_args: &[String]) {}

    /// WHY: a delegate crate is built only because `xtask` decided a subcommand
    /// belongs to it, so answering to a name it was not given would run the
    /// wrong gate under the right name. Dispatch must refuse every registered
    /// name outside the table, and the panic proves it refuses without running
    /// anything.
    #[test]
    fn dispatch_refuses_every_name_outside_the_table() {
        let table: [(&str, fn(&[String])); 1] = [("owned", unreachable_run)];
        for entry in SUBCOMMANDS {
            assert!(
                !dispatch(&table, entry.name, &["xtask".to_string()]),
                "`{}` is not in the table and must not dispatch",
                entry.name
            );
        }
        assert!(!dispatch(&table, "", &["xtask".to_string()]));
    }

    /// WHY: the delegate crates check their own tables against this one through
    /// `delegate_table_problems`, so a checker that reported nothing would let
    /// every kind of drift through while reading as coverage. Each way the two
    /// declarations can disagree must be named, and the live assignment is read
    /// at run time so a new delegated subcommand cannot escape the check.
    #[test]
    fn the_delegate_checker_names_every_kind_of_drift() {
        let package = "xtask-registry";
        let assigned = owned_by(package);
        let first = *assigned.first().expect("the registry owns subcommands");

        let unassigned: [(&str, fn(&[String])); 1] = [("dep-drift", noop)];
        let mut expected = assigned
            .iter()
            .map(|name| format!("`{package}` is assigned `{name}` but does not implement it"))
            .collect::<Vec<_>>();
        expected.push(format!(
            "`{package}` implements `dep-drift` but is not assigned it"
        ));
        expected.sort_unstable();
        assert_eq!(delegate_table_problems(package, &unassigned), expected);

        let mut duplicated = assigned
            .iter()
            .map(|name| (*name, noop as fn(&[String])))
            .collect::<Vec<_>>();
        duplicated.push((first, noop));
        assert_eq!(
            delegate_table_problems(package, &duplicated),
            vec![format!("`{package}` lists a subcommand more than once")]
        );

        let complete = assigned
            .iter()
            .map(|name| (*name, noop as fn(&[String])))
            .collect::<Vec<_>>();
        assert_eq!(
            delegate_table_problems(package, &complete),
            Vec::<String>::new()
        );
    }
}
