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

use crate::bench::{bench_crossback, bench_release, release_benchmarks};
use crate::docs::{catalog, docs_check, list_ops, op_matrix, operation_schema, optimization_docs};
use crate::gates::{
    abstraction_gate, check_cat_a, check_tier_deps, dep_drift, dup_scan, gate1, gates,
    heuristic_audit, hot_path_scan, hygiene_matrix, lego_audit, lego_quick, platform_boundary,
    verify_rewrite_proofs, whats_similar,
};
use crate::release::{
    backend_matrix, conformance_matrix, feature_matrix, launch_state, metadata_matrix,
    optimization_corpus, optimization_matrix, package_readiness, release_conformance,
    release_evidence, release_gate, release_workload_matrix, version_matrix, vyre_release_gate,
};
use crate::{c_parser_clang_oracle, compile, print_composition, shrink, trace_f32};

/// What a subcommand is for, and therefore what CI owes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
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
pub(crate) struct Subcommand {
    /// Name as typed on the command line.
    pub(crate) name: &'static str,
    /// Argument spec shown in help.
    pub(crate) usage: &'static str,
    /// One-line description shown in help.
    pub(crate) help: &'static str,
    /// What CI owes this subcommand.
    pub(crate) kind: Kind,
    /// Arguments the sweep runs a `Gate` with. Empty means run it bare.
    pub(crate) ci_args: &'static [&'static str],
    /// Entry point.
    pub(crate) run: fn(&[String]),
}

/// Every registered subcommand.
pub(crate) const SUBCOMMANDS: &[Subcommand] = &[
    Subcommand {
        name: "abstraction-gate",
        usage: "",
        help: "Enforce registered building-block boundaries",
        kind: Kind::Gate,
        ci_args: &[],
        run: abstraction_gate::run,
    },
    Subcommand {
        name: "backend-matrix",
        usage: "[--output PATH]",
        help: "Probe linked backend release policy",
        kind: Kind::Evidence,
        ci_args: &[],
        run: backend_matrix::run,
    },
    Subcommand {
        name: "bench-crossback",
        usage: "[program]",
        help: "Cross-backend perf table",
        kind: Kind::Tool(
            "prints a cross-backend performance table for a human; measured release evidence is owned by release-benchmarks",
        ),
        ci_args: &[],
        run: bench_crossback::run,
    },
    Subcommand {
        name: "bench-release",
        usage: "[--backend all]",
        help: "Run the cross-backend release benchmark coordinator",
        kind: Kind::Evidence,
        ci_args: &[],
        run: bench_release::run,
    },
    Subcommand {
        name: "c-parser-clang-oracle",
        usage: "--corpus DIR --vyre-report PATH [--output PATH]",
        help: "Cross-check the C frontend's records against a clang AST oracle over a corpus",
        kind: Kind::Tool(
            "needs a caller-supplied C corpus, a prior parser report for that corpus, and a clang install on the host",
        ),
        ci_args: &[],
        run: c_parser_clang_oracle::run,
    },
    Subcommand {
        name: "catalog",
        usage: "[--out DIR] [--check]",
        help: "Emit one markdown table per subsystem under docs/catalog; --check gates drift",
        kind: Kind::Gate,
        ci_args: &["--check"],
        run: catalog::run,
    },
    Subcommand {
        name: "check-cat-a",
        usage: "",
        help: "Run every Cat-A pre-merge gate",
        kind: Kind::Composite,
        ci_args: &[],
        run: check_cat_a::run,
    },
    Subcommand {
        name: "check-tier-deps",
        usage: "",
        help: "Reject upward tier path dependencies",
        kind: Kind::Gate,
        ci_args: &[],
        run: check_tier_deps::run,
    },
    Subcommand {
        name: "compile",
        usage: "<program.vir> --to TARGET",
        help: "Emit authenticated payloads through linked target compiler facets",
        kind: Kind::Tool("compiles a caller-supplied wire program to a caller-chosen target"),
        ci_args: &[],
        run: compile::run,
    },
    Subcommand {
        name: "conformance-matrix",
        usage: "[--check] [--output PATH]",
        help: "Enumerate or check release op/backend conformance coverage",
        kind: Kind::Gate,
        ci_args: &["--check"],
        run: conformance_matrix::run,
    },
    Subcommand {
        name: "dep-drift",
        usage: "",
        help: "Fail if a manifest pins a workspace-managed dependency to a different version",
        kind: Kind::Gate,
        ci_args: &[],
        run: dep_drift::run,
    },
    Subcommand {
        name: "docs-check",
        usage: "",
        help: "Validate manifest-backed documentation lifecycle and generated navigation",
        kind: Kind::Gate,
        ci_args: &[],
        run: docs_check::run,
    },
    Subcommand {
        name: "dup-scan",
        usage: "[--write-baseline] [--output PATH]",
        help: "Measure cross-file duplicate source blocks against the pinned per-crate baseline",
        kind: Kind::Gate,
        ci_args: &[],
        run: dup_scan::run,
    },
    Subcommand {
        name: "feature-matrix",
        usage: "[--output PATH]",
        help: "Generate crate feature evidence matrix",
        kind: Kind::Evidence,
        ci_args: &[],
        run: feature_matrix::run,
    },
    Subcommand {
        name: "gate1",
        usage: "",
        help: "Enforce Gate 1 complexity budget",
        kind: Kind::Gate,
        ci_args: &[],
        run: gate1::run,
    },
    Subcommand {
        name: "gates",
        usage: "[--list]",
        help: "Run every registered gate and hold it to the pinned baseline",
        kind: Kind::Runner,
        ci_args: &[],
        run: gates::run,
    },
    Subcommand {
        name: "heuristic-audit",
        usage: "[--strict]",
        help: "Surface hand-rolled heuristics that should be self-consumer calls",
        kind: Kind::Gate,
        ci_args: &[],
        run: heuristic_audit::run,
    },
    Subcommand {
        name: "hot-path-scan",
        usage: "[--strict]",
        help: "Scan files in HOT_PATHS.toml for clone/alloc/lock patterns",
        kind: Kind::Gate,
        ci_args: &[],
        run: hot_path_scan::run,
    },
    Subcommand {
        name: "hygiene-matrix",
        usage: "[--output PATH]",
        help: "Scan source hygiene release blockers",
        kind: Kind::Gate,
        ci_args: &[],
        run: hygiene_matrix::run,
    },
    Subcommand {
        name: "launch-state",
        usage: "[--output PATH]",
        help: "Generate public launch completion state evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: launch_state::run,
    },
    Subcommand {
        name: "lego-audit",
        usage: "[--report-only|--with-repo|--write-baseline] [--duplicate-report-json PATH]",
        help: "Deeper LEGO-block enforcement and composition baseline management",
        kind: Kind::Gate,
        ci_args: &[],
        run: lego_audit::run,
    },
    Subcommand {
        name: "lego-quick",
        usage: "[--all]",
        help: "Fast pre-commit boundary checks",
        kind: Kind::Gate,
        ci_args: &["--all"],
        run: lego_quick::run,
    },
    Subcommand {
        name: "list-ops",
        usage: "[--write PATH|--check]",
        help: "Render or check the schema-derived operation inventory",
        kind: Kind::Gate,
        ci_args: &["--check"],
        run: list_ops::run,
    },
    Subcommand {
        name: "metadata-matrix",
        usage: "[--output PATH]",
        help: "Generate crate metadata evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: metadata_matrix::run,
    },
    Subcommand {
        name: "op-matrix",
        usage: "[--output PATH]",
        help: "Generate operation/backend coverage evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: op_matrix::run,
    },
    Subcommand {
        name: "operation-schema",
        usage: "[--output PATH] [--check] [--validate PATH]",
        help: "Generate or verify the canonical live operation contract schema",
        kind: Kind::Gate,
        ci_args: &["--check"],
        run: operation_schema::run,
    },
    Subcommand {
        name: "optimization-corpus",
        usage: "[--output PATH]",
        help: "Generate release optimization corpus manifest",
        kind: Kind::Evidence,
        ci_args: &[],
        run: optimization_corpus::run,
    },
    Subcommand {
        name: "optimization-docs",
        usage: "[--output PATH] [--check]",
        help: "Generate or check the source-owned optimizer pass reference",
        kind: Kind::Gate,
        ci_args: &["--check"],
        run: optimization_docs::run,
    },
    Subcommand {
        name: "optimization-matrix",
        usage: "[--output PATH]",
        help: "Generate release optimization integration evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: optimization_matrix::run,
    },
    Subcommand {
        name: "package-readiness",
        usage: "[--output PATH]",
        help: "Generate pre-publish package order evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: package_readiness::run,
    },
    Subcommand {
        name: "platform-boundary",
        usage: "",
        help: "Fail on consumer names in platform crate docs and comments",
        kind: Kind::Gate,
        ci_args: &[],
        run: platform_boundary::run,
    },
    Subcommand {
        name: "primitive-admission-gate",
        usage: "",
        help: "Enforce canonical LEGO primitive adoption and exceptions",
        kind: Kind::Gate,
        ci_args: &[],
        run: |_args| lego_audit::run_primitive_admission_gate(),
    },
    Subcommand {
        name: "print-composition",
        usage: "<op_id>",
        help: "Walk an op's Region tree and print its decomposition chain",
        kind: Kind::Tool("walks one operation id supplied by the caller"),
        ci_args: &[],
        run: print_composition::run,
    },
    Subcommand {
        name: "release-benchmarks",
        usage: "[--backend cuda]",
        help: "Generate long-running release benchmark artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        run: release_benchmarks::run,
    },
    Subcommand {
        name: "release-conformance",
        usage: "[--backend all]",
        help: "Generate real backend conformance artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        run: release_conformance::run,
    },
    Subcommand {
        name: "release-evidence",
        usage: "",
        help: "Generate cheap structural release evidence artifacts",
        kind: Kind::Evidence,
        ci_args: &[],
        run: release_evidence::run,
    },
    Subcommand {
        name: "release-gate",
        usage: "",
        help: "Pre-publish sanity checks",
        kind: Kind::Composite,
        ci_args: &[],
        run: release_gate::run,
    },
    Subcommand {
        name: "release-workload-matrix",
        usage: "[--output PATH]",
        help: "Generate cheap release workload family evidence",
        kind: Kind::Evidence,
        ci_args: &[],
        run: release_workload_matrix::run,
    },
    Subcommand {
        name: "shrink",
        usage: "<file.vir> <oracle.sh>",
        help: "Delta-debug a crashing wire formulation down to a minimal reproducer",
        kind: Kind::Tool(
            "delta-debugs a caller-supplied crashing wire file against a caller-supplied oracle",
        ),
        ci_args: &[],
        run: shrink::run,
    },
    Subcommand {
        name: "trace-f32",
        usage: "<op_id>",
        help: "Run an op's test inputs through the reference and dump the expected output",
        kind: Kind::Tool("dumps reference output for one operation id supplied by the caller"),
        ci_args: &[],
        run: trace_f32::run_cmd,
    },
    Subcommand {
        name: "verify-rewrite-proofs",
        usage: "",
        help: "Verify optimizer rewrite proof fixtures",
        kind: Kind::Gate,
        ci_args: &[],
        run: verify_rewrite_proofs::run,
    },
    Subcommand {
        name: "version-matrix",
        usage: "[--output PATH]",
        help: "Generate manifest version matrix",
        kind: Kind::Evidence,
        ci_args: &[],
        run: version_matrix::run,
    },
    Subcommand {
        name: "vyre-release-gate",
        usage: "[--prepublish] [--manifest PATH]",
        help: "Enforce final or prepublication evidence closure",
        kind: Kind::Gate,
        ci_args: &[],
        run: vyre_release_gate::run,
    },
    Subcommand {
        name: "whats-similar",
        usage: "(--op-id <id>|--all) [--duplicate-report-json PATH]",
        help: "Duplicate query by IR shape",
        kind: Kind::Gate,
        ci_args: &["--all"],
        run: whats_similar::run,
    },
];

/// Look one subcommand up by name.
#[must_use]
pub(crate) fn find(name: &str) -> Option<&'static Subcommand> {
    SUBCOMMANDS.iter().find(|entry| entry.name == name)
}

/// Every subcommand the sweep must execute.
#[must_use]
pub(crate) fn gates() -> Vec<&'static Subcommand> {
    SUBCOMMANDS
        .iter()
        .filter(|entry| entry.kind == Kind::Gate)
        .collect()
}

/// Render the help text from the table.
#[must_use]
pub(crate) fn help_text() -> String {
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
        assert_eq!(before, names.len(), "duplicate subcommand name in the table");
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
        for name in ["catalog", "conformance-matrix", "list-ops", "operation-schema"] {
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
}
