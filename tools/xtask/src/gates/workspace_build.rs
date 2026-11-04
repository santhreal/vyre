//! The gates that build the whole workspace.
//!
//! These four used to be steps inside a `check-cat-a` composite, which was a
//! registered subcommand with its own control flow, its own pass summary and no
//! baseline. Being a composite is what kept them out of the sweep: the category
//! decided that, not the cost. They are gates now, each judging one cargo
//! invocation, and the Cat-A set is a named subset of the registry rather than a
//! subcommand that re-runs other subcommands.
//!
//! A compiler diagnostic is one finding. Counting rendered lines instead made a
//! multi-line error look like fifteen findings and a note about it look like one.

use std::path::Path;
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Turn the diagnostics of one cargo invocation into a report.
fn report_diagnostics(
    root: &Path,
    arguments: &[&str],
    fix: &str,
    judge_warnings: bool,
) -> Result<Report, GateError> {
    let mut report = Report::clean();
    let tree = Tree::open(root)?;
    report.cover_complete("workspace members", tree.member_manifests()?.len());
    let run = crate::cargo_runner::diagnostics(root, arguments, judge_warnings)?;
    if let Some(missing) = run.unmeasured {
        report.find(Finding::new(
            format!(
                "`cargo {}` measured nothing: the build named `{missing}`, which the build directory does not carry",
                arguments.join(" ")
            ),
            "run the gate again against an intact build directory; a compile whose own inputs were deleted under it reports the state of the disk, and the source it was pointed at was never read",
        ));
        report.note(format!("cargo {}", arguments.join(" ")));
        return Ok(report);
    }
    // A failing status with no parsed diagnostic is still a failure, and it is
    // the one shape a diagnostic-counting gate can report as clean. That is the
    // gate-that-cannot-fail defect, so the status is judged too.
    if run.failed_silently() {
        return Err(GateError::new(
            format!(
                "`cargo {}` exited {} and emitted no diagnostic: {}",
                arguments.join(" "),
                run.code(),
                run.stderr.trim()
            ),
            "run the same cargo command by hand and fix what it reports",
        ));
    }
    for diagnostic in &run.found {
        report.find(
            diagnostic
                .place(root, &diagnostic.message, fix)
                .unwrap_or_else(|| Finding::new(diagnostic.message.clone(), fix)),
        );
    }
    report.note(format!("cargo {}", arguments.join(" ")));
    Ok(report)
}

/// Every target of every crate compiles with every feature enabled.
pub struct WorkspaceCheck;

impl crate::gate::GateBehavior for WorkspaceCheck {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        report_diagnostics(
            &ctx.root,
            CHECK,
            "fix the compile error the diagnostic names",
            false,
        )
    }
}

/// Clippy is denied warnings across the same surface.
pub struct WorkspaceClippy;

impl crate::gate::GateBehavior for WorkspaceClippy {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        report_diagnostics(
            &ctx.root,
            CLIPPY,
            "fix the lint the diagnostic names, or justify an allow at the item with a reason",
            false,
        )
    }
}

/// Rustdoc builds the whole workspace without a broken item or a broken link.
///
/// `cargo doc` renders documentation; it does not run doctests, so nothing here
/// judges whether an example compiles. `workspace-tests` runs the doctests of
/// the crates it names, and that is the only doctest coverage the registry has.
pub struct WorkspaceDocs;

impl crate::gate::GateBehavior for WorkspaceDocs {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        report_diagnostics(
            &ctx.root,
            DOC,
            "repair the item the diagnostic names, including its intra-doc links",
            true,
        )
    }
}

/// The layers whose test suites the Cat-A surface owes on every change.
///
/// The layer is the policy; which crates sit in one is read from the ownership
/// registry at run time. A hard-coded roster of three packages named the crates
/// that a retired composite happened to run, so a crate added to a contract
/// layer was untested and nothing said so. `foundation` owns the IR and the wire
/// encoding, `libraries` the op surface, `semantics` the assignment and lifetime
/// rules.
const TESTED_LAYERS: &[&str] = &["foundation", "libraries", "semantics"];

/// Packages the ownership registry places in a tested layer.
///
/// A layer that names no crate is a finding rather than an empty roster: a
/// renamed layer would otherwise reduce this gate to running no tests and
/// reporting that nothing failed.
fn tested_packages(ctx: &GateCtx, report: &mut Report) -> Result<Vec<String>, GateError> {
    let tree = Tree::open(&ctx.root)?;
    let records = crate::gates::crate_registry::load_registry(&tree, report)?;
    let mut packages = Vec::new();
    for layer in TESTED_LAYERS {
        let mut in_layer: Vec<String> = records
            .iter()
            .filter(|record| record.layer == *layer)
            .map(|record| record.package.clone())
            .collect();
        if in_layer.is_empty() {
            report.find(Finding::in_file(
                crate::gates::crate_registry::REGISTRY,
                format!("no crate declares layer `{layer}`, so this gate would test nothing"),
                "declare the layer on the crate that owns the contract, or name the layer it \
                 was renamed to in `TESTED_LAYERS`",
            ));
        }
        packages.append(&mut in_layer);
    }
    packages.sort();
    packages.dedup();
    Ok(packages)
}

/// Every test of the contract-owning crates passes.
pub struct WorkspaceTests;

impl crate::gate::GateBehavior for WorkspaceTests {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let packages = tested_packages(ctx, &mut report)?;
        report.cover_complete("workspace packages", packages.len());
        if !report.findings.is_empty() {
            // The roster decides what runs, so a broken registry is not a tree
            // whose tests have been judged.
            return Ok(report);
        }
        let cargo = crate::cargo_runner::binary(&ctx.root);
        let mut command = Command::new(&cargo);
        command.arg("test");
        for package in &packages {
            command.args(["-p", package]);
        }
        // `--no-fail-fast` is what makes the count a count. Stopping at the
        // first failing crate pinned the number of failures to one.
        command
            .args(["--all-features", "--no-fail-fast"])
            .current_dir(&ctx.root);
        let output = command.output().map_err(|error| {
            GateError::new(
                format!("cannot run `{} test`: {error}", cargo.display()),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if let Some(missing) = crate::cargo_runner::unmeasured(&text) {
            report.find(Finding::new(
                format!(
                    "the test run measured nothing: it named `{missing}`, which the build directory does not carry"
                ),
                "run the gate again against an intact build directory; a test binary whose own inputs were deleted under it never ran the tests, and a failure read from it names the disk rather than a test",
            ));
            report.note(format!("tested {}", packages.join(", ")));
            return Ok(report);
        }
        for line in text.lines() {
            let trimmed = line.trim();
            let Some(rest) = trimmed.strip_prefix("test ") else {
                continue;
            };
            let Some(name) = rest.strip_suffix(" ... FAILED") else {
                continue;
            };
            report.find(Finding::new(
                format!("test `{name}` failed"),
                "fix the behaviour the test asserts, and never weaken the assertion to match it",
            ));
        }
        if !output.status.success() && report.findings.is_empty() {
            return Err(GateError::new(
                format!(
                    "`cargo test` exited {} and named no failing test: {}",
                    output.status.code().unwrap_or(-1),
                    text.lines().rev().take(20).collect::<Vec<_>>().join(" | ")
                ),
                "run the same cargo command by hand and fix what it reports",
            ));
        }
        report.note(format!("tested {}", packages.join(", ")));
        Ok(report)
    }
}

/// Every argument list this file hands cargo, so a test can read the real one.
///
/// Inlined at the call site, each list was only ever asserted against a copy of
/// itself in a test, which proves the copy and not the gate.
const CHECK: &[&str] = &["check", "--workspace", "--all-features", "--all-targets"];

/// Clippy's argument list, denying warnings past the driver separator.
const CLIPPY: &[&str] = &[
    "clippy",
    "--workspace",
    "--all-features",
    "--all-targets",
    "--",
    "-D",
    "warnings",
];

/// Rustdoc's argument list.
const DOC: &[&str] = &["doc", "--workspace", "--all-features", "--no-deps"];

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: every one of these gates judges the whole workspace, and a list
    /// that quietly lost `--all-features` or `--all-targets` would still report
    /// a clean tree while leaving most of it uncompiled. The earlier proofs
    /// built their own copy of each list and asserted the copy, so the gates
    /// could be narrowed without turning anything red. These read the lists the
    /// gates actually hand cargo.
    #[test]
    fn every_workspace_gate_compiles_the_whole_workspace() {
        for (name, arguments) in [("check", CHECK), ("clippy", CLIPPY), ("doc", DOC)] {
            let (cargo, _) = crate::cargo_runner::split_at_driver(arguments);
            assert_eq!(
                cargo.first(),
                Some(&name),
                "the `{name}` gate must run `cargo {name}`"
            );
            for required in ["--workspace", "--all-features"] {
                assert!(
                    cargo.contains(&required),
                    "`cargo {name}` must pass `{required}` or it judges part of the tree"
                );
            }
        }
        assert!(
            CHECK.contains(&"--all-targets") && CLIPPY.contains(&"--all-targets"),
            "a compile that skips tests and benches leaves them unjudged"
        );
        assert!(
            DOC.contains(&"--no-deps"),
            "rendering dependency documentation reports defects nobody here can fix"
        );
    }

    /// WHY: `--message-format=json` after the `--` reaches clippy-driver, which
    /// answers `Unrecognized option: 'message-format'` and exit 101 per crate,
    /// so only the clippy list may carry a driver separator at all, and what
    /// follows it must be the deny flag rather than anything cargo needs.
    #[test]
    fn only_clippy_sends_arguments_past_the_driver_separator() {
        let (cargo, driver) = crate::cargo_runner::split_at_driver(CLIPPY);
        assert_eq!(
            cargo,
            ["clippy", "--workspace", "--all-features", "--all-targets"]
        );
        assert_eq!(driver, ["--", "-D", "warnings"]);
        for (name, arguments) in [("check", CHECK), ("doc", DOC)] {
            let (_, driver) = crate::cargo_runner::split_at_driver(arguments);
            assert!(
                driver.is_empty(),
                "`cargo {name}` sends nothing to a compiler driver"
            );
        }
    }

    /// WHY: the roster of tested crates is derived from these layer names, so a
    /// layer renamed in the ownership registry reduces this gate to running no
    /// tests. Asserting the constant against a copy of itself proved the copy;
    /// this reads the registry in the checkout and fails when a name stops
    /// naming anything.
    #[test]
    fn every_tested_layer_is_one_the_registry_declares() {
        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("open the checkout");
        let mut report = Report::clean();
        let records =
            crate::gates::crate_registry::load_registry(&tree, &mut report).expect("read registry");
        assert!(
            report.findings.is_empty(),
            "the ownership registry does not parse: {:?}",
            report.findings
        );
        for layer in TESTED_LAYERS {
            assert!(
                records.iter().any(|record| record.layer == *layer),
                "no crate declares layer `{layer}`, so naming it here tests nothing"
            );
        }
    }
}
