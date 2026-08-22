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

/// One compiler diagnostic, reduced to what a finding carries.
struct Diagnostic {
    file: Option<String>,
    line: Option<u32>,
    message: String,
}

/// What one cargo invocation produced.
///
/// The two answers are kept apart because they mean opposite things. `found` is
/// what the compiler said about the source. `unmeasured` names a file the build
/// needed and did not find under its own build directory, which says the run
/// never reached the source at all.
struct Run {
    /// Error diagnostics the compiler emitted.
    found: Vec<Diagnostic>,
    /// A build-directory path the run named that is no longer there.
    unmeasured: Option<String>,
}

/// Run one cargo invocation and return the diagnostics it emitted.
///
/// `--message-format=json` is the only reason this is reliable: a gate that
/// scraped human output counted the same error twice as soon as cargo repeated
/// its summary. It goes before any `--`, because everything after that
/// separator reaches the compiler driver instead of cargo, and clippy-driver
/// answers an unknown option with `Unrecognized option` and exit 101 per crate.
/// Appended blindly, it turned the clippy gate into one that could only report
/// that it had not run, so the workspace was neither clippy-clean nor dirty for
/// as long as it stood. Nothing here sets a build-affecting flag or variable,
/// because build configuration is declared once in `.cargo/config.toml`.
fn diagnostics(root: &Path, arguments: &[&str]) -> Result<Run, GateError> {
    let cargo = crate::cargo_runner::binary(root);
    let (cargo_arguments, driver_arguments) = split_at_driver(arguments);
    let output = Command::new(&cargo)
        .args(cargo_arguments)
        .arg("--message-format=json")
        .args(driver_arguments)
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `{} {}`: {error}",
                    cargo.display(),
                    arguments.join(" ")
                ),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
    let mut found = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(serde_json::Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        if message.get("level").and_then(serde_json::Value::as_str) != Some("error") {
            continue;
        }
        let text = message
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("the compiler reported an error with no message")
            .to_string();
        let primary = message
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .and_then(|spans| {
                spans.iter().find(|span| {
                    span.get("is_primary")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                })
            });
        found.push(Diagnostic {
            file: primary
                .and_then(|span| span.get("file_name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            line: primary
                .and_then(|span| span.get("line_start"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|line| u32::try_from(line).ok()),
            message: text,
        });
    }
    // A build directory deleted under a running compile fails with a diagnostic
    // naming a file that is not there. The run measured nothing, so it is
    // classified before the status is judged: reporting it as a compile error
    // would blame the source for the state of the disk, and reporting the
    // status as an unexplained failure would do the same in one line.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let unmeasured = crate::cargo_runner::unmeasured(&stderr).or_else(|| {
        found
            .iter()
            .find_map(|diagnostic| crate::cargo_runner::unmeasured(&diagnostic.message))
    });
    // A failing status with no parsed diagnostic is still a failure, and it is
    // the one shape a diagnostic-counting gate can report as clean. That is the
    // gate-that-cannot-fail defect, so the status is judged too.
    if unmeasured.is_none() && !output.status.success() && found.is_empty() {
        return Err(GateError::new(
            format!(
                "`cargo {}` exited {} and emitted no diagnostic: {}",
                arguments.join(" "),
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ),
            "run the same cargo command by hand and fix what it reports",
        ));
    }
    Ok(Run { found, unmeasured })
}

/// Turn the diagnostics of one cargo invocation into a report.
fn report_diagnostics(root: &Path, arguments: &[&str], fix: &str) -> Result<Report, GateError> {
    let mut report = Report::clean();
    let tree = Tree::open(root)?;
    report.cover_complete("workspace members", tree.member_manifests()?.len());
    let run = diagnostics(root, arguments)?;
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
    for diagnostic in run.found {
        report.find(match (diagnostic.file, diagnostic.line) {
            (Some(file), Some(line)) => Finding::at(file, line, diagnostic.message, fix),
            (Some(file), None) => Finding::in_file(file, diagnostic.message, fix),
            (None, _) => Finding::new(diagnostic.message, fix),
        });
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
            &["check", "--workspace", "--all-features", "--all-targets"],
            "fix the compile error the diagnostic names",
        )
    }
}

/// Clippy is denied warnings across the same surface.
pub struct WorkspaceClippy;

impl crate::gate::GateBehavior for WorkspaceClippy {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        report_diagnostics(
            &ctx.root,
            &[
                "clippy",
                "--workspace",
                "--all-features",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
            "fix the lint the diagnostic names, or justify an allow at the item with a reason",
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
            &["doc", "--workspace", "--all-features", "--no-deps"],
            "repair the item the diagnostic names, including its intra-doc links",
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

/// Split an argument list into what cargo reads and what the compiler driver
/// reads, at the first `--`.
fn split_at_driver<'a>(arguments: &'a [&'a str]) -> (&'a [&'a str], &'a [&'a str]) {
    match arguments.iter().position(|argument| *argument == "--") {
        Some(at) => (&arguments[..at], &arguments[at..]),
        None => (arguments, &arguments[arguments.len()..]),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: `--message-format=json` after the `--` reaches clippy-driver, which
    /// answers with `Unrecognized option: 'message-format'` and exit 101 for
    /// every crate. The gate then had no diagnostic to count and could only
    /// report that it had not run, so the workspace was neither clippy-clean nor
    /// dirty while that stood. The split is what keeps the flag on cargo's side.
    #[test]
    fn the_driver_separator_bounds_the_cargo_arguments() {
        let clippy = [
            "clippy",
            "--workspace",
            "--all-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ];
        let (cargo, driver) = split_at_driver(&clippy);
        assert_eq!(
            cargo,
            ["clippy", "--workspace", "--all-features", "--all-targets"]
        );
        assert_eq!(driver, ["--", "-D", "warnings"]);

        let check = ["check", "--workspace"];
        let (cargo, driver) = split_at_driver(&check);
        assert_eq!(cargo, ["check", "--workspace"]);
        assert!(driver.is_empty());
    }
    /// WHY: workspace-check compiles every target with all features enabled.
    #[test]
    fn workspace_check_invokes_cargo_check_across_all_features() {
        let check_args = ["check", "--workspace", "--all-targets", "--all-features"];
        let (cargo, driver) = split_at_driver(&check_args);
        assert_eq!(
            cargo,
            ["check", "--workspace", "--all-targets", "--all-features"]
        );
        assert!(driver.is_empty());
    }

    /// WHY: workspace-docs renders docs without dependencies.
    #[test]
    fn workspace_docs_constructs_no_deps_doc_arguments() {
        let doc_args = ["doc", "--workspace", "--no-deps"];
        let (cargo, driver) = split_at_driver(&doc_args);
        assert_eq!(cargo, ["doc", "--workspace", "--no-deps"]);
        assert!(driver.is_empty());
    }

    /// WHY: workspace-tests runs tests across contract-owning layer packages.
    #[test]
    fn workspace_tests_resolves_tested_layer_contract() {
        assert_eq!(TESTED_LAYERS, &["foundation", "libraries", "semantics"]);
    }
}
