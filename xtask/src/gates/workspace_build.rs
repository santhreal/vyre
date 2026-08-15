//! The gates that run a full cargo build of the workspace.
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

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// One compiler diagnostic, reduced to what a finding carries.
struct Diagnostic {
    file: Option<String>,
    line: Option<u32>,
    message: String,
}

/// The argv cargo receives, with `--message-format=json` on cargo's side.
///
/// Everything after `--` belongs to the lint driver. Appended at the end, the
/// flag reached clippy-driver, which answered `Unrecognized option:
/// 'message-format'` for every crate, so `workspace-clippy` could not run at
/// all and reported an error instead of a lint.
fn argv<'a>(arguments: &[&'a str]) -> Vec<&'a str> {
    let mut argv = Vec::with_capacity(arguments.len() + 1);
    let at = arguments
        .iter()
        .position(|argument| *argument == "--")
        .unwrap_or(arguments.len());
    argv.extend_from_slice(&arguments[..at]);
    argv.push("--message-format=json");
    argv.extend_from_slice(&arguments[at..]);
    argv
}

/// Run one cargo invocation and return the diagnostics it emitted.
///
/// The json format is the only reason this is reliable: a gate that scraped
/// human output counted the same error twice as soon as cargo repeated its
/// summary. Nothing here sets a build-affecting flag or variable, because build
/// configuration is declared once in `.cargo/config.toml`.
fn diagnostics(root: &Path, arguments: &[&str]) -> Result<Vec<Diagnostic>, GateError> {
    let cargo = crate::output_arg::cargo_runner(root);
    let output = Command::new(&cargo)
        .args(argv(arguments))
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
    // A failing status with no parsed diagnostic is still a failure, and it is
    // the one shape a diagnostic-counting gate can report as clean. That is the
    // gate-that-cannot-fail defect, so the status is judged too.
    if !output.status.success() && found.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
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
    Ok(found)
}

/// Turn the diagnostics of one cargo invocation into a report.
fn report_diagnostics(root: &Path, arguments: &[&str], fix: &str) -> Result<Report, GateError> {
    let mut report = Report::clean();
    for diagnostic in diagnostics(root, arguments)? {
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

impl Gate for WorkspaceCheck {
    fn name(&self) -> &'static str {
        "workspace-check"
    }

    fn help(&self) -> &'static str {
        "Compile every target of every workspace crate with all features enabled"
    }

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

impl Gate for WorkspaceClippy {
    fn name(&self) -> &'static str {
        "workspace-clippy"
    }

    fn help(&self) -> &'static str {
        "Hold every target of every workspace crate to clippy with warnings denied"
    }

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

/// Rustdoc builds the whole workspace without a broken link or a bad doctest.
pub struct WorkspaceDocs;

impl Gate for WorkspaceDocs {
    fn name(&self) -> &'static str {
        "workspace-docs"
    }

    fn help(&self) -> &'static str {
        "Build the documentation of every workspace crate with all features enabled"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        report_diagnostics(
            &ctx.root,
            &["doc", "--workspace", "--all-features", "--no-deps"],
            "repair the item the diagnostic names, including its intra-doc links",
        )
    }
}

/// The crates whose tests the Cat-A surface owes on every change.
///
/// These are the three the composite ran, and the reason each is here is the
/// contract it covers: `vyre-libs` the op surface, `vyre-foundation` the IR and
/// wire encoding, `vyre-reference` the assignment and lifetime rules.
const TESTED_PACKAGES: &[&str] = &["vyre-libs", "vyre-foundation", "vyre-reference"];

/// Every test of the contract-owning crates passes.
pub struct WorkspaceTests;

impl Gate for WorkspaceTests {
    fn name(&self) -> &'static str {
        "workspace-tests"
    }

    fn help(&self) -> &'static str {
        "Run the test suites of the crates that own the op, IR and reference contracts"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let cargo = crate::output_arg::cargo_runner(&ctx.root);
        let mut report = Report::clean();
        let mut command = Command::new(&cargo);
        command.arg("test");
        for package in TESTED_PACKAGES {
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
        report.note(format!("tested {}", TESTED_PACKAGES.join(", ")));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: `workspace-clippy` is the only gate that judges lints, and for the
    /// whole of the cutover it judged nothing: the json flag was appended after
    /// the `--` that hands the rest to clippy-driver, which rejected it per
    /// crate, so the gate reported a gate error instead of a lint. The flag has
    /// to sit on cargo's side of the separator for every gate in this module,
    /// not only the one that was reported.
    #[test]
    fn the_json_flag_stays_on_cargos_side_of_the_lint_separator() {
        let clippy = argv(&[
            "clippy",
            "--workspace",
            "--all-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]);
        let flag = clippy
            .iter()
            .position(|argument| *argument == "--message-format=json")
            .expect("cargo must be asked for json diagnostics");
        let separator = clippy
            .iter()
            .position(|argument| *argument == "--")
            .expect("the lint arguments must still be passed through");
        assert!(
            flag < separator,
            "the driver receives everything after `--`: {clippy:?}"
        );
        assert_eq!(&clippy[separator..], &["--", "-D", "warnings"]);
    }

    /// WHY: an invocation with no separator must still ask for json, and the
    /// flag must not be inserted before the subcommand, which cargo reads first.
    #[test]
    fn an_invocation_without_a_separator_still_asks_for_json() {
        let check = argv(&["check", "--workspace", "--all-features", "--all-targets"]);
        assert_eq!(
            check,
            vec![
                "check",
                "--workspace",
                "--all-features",
                "--all-targets",
                "--message-format=json",
            ]
        );
    }
}
