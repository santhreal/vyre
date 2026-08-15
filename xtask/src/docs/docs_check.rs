//! The `docs-check` gate: the documentation manifest matches the tree.
//!
//! The manifest validator is a Python program, so this gate runs it and turns
//! each line it reports into a finding. A validator that cannot be launched is a
//! gate that could not run, not a clean tree.

use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// The validator this gate runs, relative to the checkout root.
const VALIDATOR: &str = "scripts/docs_manifest.py";

/// Holds the documentation manifest to the pages and navigation on disk.
pub struct DocsCheck;

impl Gate for DocsCheck {
    fn name(&self) -> &'static str {
        "docs-check"
    }

    fn help(&self) -> &'static str {
        "Hold the manifest-backed documentation lifecycle and generated navigation to the tree"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let output = Command::new("python3")
            .arg(VALIDATOR)
            .arg("--check")
            .current_dir(&ctx.root)
            .output()
            .map_err(|error| {
                GateError::new(
                    format!("cannot run `python3 {VALIDATOR} --check`: {error}"),
                    "install python3, which the documentation manifest validator needs",
                )
            })?;

        let mut report = Report::clean();
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                report.note(format!("documented page {line}"));
            }
            return Ok(report);
        }
        for line in String::from_utf8_lossy(&output.stderr).lines() {
            let message = line.trim();
            if message.is_empty() {
                continue;
            }
            report.find(Finding::in_file(
                VALIDATOR,
                message.to_string(),
                "record the page in docs/DOCS.toml, or regenerate the navigation with `python3 scripts/docs_manifest.py --write`",
            ));
        }
        if report.count() == 0 {
            return Err(GateError::new(
                format!(
                    "`python3 {VALIDATOR} --check` exited {} and said nothing",
                    output.status.code().unwrap_or(-1)
                ),
                "make the validator report what it rejected; a refusal a reader cannot act on is not a finding",
            ));
        }
        Ok(report)
    }
}
