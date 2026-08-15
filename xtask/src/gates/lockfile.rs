//! The gate that holds `Cargo.lock` to the commit.
//!
//! This was step 7 of the `release-gate` composite. A dirty lockfile at publish
//! time means the versions that were tested are not the versions that ship, and
//! nothing else in the registry reads it.

use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Reports a `Cargo.lock` that differs from the commit.
pub struct LockfileClean;

impl Gate for LockfileClean {
    fn name(&self) -> &'static str {
        "lockfile-clean"
    }

    fn help(&self) -> &'static str {
        "Fail when Cargo.lock differs from the commit, because the resolved versions would not be the tested ones"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let output = Command::new("git")
            .args(["status", "--porcelain", "Cargo.lock"])
            .current_dir(&ctx.root)
            .output()
            .map_err(|error| {
                GateError::new(
                    format!("cannot run `git status --porcelain Cargo.lock`: {error}"),
                    "run the gate inside a git checkout with git on PATH",
                )
            })?;
        if !output.status.success() {
            return Err(GateError::new(
                format!(
                    "`git status --porcelain Cargo.lock` exited {}: {}",
                    output.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
                "repair the checkout so git can report the lockfile state",
            ));
        }
        let mut report = Report::clean();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            report.find(Finding::in_file(
                "Cargo.lock",
                format!("differs from the commit: {trimmed}"),
                "commit the resolved lockfile, or restore it if the change was accidental",
            ));
        }
        Ok(report)
    }
}
