//! The gate that holds `Cargo.lock` to the commit.
//!
//! This was step 7 of the `release-gate` composite. A dirty lockfile at publish
//! time means the versions that were tested are not the versions that ship, and
//! nothing else in the registry reads it.

use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};

/// Reports a `Cargo.lock` that differs from the commit.
pub struct LockfileClean;

impl crate::gate::GateBehavior for LockfileClean {
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
        collect_lockfile_findings(&String::from_utf8_lossy(&output.stdout), &mut report);
        Ok(report)
    }
}

fn collect_lockfile_findings(stdout: &str, report: &mut Report) {
    report.cover_complete("workspace lockfile", 1);
    for line in stdout.lines() {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lockfile_findings_detect_porcelain_diff_and_pass_clean() {
        let mut clean_report = Report::clean();
        collect_lockfile_findings("", &mut clean_report);
        assert!(clean_report.findings.is_empty());

        let mut dirty_report = Report::clean();
        collect_lockfile_findings(" M Cargo.lock\n", &mut dirty_report);
        assert_eq!(dirty_report.findings.len(), 1);
        assert!(dirty_report.findings[0]
            .message
            .contains("differs from the commit: M Cargo.lock"));
    }
}
