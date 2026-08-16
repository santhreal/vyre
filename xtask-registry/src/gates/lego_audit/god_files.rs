//! Check 5: the large-file review advisory.
//!
//! This check reports and never fails, and that is deliberate rather than a
//! defect in the rule. A line count is a prompt to ask whether a file carries
//! more than one responsibility, and the answer is a reading, not a threshold.
//! The blocking per-file ceiling is the `file-size` gate, which ratchets a
//! measured count per file and fails on a file over its cap. Turning this
//! advisory into a finding would either duplicate that gate at a second
//! threshold or fail every file over 500 lines, which is most of the tree.

use super::*;

/// Line count at which a source file is flagged for a split-by-responsibility
/// *review*. Crossing it is a guideline prompt, not a law and not a build
/// failure. The hard god-file ceiling (ratcheted, with a per-file exception
/// list) is enforced by the `file-size` gate.
pub(super) const LARGE_FILE_ADVISORY_LINES: usize = 500;

pub(super) fn check_5_god_files(report: &mut Report) -> usize {
    report.note(format!("[5/10] Large-file advisory (files over {LARGE_FILE_ADVISORY_LINES} lines flagged for split-by-responsibility review; non-blocking)"));
    let Some(root) = workspace_root() else {
        // A missing workspace root is a real environment failure, not a
        // size advisory, so it still fails the audit.
        report.find(violation("  ✗ workspace root not reachable from xtask. Fix: run from the vyre workspace checkout.".to_string()));
        return 1;
    };

    let mut advisories = 0usize;
    let mut errors = 0usize;
    for entry in walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "target" | "target-codex" | "target-fusion-fix"
            )
        })
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.find(violation(format!("  ✗ walkdir failed while scanning source files: {error}. Fix: make the checked source tree fully readable.")));
                errors += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let text = match read_text_bounded(path) {
            Ok(text) => text,
            Err(error) => {
                report.find(violation(format!("  ✗ {} could not be read for the large-file advisory: {error}. Fix: make the checked source tree fully readable.",
                    path.strip_prefix(&root).unwrap_or(path).display())));
                errors += 1;
                continue;
            }
        };
        let line_count = text.lines().count();
        if line_count > LARGE_FILE_ADVISORY_LINES {
            report.note(format!("  • {} has {line_count} lines. Review: does this file carry more than one responsibility? If so, split it (advisory, not a failure).",
                path.strip_prefix(&root).unwrap_or(path).display()));
            advisories += 1;
        }
    }
    if advisories == 0 {
        report.note(format!(
            "  ✓ no Rust source file is over the {LARGE_FILE_ADVISORY_LINES}-line review guideline"
        ));
    } else {
        report.note(format!("  • {advisories} file(s) over the {LARGE_FILE_ADVISORY_LINES}-line guideline flagged for review (non-blocking)"));
    }
    // Only genuine I/O errors fail this check; the size guideline is advisory.
    errors
}
