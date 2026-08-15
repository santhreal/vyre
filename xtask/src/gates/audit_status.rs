//! Status-managed audits tag every finding row.
//!
//! An audit that declares a status legend is a tracked queue of work, and a
//! numbered row without a status tag is open work hiding as prose. The tags are
//! `open`, `in_progress` and `fixed`, in backticks, immediately after the row
//! number.
//!
//! The execution-order section at the end of such an audit is a summary of rows
//! stated elsewhere, so its numbering is out of scope.

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The tags a numbered finding row may carry.
const TAGS: &[&str] = &["`open`", "`in_progress`", "`fixed`"];

/// The heading that ends the finding rows.
const EXECUTION_ORDER: &str = "## Highest Leverage Execution Order";

/// The line that marks an audit as status managed.
const LEGEND: &str = "Status legend:";

/// Every managed audit document carries a status legend and per-row statuses.
pub struct AuditStatus;

impl Gate for AuditStatus {
    fn name(&self) -> &'static str {
        "audit-status"
    }

    fn help(&self) -> &'static str {
        "untagged finding rows in status-managed audits"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let audits = tree.scope(&["audits"], &["md"])?;
        let mut managed = 0_usize;
        for file in &audits {
            let text = tree.read(file)?;
            if !text.lines().any(|line| line.starts_with(LEGEND)) {
                continue;
            }
            managed += 1;
            for (number, line) in crate::gates::scan::numbered(&text) {
                if line.starts_with(EXECUTION_ORDER) {
                    break;
                }
                let Some(rest) = numbered_row(line) else {
                    continue;
                };
                if !TAGS.iter().any(|tag| rest.starts_with(tag)) {
                    report.find(Finding::at(
                        file.clone(),
                        number,
                        format!("finding row carries no status tag: {}", line.trim()),
                        "prefix the row with `open`, `in_progress` or `fixed` so open work \
                         cannot read as prose",
                    ));
                }
            }
        }
        report.note(format!("{managed} status-managed audit file(s)"));
        if managed == 0 {
            report.find(Finding::new(
                "no status-managed audit file declares a status legend",
                "give the audit a `Status legend:` line, or delete the rule if audits no \
                 longer track work this way",
            ));
        }
        Ok(report)
    }
}

/// The text after `N. ` on a numbered row, when the line is one.
fn numbered_row(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let digits = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(0);
    if digits == 0 {
        return None;
    }
    let rest = trimmed[digits..].strip_prefix(". ")?;
    Some(rest.trim_start())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a version list, a step in a procedure and a finding row all start
    /// with a number, and only the last is in scope. The audits in the tree carry
    /// all three shapes.
    #[test]
    fn a_numbered_row_is_told_apart_from_prose() {
        assert_eq!(numbered_row("1. `open` a thing"), Some("`open` a thing"));
        assert_eq!(numbered_row("  12. untagged row"), Some("untagged row"));
        assert_eq!(numbered_row("- a bullet"), None);
        assert_eq!(numbered_row("1.5 is not a row"), None);
        assert_eq!(numbered_row("text 1. not at the start"), None);
    }
}
