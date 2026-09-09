//! `operation-law-decisions` - enforce every registered semantic operation carries algebraic laws or an explicit opaque decision.
//!
//! The live registry answers for the operations linked into this binary. A
//! registration in a test file is not one of them: it reaches the registry only
//! in the binary that compiles that file, and a device-gated one only on a host
//! with a device. Four such registrations declared no transform decision, and
//! the first of them to be validated panicked inside a `LazyLock`, which
//! poisoned the registry and failed seven unrelated tests with a message naming
//! the lock rather than the operation. So the tree is read as well: every
//! `inventory::submit!` that constructs an `OperationRegistration` is checked
//! whether or not this binary links it.

use std::collections::BTreeSet;
use std::path::Path;

use structure_gate::source_scan::{
    mask_comments_and_strings, matching_brace, rust_sources_with_text,
};
use vyre_foundation::operation::SemanticOperation;
use xtask::gate::{Finding, GateCtx, GateError, Report};

/// Entry point for `cargo xtask operation-law-decisions`.
pub struct OperationLawDecisions;

impl xtask::gate::GateBehavior for OperationLawDecisions {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let registry = vyre_registry_link::operation::live_operation_registry();
        let mut report = Report::clean();
        let operations: Vec<SemanticOperation> = registry.iter().collect();
        report.cover_complete("registered operations", operations.len());

        let findings = check_operations(&operations);
        for finding in findings {
            report.find(finding);
        }

        let (submissions, source_findings) = check_sources(&ctx.root);
        for finding in source_findings {
            report.find(finding);
        }

        report.note(format!(
            "{} registered semantic operation(s) and {submissions} source registration(s) checked for transform decisions",
            operations.len()
        ));
        Ok(report)
    }
}

/// Check that every operation in `operations` carries either valid algebraic laws
/// or an explicit, non-placeholder opaque decision.
#[must_use]
pub fn check_operations(operations: &[SemanticOperation]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let known_laws: BTreeSet<&str> = vyre_spec::law_catalog().iter().copied().collect();

    for op in operations {
        if op.laws.is_empty() && op.opaque_reason().is_none() {
            findings.push(Finding::new(
                format!(
                    "operation `{}` has neither an algebraic law nor an explicit opaque decision",
                    op.id
                ),
                "declare algebraic laws with .with_laws(...) or an explicit opaque decision with .with_opaque(\"...\")",
            ));
            continue;
        }

        if let Some(reason) = op.opaque_reason() {
            let trimmed = reason.trim();
            if trimmed.len() < 5 || is_placeholder(trimmed) {
                findings.push(Finding::new(
                    format!(
                        "operation `{}` records an invalid or placeholder opaque reason `{reason}`",
                        op.id
                    ),
                    "state a concrete, non-empty one-line reason explaining why no algebraic transform applies",
                ));
            }
        }

        for law in op.laws {
            if !known_laws.contains(law) {
                findings.push(Finding::new(
                    format!("operation `{}` cites unknown law `{law}`", op.id),
                    "use a known law name from vyre_spec::law_catalog()",
                ));
            }
        }
    }

    findings
}

/// Every `inventory::submit!` in the tree that constructs an
/// `OperationRegistration`, and a finding for each that records no transform
/// decision.
///
/// Returns the number of submissions read alongside the findings, so the report
/// states the population it judged rather than only what it rejected.
///
/// Comments and string literals are masked first, because this tree embeds
/// sample registrations as raw-string fixtures for other gates to parse. A
/// submission is recognized only at column zero, which is where an item can
/// appear; that is also what keeps an indented fixture inside a masked literal
/// from being read as one.
pub fn check_sources(root: &Path) -> (usize, Vec<Finding>) {
    let mut submissions = 0usize;
    let mut findings = Vec::new();
    for source in rust_sources_with_text(root) {
        let structure_gate::source_scan::SourceText::Read { path, text } = source else {
            continue;
        };
        let masked = mask_comments_and_strings(&text);
        for (line, block) in submission_blocks(&masked) {
            if !block.contains("OperationRegistration") {
                continue;
            }
            submissions += 1;
            if records_decision(block) {
                continue;
            }
            findings.push(Finding::new(
                format!(
                    "{path}:{line}: operation registration records no transform decision"
                ),
                "declare the algebraic laws it obeys with `with_laws`, or state why it has none with `with_opaque`; an undeclared registration panics registry validation and poisons the registry for every later reader in that binary",
            ));
        }
    }
    (submissions, findings)
}

/// Each column-zero `inventory::submit!` block in `masked`, as a one-based line
/// number and the braced block including both braces.
fn submission_blocks(masked: &str) -> Vec<(usize, &str)> {
    const MARKER: &str = "inventory::submit!";
    let bytes = masked.as_bytes();
    let mut found = Vec::new();
    let mut at = 0usize;
    while let Some(offset) = masked[at..].find(MARKER) {
        let start = at + offset;
        at = start + MARKER.len();
        if start != 0 && bytes[start - 1] != b'\n' {
            continue;
        }
        let Some(open) = masked[start..].find('{').map(|index| start + index) else {
            continue;
        };
        let Some(close) = matching_brace(bytes, open) else {
            continue;
        };
        found.push((
            masked[..start].matches('\n').count() + 1,
            &masked[open..=close],
        ));
        at = close + 1;
    }
    found
}

/// Whether a submission block records laws or an explicit opaque decision.
///
/// The builder forms and the struct-literal form are both accepted, because the
/// dialect macro writes the fields directly. An empty `laws: &[]` is not a
/// decision: that field is what a registration carrying no laws already says,
/// and the macro pairs it with an `opaque_reason`.
fn records_decision(block: &str) -> bool {
    if block.contains("with_opaque(")
        || block.contains("with_no_transform(")
        || block.contains("with_laws(")
    {
        return true;
    }
    if let Some(rest) = block.split_once("opaque_reason:").map(|(_, rest)| rest) {
        if rest.trim_start().starts_with("Some") {
            return true;
        }
    }
    block
        .split_once("laws:")
        .and_then(|(_, rest)| rest.split_once(']'))
        .is_some_and(|(list, _)| list.contains('"'))
}

fn is_placeholder(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "todo"
            | "tbd"
            | "placeholder"
            | "none"
            | "unimplemented"
            | "opaque"
            | "no-op"
            | "no transform"
            | "not implemented"
    ) || lower.starts_with("todo:")
        || lower.starts_with("placeholder:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::operation::{OperationRegistration, OperationTier};

    #[test]
    fn an_operation_without_a_decision_is_reported() {
        static REG: OperationRegistration = OperationRegistration::new_unconstrained(
            "test::undecided_operation",
            OperationTier::Library,
            None,
            None,
            None,
        );
        let op = SemanticOperation::from(&REG);
        let findings = check_operations(&[op]);
        assert_eq!(findings.len(), 1);
        assert!(findings[0]
            .message
            .contains("neither an algebraic law nor an explicit opaque decision"));
    }

    #[test]
    fn an_operation_with_laws_is_clean() {
        static REG: OperationRegistration = OperationRegistration::new_unconstrained(
            "test::commutative_operation",
            OperationTier::Library,
            None,
            None,
            None,
        )
        .with_laws(&["commutative"]);
        let op = SemanticOperation::from(&REG);
        let findings = check_operations(&[op]);
        assert!(findings.is_empty(), "expected 0 findings, got {findings:?}");
    }

    #[test]
    fn an_operation_with_valid_opaque_reason_is_clean() {
        static REG: OperationRegistration = OperationRegistration::new_unconstrained(
            "test::opaque_operation",
            OperationTier::Library,
            None,
            None,
            None,
        )
        .with_opaque("cryptographic hash compression state step");
        let op = SemanticOperation::from(&REG);
        let findings = check_operations(&[op]);
        assert!(findings.is_empty(), "expected 0 findings, got {findings:?}");
    }

    #[test]
    fn an_operation_with_placeholder_opaque_reason_is_reported() {
        static REG: OperationRegistration = OperationRegistration::new_unconstrained(
            "test::placeholder_operation",
            OperationTier::Library,
            None,
            None,
            None,
        )
        .with_opaque("todo");
        let op = SemanticOperation::from(&REG);
        let findings = check_operations(&[op]);
        assert_eq!(findings.len(), 1);
        assert!(findings[0]
            .message
            .contains("invalid or placeholder opaque reason"));
    }
    /// WHY: this is the assertion the live-registry check cannot make. A
    /// registration in a test file reaches the registry only in the binary that
    /// compiles it, so the tree is the only place every one of them is visible
    /// at once. A finding here means a registration will panic registry
    /// validation in whichever binary links it first, and poison the registry
    /// for every later reader in that process.
    #[test]
    fn every_registration_in_the_tree_records_a_transform_decision() {
        let root = structure_gate::workspace_root();
        let (submissions, findings) = check_sources(&root);
        assert!(
            submissions > 100,
            "the source enumeration is broken: {submissions} registration(s) found"
        );
        assert!(findings.is_empty(), "{findings:?}");
    }

    /// WHY: the assertion above passes just as well against a scan that finds
    /// nothing, and the reason it once found nothing is that fixtures in this
    /// tree embed sample registrations inside raw strings. Both halves are
    /// pinned from one input: the undecided submission is reported, the decided
    /// ones are not, and the indented one is not a submission at all.
    #[test]
    fn the_scan_reports_an_undecided_submission_and_ignores_a_masked_fixture() {
        let source = concat!(
            "inventory::submit! {\n",
            "    OperationRegistration::new_unconstrained(A, T, None, None, None)\n",
            "}\n",
            "inventory::submit! {\n",
            "    OperationRegistration::new_unconstrained(B, T, None, None, None)\n",
            "        .with_opaque(\"an indexed read states no reorderable law\")\n",
            "}\n",
            "inventory::submit! {\n",
            "    OperationRegistration { laws: &[\"commutative\"], opaque_reason: None }\n",
            "}\n",
            "inventory::submit! {\n",
            "    OperationRegistration { laws: &[], opaque_reason: Some(\"macro states it\") }\n",
            "}\n",
            "    inventory::submit! {\n",
            "        OperationRegistration::new_unconstrained(C, T, None, None, None)\n",
            "    }\n",
        );
        let blocks = submission_blocks(source);
        assert_eq!(blocks.len(), 4, "column zero only: {blocks:?}");
        let undecided: Vec<usize> = blocks
            .iter()
            .filter(|(_, block)| !records_decision(block))
            .map(|(line, _)| *line)
            .collect();
        assert_eq!(undecided, vec![1], "only the first block states nothing");
    }

    /// WHY: an empty law list is what every registration carrying no laws
    /// already writes, so accepting it as a decision would clear the whole
    /// class and leave the gate certifying what it never checked.
    #[test]
    fn an_empty_law_list_is_not_a_decision() {
        assert!(!records_decision(
            "{ OperationRegistration { laws: &[], opaque_reason: None } }"
        ));
        assert!(records_decision(
            "{ OperationRegistration { laws: &[\"idempotent\"], opaque_reason: None } }"
        ));
    }
}
