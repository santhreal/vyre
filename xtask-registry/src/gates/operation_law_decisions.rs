//! `operation-law-decisions` - enforce every registered semantic operation carries algebraic laws or an explicit opaque decision.

use std::collections::BTreeSet;

use vyre_foundation::operation::SemanticOperation;
use xtask::gate::{Finding, GateCtx, GateError, Report};

/// Entry point for `cargo xtask operation-law-decisions`.
pub struct OperationLawDecisions;

impl xtask::gate::GateBehavior for OperationLawDecisions {
    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let registry = vyre_registry_link::operation::live_operation_registry();
        let mut report = Report::clean();
        let operations: Vec<SemanticOperation> = registry.iter().collect();
        report.cover_complete("registered operations", operations.len());

        let findings = check_operations(&operations);
        for finding in findings {
            report.find(finding);
        }

        report.note(format!(
            "{} registered semantic operation(s) checked for transform decisions",
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
}
