use crate::diagnostics::Diagnostic;

use super::ValidationError;

/// Full result of a validation run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    /// Hard validation failures that reject the Program.
    pub errors: Vec<ValidationError>,
    /// Non-fatal diagnostics emitted during validation.
    pub warnings: Vec<ValidationWarning>,
    /// Ordered trace of every issue emitted at the shared validation choke point.
    pub trace: Vec<super::ValidationTraceEvent>,
}

impl ValidationReport {
    /// Return true when the report contains no hard validation failures.
    #[must_use]
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Non-fatal structured validation diagnostic.
pub type ValidationWarning = Diagnostic;

#[inline]
pub(crate) fn warn(
    code: &'static str,
    location: super::ValidationLocation,
    message: impl Into<std::borrow::Cow<'static, str>>,
    fix: impl Into<std::borrow::Cow<'static, str>>,
) -> ValidationWarning {
    let validation_code = super::ValidationCode::new(code);
    assert_eq!(
        validation_code.phase(),
        Some(super::ValidationPhase::Type),
        "validation warning {validation_code} emitted from the wrong phase"
    );
    Diagnostic::warning(code, message)
        .with_location(location.diagnostic_location())
        .with_fix(fix)
        .with_cause(
            super::ValidationPhase::Type.as_str(),
            "non-fatal validation rule",
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_ok() {
        let report = ValidationReport::default();
        assert_eq!(report.errors.len(), 0);
    }

    #[test]
    fn report_with_error_is_not_ok() {
        let mut report = ValidationReport::default();
        report.errors.push(ValidationError::new(
            crate::validate::ValidationCode::new("V105"),
            crate::validate::ValidationPhase::Program,
            crate::validate::ValidationLocation::Program,
            "test error",
            "repair test input",
        ));
        assert!(!report.is_ok());
    }

    #[test]
    fn trace_has_one_event_per_emitted_issue() {
        use crate::ir::{BufferAccess, BufferDecl, DataType, Program};

        let program = Program::wrapped(
            vec![
                BufferDecl::output("out", 0, DataType::U32),
                BufferDecl::storage("duplicate", 0, BufferAccess::ReadWrite, DataType::U32),
            ],
            [0, 1, 1],
            Vec::new(),
        );
        let report = crate::validate::validate_with_options(
            &program,
            crate::validate::ValidationOptions::default(),
        );

        assert_eq!(report.trace.len(), report.errors.len());
        for (event, issue) in report.trace.iter().zip(&report.errors) {
            assert_eq!(event, &issue.trace_event());
        }
        let unique = report
            .trace
            .iter()
            .map(|event| (&event.code, &event.location))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            unique.len(),
            report.trace.len(),
            "one rule may emit at most once per typed location"
        );
    }

    #[test]
    fn warn_builds_warning() {
        let warning = warn(
            "V035",
            crate::validate::ValidationLocation::Program,
            "narrowing cast",
            "use a wider target",
        );
        assert_eq!(warning.message, "narrowing cast");
    }

    #[test]
    fn warning_clone_and_eq() {
        let a = warn(
            "V035",
            crate::validate::ValidationLocation::Program,
            "test",
            "use a wider target",
        );
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn report_clone_and_eq() {
        let a = ValidationReport::default();
        let b = a.clone();
        assert_eq!(a, b);
    }
}
