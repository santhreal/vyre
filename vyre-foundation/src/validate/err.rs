pub use super::depth::{DEFAULT_MAX_CALL_DEPTH, DEFAULT_MAX_NESTING_DEPTH, DEFAULT_MAX_NODE_COUNT};
use super::{ValidationCode, ValidationError, ValidationLocation, ValidationPhase};
use std::borrow::Cow;

#[inline]
pub(crate) fn issue(
    code: ValidationCode,
    phase: ValidationPhase,
    location: ValidationLocation,
    cause: impl Into<Cow<'static, str>>,
    corrective_action: impl Into<Cow<'static, str>>,
) -> ValidationError {
    ValidationError::new(code, phase, location, cause, corrective_action)
}

#[inline]
pub(crate) fn err(
    code: &'static str,
    phase: ValidationPhase,
    location: ValidationLocation,
    cause: impl Into<Cow<'static, str>>,
    corrective_action: impl Into<Cow<'static, str>>,
) -> ValidationError {
    issue(
        ValidationCode::new(code),
        phase,
        location,
        cause,
        corrective_action,
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_preserves_typed_fields() {
        let issue = issue(
            ValidationCode::new("V028"),
            ValidationPhase::Type,
            ValidationLocation::Expression { node: 2, depth: 1 },
            "wrong type",
            "cast the operand",
        );
        assert_eq!(issue.code().as_str(), "V028");
        assert_eq!(issue.phase(), ValidationPhase::Type);
        assert_eq!(issue.corrective_action(), "cast the operand");
    }

    const _: () = assert!(DEFAULT_MAX_CALL_DEPTH > 0);
    const _: () = assert!(DEFAULT_MAX_NESTING_DEPTH > 0);
    const _: () = assert!(DEFAULT_MAX_NODE_COUNT > 0);
}
