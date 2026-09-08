use std::fmt;

/// The work one reference evaluation was allowed and what it exceeded.
///
/// A caller waiting on the parity oracle reads the ceiling and the program that
/// reached it, rather than matching a message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepCeilingExceeded {
    /// The program that reached the ceiling: its entry operation id when it
    /// declares one, otherwise its fingerprint prefix.
    pub program: String,
    /// Steps the interpreter admits for one evaluation.
    pub ceiling: u64,
}

/// Reference-interpreter failure with owner-local recovery guidance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceError {
    message: String,
    validation: Option<vyre_foundation::validate::ValidationError>,
    step_ceiling: Option<StepCeilingExceeded>,
}

impl ReferenceError {
    /// Build a reference-interpreter failure.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            validation: None,
            step_ceiling: None,
        }
    }

    /// Preserve a foundation validation issue as owner-local context.
    #[must_use]
    pub fn validation(source: vyre_foundation::validate::ValidationError) -> Self {
        Self {
            message: source.to_string(),
            validation: Some(source),
            step_ceiling: None,
        }
    }

    /// Report a program that exceeded the interpreter's work ceiling.
    #[must_use]
    pub fn step_ceiling(source: StepCeilingExceeded) -> Self {
        let message = format!(
            "program `{}` executed more than {} interpreter steps. Fix: bound the program's trip counts by a declared extent rather than by data, or evaluate a smaller input; the reference interpreter is a termination-bounded oracle, not an unbounded evaluator.",
            source.program, source.ceiling
        );
        Self {
            message,
            validation: None,
            step_ceiling: Some(source),
        }
    }

    /// Return the structured validation source when validation rejected input.
    #[must_use]
    pub fn validation_source(&self) -> Option<&vyre_foundation::validate::ValidationError> {
        self.validation.as_ref()
    }

    /// Return the work ceiling this failure exceeded, when it exceeded one.
    #[must_use]
    pub fn step_ceiling_source(&self) -> Option<&StepCeilingExceeded> {
        self.step_ceiling.as_ref()
    }
}

impl From<vyre_foundation::validate::ValidationError> for ReferenceError {
    fn from(source: vyre_foundation::validate::ValidationError) -> Self {
        Self::validation(source)
    }
}

impl fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "vyre reference interpreter: {}", self.message)
    }
}

impl std::error::Error for ReferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.validation
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}
