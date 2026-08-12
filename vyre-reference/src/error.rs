use std::fmt;

/// Reference-interpreter failure with owner-local recovery guidance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceError {
    message: String,
    validation: Option<vyre_foundation::validate::ValidationError>,
}

impl ReferenceError {
    /// Build a reference-interpreter failure.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            validation: None,
        }
    }

    /// Preserve a foundation validation issue as owner-local context.
    #[must_use]
    pub fn validation(source: vyre_foundation::validate::ValidationError) -> Self {
        Self {
            message: source.to_string(),
            validation: Some(source),
        }
    }

    /// Return the structured validation source when validation rejected input.
    #[must_use]
    pub fn validation_source(&self) -> Option<&vyre_foundation::validate::ValidationError> {
        self.validation.as_ref()
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
