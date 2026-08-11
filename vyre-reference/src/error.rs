use std::fmt;

/// Reference-interpreter failure with owner-local recovery guidance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceError {
    message: String,
}

impl ReferenceError {
    /// Build a reference-interpreter failure.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "vyre reference interpreter: {}", self.message)?;
        if !self.message.contains("Fix:") {
            let separator = if self.message.ends_with(['.', '!', '?']) {
                " "
            } else {
                ". "
            };
            write!(
                formatter,
                "{separator}Fix: validate the Program and input buffer set before invoking the reference backend."
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ReferenceError {}
