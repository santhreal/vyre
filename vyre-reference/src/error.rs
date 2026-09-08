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

/// Operation kind that attempted an out-of-bounds access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutOfBoundsOp {
    /// Buffer element or slice read.
    Load,
    /// Buffer element or slice write.
    Store,
    /// Atomic read operation.
    AtomicLoad,
    /// Atomic write operation.
    AtomicStore,
}

impl fmt::Display for OutOfBoundsOp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load => write!(formatter, "load"),
            Self::Store => write!(formatter, "store"),
            Self::AtomicLoad => write!(formatter, "atomic load"),
            Self::AtomicStore => write!(formatter, "atomic store"),
        }
    }
}

/// Structured record of an out-of-bounds buffer access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutOfBoundsAccess {
    /// Name of the accessed buffer.
    pub buffer: String,
    /// Requested element or byte index.
    pub index: u64,
    /// Declared buffer extent (length in elements or bytes).
    pub extent: u64,
    /// Operation attempted.
    pub operation: OutOfBoundsOp,
}

/// Reference-interpreter failure with owner-local recovery guidance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceError {
    message: String,
    validation: Option<vyre_foundation::validate::ValidationError>,
    step_ceiling: Option<StepCeilingExceeded>,
    out_of_bounds: Option<OutOfBoundsAccess>,
}

impl ReferenceError {
    /// Build a reference-interpreter failure.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            validation: None,
            step_ceiling: None,
            out_of_bounds: None,
        }
    }

    /// Preserve a foundation validation issue as owner-local context.
    #[must_use]
    pub fn validation(source: vyre_foundation::validate::ValidationError) -> Self {
        Self {
            message: source.to_string(),
            validation: Some(source),
            step_ceiling: None,
            out_of_bounds: None,
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
            out_of_bounds: None,
        }
    }

    /// Report an out-of-bounds buffer access with exact buffer, index, and extent.
    #[must_use]
    pub fn out_of_bounds(
        buffer: impl Into<String>,
        index: u64,
        extent: u64,
        operation: OutOfBoundsOp,
    ) -> Self {
        let buffer = buffer.into();
        let message = format!(
            "out-of-bounds {operation} on buffer `{buffer}` at index {index} with extent {extent}. Fix: ensure buffer access is within declared bounds [0, {extent})."
        );
        Self {
            message,
            validation: None,
            step_ceiling: None,
            out_of_bounds: Some(OutOfBoundsAccess {
                buffer,
                index,
                extent,
                operation,
            }),
        }
    }

    /// Report an out-of-bounds load on `buffer` at `index` exceeding `extent`.
    #[must_use]
    pub fn out_of_bounds_load(buffer: impl Into<String>, index: u64, extent: u64) -> Self {
        Self::out_of_bounds(buffer, index, extent, OutOfBoundsOp::Load)
    }

    /// Report an out-of-bounds store on `buffer` at `index` exceeding `extent`.
    #[must_use]
    pub fn out_of_bounds_store(buffer: impl Into<String>, index: u64, extent: u64) -> Self {
        Self::out_of_bounds(buffer, index, extent, OutOfBoundsOp::Store)
    }

    /// Return the structured validation source when validation rejected input.
    #[must_use]
    pub fn validation_source(&self) -> Option<&vyre_foundation::validate::ValidationError> {
        self.validation.as_ref()
    }

    /// Return the structured step ceiling source when execution exceeded budget.
    #[must_use]
    pub fn step_ceiling_source(&self) -> Option<&StepCeilingExceeded> {
        self.step_ceiling.as_ref()
    }

    /// Return the structured out-of-bounds record if an out-of-bounds access occurred.
    #[must_use]
    pub fn out_of_bounds_source(&self) -> Option<&OutOfBoundsAccess> {
        self.out_of_bounds.as_ref()
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
