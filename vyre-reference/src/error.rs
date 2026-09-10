use std::fmt;
use vyre_foundation::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, RetryClass, ToDiagnostic,
};

/// The eight closed failure classes returned by the strict reference oracle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ReferenceErrorClass {
    /// Absent input buffer, unassigned variable, or missing callee.
    MissingValue,
    /// Ill-typed argument, operand, or invalid type conversion.
    TypeMismatch,
    /// Poisoned synchronization primitive or memory lock.
    Poison,
    /// Arithmetic overflow in dimensions, strides, or indexing.
    Overflow,
    /// Out-of-bounds load, store, or atomic memory access.
    OutOfBoundsAccess,
    /// Unsupported dispatch grid, non-uniform barrier, or non-WORLD collective in single-rank.
    IncompleteDispatchSemantics,
    /// Infinite loop or non-terminating program execution.
    Nontermination,
    /// Work step ceiling, memory limit, or recursion depth budget exhausted.
    BudgetExhaustion,
}

impl ReferenceErrorClass {
    /// Every failure class, in declaration order.
    ///
    /// Derived from [`Self::successor`], whose match has no catch-all arm, so
    /// a new class does not compile until it is placed in the chain and the
    /// array length is corrected.
    pub const ALL: [Self; Self::COUNT] = {
        let mut classes = [Self::MissingValue; Self::COUNT];
        let mut index = 1;
        while index < Self::COUNT {
            match classes[index - 1].successor() {
                Some(next) => classes[index] = next,
                None => panic!(
                    "Fix: ReferenceErrorClass::COUNT exceeds the successor chain; correct COUNT."
                ),
            }
            index += 1;
        }
        match classes[Self::COUNT - 1].successor() {
            Some(_) => panic!(
                "Fix: a ReferenceErrorClass is missing from ALL; raise COUNT to the chain length."
            ),
            None => classes,
        }
    };

    /// Number of failure classes.
    const COUNT: usize = 8;

    /// The class declared after this one, or `None` for the last.
    ///
    /// The match has no catch-all arm, so adding a variant is a build failure
    /// here rather than a silently short [`Self::ALL`].
    #[must_use]
    const fn successor(self) -> Option<Self> {
        match self {
            Self::MissingValue => Some(Self::TypeMismatch),
            Self::TypeMismatch => Some(Self::Poison),
            Self::Poison => Some(Self::Overflow),
            Self::Overflow => Some(Self::OutOfBoundsAccess),
            Self::OutOfBoundsAccess => Some(Self::IncompleteDispatchSemantics),
            Self::IncompleteDispatchSemantics => Some(Self::Nontermination),
            Self::Nontermination => Some(Self::BudgetExhaustion),
            Self::BudgetExhaustion => None,
        }
    }

    /// Stable identifier for this error class.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MissingValue => "missing_value",
            Self::TypeMismatch => "type_mismatch",
            Self::Poison => "poison",
            Self::Overflow => "overflow",
            Self::OutOfBoundsAccess => "out_of_bounds_access",
            Self::IncompleteDispatchSemantics => "incomplete_dispatch_semantics",
            Self::Nontermination => "nontermination",
            Self::BudgetExhaustion => "budget_exhaustion",
        }
    }

    /// Recovery class a caller routes on for this failure class.
    ///
    /// The match has no catch-all arm, so a new reference failure class is a
    /// recorded decision rather than a silent reuse of an existing class.
    #[must_use]
    pub const fn cause_kind(self) -> CauseKind {
        match self {
            Self::MissingValue => CauseKind::Configuration,
            Self::TypeMismatch => CauseKind::InvalidInput,
            Self::Poison => CauseKind::InternalInvariant,
            Self::Overflow => CauseKind::NumericOverflow,
            Self::OutOfBoundsAccess => CauseKind::InvalidInput,
            Self::IncompleteDispatchSemantics => CauseKind::UnsupportedCapability,
            Self::Nontermination => CauseKind::Timeout,
            Self::BudgetExhaustion => CauseKind::ResourceExhausted,
        }
    }
}

/// Structured failure payload for one of the eight reference oracle failure classes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceErrorKind {
    /// Absent input buffer, unassigned variable, or missing callee.
    MissingValue {
        /// Detail message.
        detail: String,
    },
    /// Ill-typed argument, operand, or invalid type conversion.
    TypeMismatch {
        /// Detail message.
        detail: String,
    },
    /// Poisoned synchronization primitive or memory lock.
    Poison {
        /// Detail message.
        detail: String,
    },
    /// Arithmetic overflow in dimensions, strides, or indexing.
    Overflow {
        /// Detail message.
        detail: String,
    },
    /// Out-of-bounds load, store, or atomic memory access.
    OutOfBoundsAccess {
        /// Detail message.
        detail: String,
    },
    /// Unsupported dispatch grid, non-uniform barrier, or non-WORLD collective in single-rank.
    IncompleteDispatchSemantics {
        /// Detail message.
        detail: String,
    },
    /// Infinite loop or non-terminating program execution.
    Nontermination {
        /// Detail message.
        detail: String,
    },
    /// Work step ceiling, memory limit, or recursion depth budget exhausted.
    BudgetExhaustion {
        /// Detail message.
        detail: String,
    },
}

impl ReferenceErrorKind {
    /// Return the error class for this kind.
    #[must_use]
    pub const fn error_class(&self) -> ReferenceErrorClass {
        match self {
            Self::MissingValue { .. } => ReferenceErrorClass::MissingValue,
            Self::TypeMismatch { .. } => ReferenceErrorClass::TypeMismatch,
            Self::Poison { .. } => ReferenceErrorClass::Poison,
            Self::Overflow { .. } => ReferenceErrorClass::Overflow,
            Self::OutOfBoundsAccess { .. } => ReferenceErrorClass::OutOfBoundsAccess,
            Self::IncompleteDispatchSemantics { .. } => {
                ReferenceErrorClass::IncompleteDispatchSemantics
            }
            Self::Nontermination { .. } => ReferenceErrorClass::Nontermination,
            Self::BudgetExhaustion { .. } => ReferenceErrorClass::BudgetExhaustion,
        }
    }

    /// Return the detail message string.
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Self::MissingValue { detail }
            | Self::TypeMismatch { detail }
            | Self::Poison { detail }
            | Self::Overflow { detail }
            | Self::OutOfBoundsAccess { detail }
            | Self::IncompleteDispatchSemantics { detail }
            | Self::Nontermination { detail }
            | Self::BudgetExhaustion { detail } => detail.as_str(),
        }
    }
}

fn classify_message(msg: &str) -> ReferenceErrorKind {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("out of bounds")
        || lower.contains("out of range")
        || lower.contains("oob")
        || lower.contains("past buffer")
    {
        ReferenceErrorKind::OutOfBoundsAccess {
            detail: msg.to_string(),
        }
    } else if lower.contains("missing")
        || lower.contains("not found")
        || lower.contains("unassigned")
    {
        ReferenceErrorKind::MissingValue {
            detail: msg.to_string(),
        }
    } else if lower.contains("mismatch")
        || lower.contains("cannot be represented")
        || lower.contains("invalid type")
        || lower.contains("must be u32")
        || lower.contains("type")
    {
        ReferenceErrorKind::TypeMismatch {
            detail: msg.to_string(),
        }
    } else if lower.contains("poison") {
        ReferenceErrorKind::Poison {
            detail: msg.to_string(),
        }
    } else if lower.contains("overflow") {
        ReferenceErrorKind::Overflow {
            detail: msg.to_string(),
        }
    } else if lower.contains("nontermination") || lower.contains("infinite loop") {
        ReferenceErrorKind::Nontermination {
            detail: msg.to_string(),
        }
    } else if lower.contains("step")
        || lower.contains("ceiling")
        || lower.contains("budget")
        || lower.contains("exceeded")
    {
        ReferenceErrorKind::BudgetExhaustion {
            detail: msg.to_string(),
        }
    } else {
        ReferenceErrorKind::IncompleteDispatchSemantics {
            detail: msg.to_string(),
        }
    }
}

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
    kind: ReferenceErrorKind,
    validation: Option<vyre_foundation::validate::ValidationError>,
    step_ceiling: Option<StepCeilingExceeded>,
}

impl ReferenceError {
    /// Build a reference-interpreter failure with classified kind.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        let msg = message.into();
        let kind = classify_message(&msg);
        Self {
            kind,
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build a MissingValue error.
    #[must_use]
    pub fn missing_value(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::MissingValue { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build a TypeMismatch error.
    #[must_use]
    pub fn type_mismatch(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::TypeMismatch { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build a Poison error.
    #[must_use]
    pub fn poison(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::Poison { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build an Overflow error.
    #[must_use]
    pub fn overflow(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::Overflow { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build an OutOfBoundsAccess error.
    #[must_use]
    pub fn out_of_bounds(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::OutOfBoundsAccess { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build an IncompleteDispatchSemantics error.
    #[must_use]
    pub fn incomplete_dispatch_semantics(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::IncompleteDispatchSemantics { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build a Nontermination error.
    #[must_use]
    pub fn nontermination(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::Nontermination { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Build a BudgetExhaustion error.
    #[must_use]
    pub fn budget_exhaustion(detail: impl Into<String>) -> Self {
        let msg = detail.into();
        Self {
            kind: ReferenceErrorKind::BudgetExhaustion { detail: msg },
            validation: None,
            step_ceiling: None,
        }
    }

    /// Return the error class.
    #[must_use]
    pub fn error_class(&self) -> ReferenceErrorClass {
        self.kind.error_class()
    }

    /// Return the structured error kind.
    #[must_use]
    pub fn kind(&self) -> &ReferenceErrorKind {
        &self.kind
    }

    /// Return the error detail message.
    #[must_use]
    pub fn message(&self) -> &str {
        self.kind.detail()
    }

    /// Preserve a foundation validation issue as owner-local context.
    #[must_use]
    pub fn validation(source: vyre_foundation::validate::ValidationError) -> Self {
        let message = source.to_string();
        Self {
            kind: classify_message(&message),
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
            kind: ReferenceErrorKind::BudgetExhaustion { detail: message },
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
        write!(
            formatter,
            "vyre reference interpreter: {}",
            self.kind.detail()
        )
    }
}

impl std::error::Error for ReferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.validation
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl ToDiagnostic for ReferenceError {
    fn to_diagnostic(&self) -> Diagnostic {
        let class = self.kind.error_class();
        if let Some(validation) = &self.validation {
            return validation.to_diagnostic().with_cause(
                class.cause_kind(),
                "reference_validation_failure",
                self.kind.detail().to_string(),
            );
        }
        let retry = if self.step_ceiling.is_some() {
            RetryClass::RecompileSource
        } else {
            RetryClass::Never
        };
        let mut diagnostic = Diagnostic::error("REF001_REFERENCE_ERROR", self.kind.detail().to_string())
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_cause(
                class.cause_kind(),
                "reference_execution_error",
                self.kind.detail().to_string(),
            )
            .with_retry(retry)
            .with_context_value("reference_error_class", class.name());
        if let Some(ceiling) = &self.step_ceiling {
            diagnostic = diagnostic
                .with_fix(
                    "bound program trip counts by a declared extent or evaluate a smaller input",
                )
                .with_context_value("program", ceiling.program.clone())
                .with_context_value("ceiling", ceiling.ceiling.to_string());
        }
        diagnostic
    }
}

vyre_foundation::diagnostic_conversions!(ReferenceError);
