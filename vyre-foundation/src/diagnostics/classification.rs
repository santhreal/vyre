//! How a diagnostic is classified once it exists.
//!
//! Severity states how bad it is, compiler level states which stage of the
//! compiler owns it, stage states where in a compile it arose, and retry class
//! states whether repeating the operation can succeed. Four independent axes,
//! kept apart from the record that carries them.

use serde::{Deserialize, Serialize};

/// Severity of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Severity {
    /// A hard failure. The rejected product must not be used.
    Error,
    /// A soft failure attached to a usable product.
    Warning,
    /// Informational context attached to another diagnostic.
    Note,
}

impl Severity {
    /// Stable human-readable severity label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Note => "note",
        }
    }
}

/// Architectural compiler tier or level producing a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CompilerLevel {
    /// Frozen specification and schema level (Tier 0).
    Spec,
    /// Semantic IR and type system level (Tier 1).
    FoundationIr,
    /// Optimizer and pass engine level (Tier 2).
    Optimizer,
    /// LEGO primitive and dialect level (Tier 2.5 / 3).
    PrimitivesDialects,
    /// Target lowering level (Tier 4).
    Lowering,
    /// Backend emission and codegen level (Tier 5).
    Emission,
    /// Driver, materialization, and runtime execution level (Tier 6).
    DriverRuntime,
    /// Tooling, conformance, and evidence verification level.
    ToolingEvidence,
}

impl CompilerLevel {
    /// Stable human-readable compiler level label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Spec => "spec",
            Self::FoundationIr => "foundation_ir",
            Self::Optimizer => "optimizer",
            Self::PrimitivesDialects => "primitives_dialects",
            Self::Lowering => "lowering",
            Self::Emission => "emission",
            Self::DriverRuntime => "driver_runtime",
            Self::ToolingEvidence => "tooling_evidence",
        }
    }
}


/// Compiler or workflow stage that produced a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DiagnosticStage {
    /// Semantic or structural validation.
    Validate,
    /// Semantic optimization.
    Optimize,
    /// Whole-graph planning and selection.
    Plan,
    /// Verified descriptor lowering.
    Lower,
    /// Target payload emission.
    Emit,
    /// Artifact admission and authentication.
    Admit,
    /// Device-specific materialization.
    Materialize,
    /// Typed submission.
    Submit,
    /// Completion and readback.
    Complete,
}

impl DiagnosticStage {
    /// Stable serialized label, identical to the serde representation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Validate => "validate",
            Self::Optimize => "optimize",
            Self::Plan => "plan",
            Self::Lower => "lower",
            Self::Emit => "emit",
            Self::Admit => "admit",
            Self::Materialize => "materialize",
            Self::Submit => "submit",
            Self::Complete => "complete",
        }
    }
}

/// Whether and where a failed workflow may be retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RetryClass {
    /// Repeating the operation cannot succeed without changing its inputs.
    Never,
    /// Retry on the same device generation may succeed.
    SameDevice,
    /// Retry only after acquiring a new device generation.
    NewDevice,
    /// Recompile the source graph before retrying.
    RecompileSource,
}

impl RetryClass {
    /// Stable serialized label, identical to the serde representation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::SameDevice => "same_device",
            Self::NewDevice => "new_device",
            Self::RecompileSource => "recompile_source",
        }
    }
}
