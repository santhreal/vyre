//! Shared structured diagnostic protocol for compiler and workflow boundaries.

use std::borrow::Cow;
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer, Serialize};

fn deserialize_cow_static<'de, D>(deserializer: D) -> Result<Cow<'static, str>, D::Error>
where
    D: Deserializer<'de>,
{
    String::deserialize(deserializer).map(Cow::Owned)
}

fn deserialize_optional_cow_static<'de, D>(
    deserializer: D,
) -> Result<Option<Cow<'static, str>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(|value| value.map(Cow::Owned))
}

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

/// Stable, machine-readable diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DiagnosticCode(
    #[serde(deserialize_with = "deserialize_cow_static")] pub Cow<'static, str>,
);

impl DiagnosticCode {
    /// Construct a code from a stable static string.
    #[must_use]
    pub const fn new(code: &'static str) -> Self {
        Self(Cow::Borrowed(code))
    }

    /// Return the raw stable code.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DiagnosticCode {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str(&self.0)
    }
}

/// Typed location of a diagnostic inside source, graph, or artifact state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpLocation {
    /// Stable operation or pass identifier when available.
    #[serde(deserialize_with = "deserialize_cow_static")]
    pub op_id: Cow<'static, str>,
    /// Zero-based operand index.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub operand_idx: Option<u32>,
    /// Attribute name.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub attr_name: Option<Cow<'static, str>>,
    /// Typed graph node identity.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub graph_node: Option<u32>,
    /// Typed graph value identity.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub graph_value: Option<u32>,
    /// Canonical request, source, or artifact path.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub path: Option<String>,
    /// Byte span inside the source path.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source_span: Option<[u32; 2]>,
}

impl OpLocation {
    /// Build a location that identifies an operation or pass.
    #[must_use]
    pub fn op(op_id: impl Into<Cow<'static, str>>) -> Self {
        Self {
            op_id: op_id.into(),
            operand_idx: None,
            attr_name: None,
            graph_node: None,
            graph_value: None,
            path: None,
            source_span: None,
        }
    }

    /// Attach a specific operand index.
    #[must_use]
    pub fn with_operand(mut self, index: u32) -> Self {
        self.operand_idx = Some(index);
        self
    }

    /// Attach a specific attribute name.
    #[must_use]
    pub fn with_attr(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.attr_name = Some(name.into());
        self
    }

    /// Attach a typed graph node identity.
    #[must_use]
    pub const fn with_graph_node(mut self, node: u32) -> Self {
        self.graph_node = Some(node);
        self
    }

    /// Attach a typed graph value identity.
    #[must_use]
    pub const fn with_graph_value(mut self, value: u32) -> Self {
        self.graph_value = Some(value);
        self
    }

    /// Attach a canonical source or artifact path.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// Structured cause preserved across owner boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticCause {
    /// Stable cause family, such as `device_lost` or `version_skew`.
    pub kind: String,
    /// Deterministic cause detail.
    pub detail: String,
}

/// Serializable diagnostic shared by compiler, AOT, runtime, and drivers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Stable machine-readable code.
    pub code: DiagnosticCode,
    /// Stage that produced the diagnostic.
    pub stage: DiagnosticStage,
    /// Deterministic failure detail.
    #[serde(deserialize_with = "deserialize_cow_static")]
    pub message: Cow<'static, str>,
    /// Typed source, graph, operation, or artifact location.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub location: Option<OpLocation>,
    /// Corrective action the caller can apply.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub suggested_fix: Option<Cow<'static, str>>,
    /// Structured cause retained from the owning stage.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cause: Option<DiagnosticCause>,
    /// Retry policy for this failure.
    pub retry: RetryClass,
    /// Optional stable documentation URL.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub doc_url: Option<Cow<'static, str>>,
}

impl Diagnostic {
    /// Construct an error diagnostic at validation stage with no retry.
    #[must_use]
    pub fn error(code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Severity::Error, code, message)
    }

    /// Construct a warning diagnostic at validation stage with no retry.
    #[must_use]
    pub fn warning(code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Severity::Warning, code, message)
    }

    /// Construct a note diagnostic at validation stage with no retry.
    #[must_use]
    pub fn note(code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Severity::Note, code, message)
    }

    fn new(severity: Severity, code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            severity,
            code: DiagnosticCode::new(code),
            stage: DiagnosticStage::Validate,
            message: message.into(),
            location: None,
            suggested_fix: None,
            cause: None,
            retry: RetryClass::Never,
            doc_url: None,
        }
    }

    /// Set the owning workflow stage.
    #[must_use]
    pub const fn with_stage(mut self, stage: DiagnosticStage) -> Self {
        self.stage = stage;
        self
    }

    /// Attach a typed location.
    #[must_use]
    pub fn with_location(mut self, location: OpLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Attach a corrective action.
    #[must_use]
    pub fn with_fix(mut self, fix: impl Into<Cow<'static, str>>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Attach a structured cause.
    #[must_use]
    pub fn with_cause(mut self, kind: impl Into<String>, detail: impl Into<String>) -> Self {
        self.cause = Some(DiagnosticCause {
            kind: kind.into(),
            detail: detail.into(),
        });
        self
    }

    /// Set the retry policy.
    #[must_use]
    pub const fn with_retry(mut self, retry: RetryClass) -> Self {
        self.retry = retry;
        self
    }

    /// Attach a documentation URL.
    #[must_use]
    pub fn with_doc_url(mut self, url: impl Into<Cow<'static, str>>) -> Self {
        self.doc_url = Some(url.into());
        self
    }

    /// Render a deterministic rustc-style diagnostic.
    #[must_use]
    pub fn render_human(&self) -> String {
        let mut output = String::with_capacity(256);
        let _ = write!(
            output,
            "{}[{}]({:?}): {}",
            self.severity.label(),
            self.code,
            self.stage,
            self.message
        );
        if let Some(location) = &self.location {
            output.push_str("\n  --> op `");
            output.push_str(&location.op_id);
            output.push('`');
            if let Some(index) = location.operand_idx {
                let _ = write!(output, " operand[{index}]");
            }
            if let Some(attribute) = &location.attr_name {
                output.push_str(" attr `");
                output.push_str(attribute);
                output.push('`');
            }
            if let Some(path) = &location.path {
                output.push_str(" at ");
                output.push_str(path);
            }
        }
        if let Some(fix) = &self.suggested_fix {
            output.push_str("\n  = help: ");
            output.push_str(fix);
        }
        if let Some(cause) = &self.cause {
            let _ = write!(output, "\n  = cause[{}]: {}", cause.kind, cause.detail);
        }
        if let Some(url) = &self.doc_url {
            output.push_str("\n  = note: ");
            output.push_str(url);
        }
        output
    }

    /// Serialize this diagnostic as canonical JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Diagnostic serialization is infallible")
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str(&self.render_human())
    }
}
