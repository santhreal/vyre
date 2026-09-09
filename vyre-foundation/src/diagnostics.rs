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

/// Trait for types that can be projected into a structured [`Diagnostic`].
pub trait ToDiagnostic {
    /// Convert this error or event into a structured diagnostic record.
    fn to_diagnostic(&self) -> Diagnostic;
}

/// Implement the conversions an error type carries alongside `ToDiagnostic`.
///
/// A blanket `impl<E: ToDiagnostic> From<E> for Diagnostic` overlaps the
/// reflexive `From<T> for T`, so coherence rejects it and each error type states
/// the same three items. Eleven crates stated them by hand, and each copy was
/// free to project through a different method than the trait it also implements.
///
/// One argument names the error type and reuses its existing `ToDiagnostic`
/// implementation. A second argument names the inherent projection method, and
/// the macro implements the trait through it as well.
#[macro_export]
macro_rules! diagnostic_conversions {
    ($error:ty) => {
        impl From<&$error> for $crate::diagnostics::Diagnostic {
            fn from(error: &$error) -> Self {
                $crate::diagnostics::ToDiagnostic::to_diagnostic(error)
            }
        }

        impl From<$error> for $crate::diagnostics::Diagnostic {
            fn from(error: $error) -> Self {
                $crate::diagnostics::ToDiagnostic::to_diagnostic(&error)
            }
        }
    };
    ($error:ty, $project:ident) => {
        impl $crate::diagnostics::ToDiagnostic for $error {
            fn to_diagnostic(&self) -> $crate::diagnostics::Diagnostic {
                self.$project()
            }
        }

        $crate::diagnostic_conversions!($error);
    };
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

    /// Construct a code from validated owned data.
    #[must_use]
    pub fn from_owned(code: String) -> Self {
        Self(Cow::Owned(code))
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
    /// Optional field or structural path within an attribute or node.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub field_path: Option<Cow<'static, str>>,
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
            field_path: None,
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

    /// Attach a specific field path.
    #[must_use]
    pub fn with_field_path(mut self, path: impl Into<Cow<'static, str>>) -> Self {
        self.field_path = Some(path.into());
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

    /// Attach a source byte span.
    #[must_use]
    pub const fn with_source_span(mut self, start: u32, end: u32) -> Self {
        self.source_span = Some([start, end]);
        self
    }

    /// Attach a source byte span array.
    #[must_use]
    pub const fn with_span(mut self, span: [u32; 2]) -> Self {
        self.source_span = Some(span);
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
    /// Architectural compiler level when known.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub compiler_level: Option<CompilerLevel>,
    /// Deterministic failure detail.
    #[serde(deserialize_with = "deserialize_cow_static")]
    pub message: Cow<'static, str>,
    /// Typed source, graph, operation, or artifact location.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub location: Option<OpLocation>,
    /// Typed artifact identity or digest.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub artifact_id: Option<String>,
    /// Compilation target identifier where admissible.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub target: Option<String>,
    /// Target device identifier where admissible.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub device: Option<String>,
    /// Corrective action the caller can apply.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub suggested_fix: Option<Cow<'static, str>>,
    /// Primary structured cause retained from the owning stage.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cause: Option<DiagnosticCause>,
    /// Complete structured cause chain from root cause to boundary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cause_chain: Vec<DiagnosticCause>,
    /// Retry policy for this failure.
    pub retry: RetryClass,
    /// Bounded key-value contextual metadata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_values: Vec<(String, String)>,
    /// Optional stable documentation URL.
    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_optional_cow_static"
    )]
    pub doc_url: Option<Cow<'static, str>>,
    /// Additional contextual notes attached to this diagnostic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<Cow<'static, str>>,
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

    /// Construct an emission-stage error for one target, retryable by
    /// recompiling the source.
    ///
    /// Every emitter reports the same shape, and each one used to restate the
    /// sixteen fields of the struct per error variant. A field added to
    /// `Diagnostic` then reached whichever copies someone remembered.
    #[must_use]
    pub fn emission_error(
        target: impl Into<String>,
        code: &'static str,
        message: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::error(code, message)
            .with_stage(DiagnosticStage::Emit)
            .with_compiler_level(CompilerLevel::Emission)
            .with_target(target)
            .with_retry(RetryClass::RecompileSource)
    }

    fn new(severity: Severity, code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            severity,
            code: DiagnosticCode::new(code),
            stage: DiagnosticStage::Validate,
            compiler_level: None,
            message: message.into(),
            location: None,
            artifact_id: None,
            target: None,
            device: None,
            suggested_fix: None,
            cause: None,
            cause_chain: Vec::new(),
            retry: RetryClass::Never,
            context_values: Vec::new(),
            doc_url: None,
            notes: Vec::new(),
        }
    }

    /// Set the owning workflow stage.
    #[must_use]
    pub const fn with_stage(mut self, stage: DiagnosticStage) -> Self {
        self.stage = stage;
        self
    }

    /// Set the architectural compiler level.
    #[must_use]
    pub const fn with_compiler_level(mut self, level: CompilerLevel) -> Self {
        self.compiler_level = Some(level);
        self
    }

    /// Attach an artifact identifier.
    #[must_use]
    pub fn with_artifact_id(mut self, artifact_id: impl Into<String>) -> Self {
        self.artifact_id = Some(artifact_id.into());
        self
    }

    /// Attach a target identity.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Attach a device identity.
    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Attach a bounded contextual key-value pair.
    #[must_use]
    pub fn with_context_value(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.context_values.len() < 32 {
            let mut k = key.into();
            let mut v = value.into();
            if k.len() > 1024 {
                k.truncate(1024);
            }
            if v.len() > 1024 {
                v.truncate(1024);
            }
            self.context_values.push((k, v));
        }
        self
    }

    /// Attach multiple bounded contextual key-value pairs.
    #[must_use]
    pub fn with_context_values(
        mut self,
        values: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        for (k, v) in values {
            self = self.with_context_value(k, v);
        }
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
        let cause = DiagnosticCause {
            kind: kind.into(),
            detail: detail.into(),
        };
        self.cause = Some(cause.clone());
        self.cause_chain.push(cause);
        self
    }

    /// Attach a structured cause chain.
    #[must_use]
    pub fn with_cause_chain(
        mut self,
        chain: impl IntoIterator<Item = DiagnosticCause>,
    ) -> Self {
        for cause in chain {
            if self.cause.is_none() {
                self.cause = Some(cause.clone());
            }
            self.cause_chain.push(cause);
        }
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

    /// Attach an informational note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<Cow<'static, str>>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Attach multiple informational notes.
    #[must_use]
    pub fn with_notes(
        mut self,
        notes: impl IntoIterator<Item = impl Into<Cow<'static, str>>>,
    ) -> Self {
        self.notes.extend(notes.into_iter().map(Into::into));
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
        if let Some(level) = self.compiler_level {
            let _ = write!(output, "\n  = level: {}", level.label());
        }
        if let Some(target) = &self.target {
            let _ = write!(output, "\n  = target: {target}");
        }
        if let Some(device) = &self.device {
            let _ = write!(output, "\n  = device: {device}");
        }
        if let Some(artifact_id) = &self.artifact_id {
            let _ = write!(output, "\n  = artifact: {artifact_id}");
        }
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
            if let Some(field) = &location.field_path {
                output.push_str(" field `");
                output.push_str(field);
                output.push('`');
            }
            if let Some(path) = &location.path {
                output.push_str(" at ");
                output.push_str(path);
                if let Some([start, end]) = location.source_span {
                    let _ = write!(output, ":{start}..{end}");
                }
            } else if let Some([start, end]) = location.source_span {
                let _ = write!(output, " at span {start}..{end}");
            }
        }
        if let Some(fix) = &self.suggested_fix {
            output.push_str("\n  = help: ");
            output.push_str(fix);
        }
        if !self.cause_chain.is_empty() {
            for cause in &self.cause_chain {
                let _ = write!(output, "\n  = cause[{}]: {}", cause.kind, cause.detail);
            }
        } else if let Some(cause) = &self.cause {
            let _ = write!(output, "\n  = cause[{}]: {}", cause.kind, cause.detail);
        }
        for (k, v) in &self.context_values {
            let _ = write!(output, "\n  = context `{k}`: {v}");
        }
        if let Some(url) = &self.doc_url {
            output.push_str("\n  = note: ");
            output.push_str(url);
        }
        for note in &self.notes {
            output.push_str("\n  = note: ");
            output.push_str(note);
        }
        output
    }

    /// Serialize this diagnostic as canonical JSON.
    ///
    /// Every field is an owned string, integer, or enum, so `serde_json` has no
    /// failing path here: it fails only on a map with non-string keys, a
    /// non-finite float, or a `Serialize` impl that returns an error.
    ///
    /// # Panics
    ///
    /// Panics if serialization fails due to a custom `Serialize` implementation returning an error.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect(
            "a diagnostic holds only owned strings, integers, and enums. \
             Fix: a field added to Diagnostic serializes fallibly; make it data \
             or give it a Serialize impl that cannot fail",
        )
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str(&self.render_human())
    }
}
