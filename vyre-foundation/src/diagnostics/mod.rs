//! The one structured diagnostic record every reporting surface renders.
//!
//! A caller classifies a failure from typed fields: the stable code, the
//! severity, the owning stage and compiler level, the typed location, the
//! target and device identity, the recovery class of every preserved cause, and
//! the retry class. Rendered text is a projection of that record and never an
//! input to a decision.
//!
//! The record is versioned by [`SchemaId::DiagnosticRecord`] in the canonical
//! schema registry, which also states its size and element bounds. A decoder
//! rejects a record whose version it does not implement rather than reading it
//! partially.
//!
//! Every string a record carries is bounded by a stated rule, so two renderings
//! of one failure are byte-identical and a hostile input cannot grow the record
//! past its declared limit.

mod cause;

use std::borrow::Cow;
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer, Serialize};
use vyre_spec::schema_registry::SchemaId;

/// The cause types, published at this one path so a caller cites one name.
pub use cause::{CauseKind, DiagnosticCause};

/// Schema identity of the diagnostic record.
pub const DIAGNOSTIC_SCHEMA_ID: SchemaId = SchemaId::DiagnosticRecord;

/// Version of the diagnostic record this build produces and accepts.
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = DIAGNOSTIC_SCHEMA_ID.version_u32();

/// Signature domain separator for a diagnostic record digest.
pub const DIAGNOSTIC_DOMAIN_SEPARATOR: &str = DIAGNOSTIC_SCHEMA_ID.domain_separator();

/// Maximum serialized size of one record, from the canonical schema registry.
pub const RECORD_MAX_BYTES: usize = DIAGNOSTIC_SCHEMA_ID.definition().bounds.max_bytes;

/// Maximum number of contextual key-value pairs, from the schema registry.
pub const CONTEXT_MAX_PAIRS: usize = DIAGNOSTIC_SCHEMA_ID.definition().bounds.max_elements;

/// Maximum bytes of one contextual key.
pub const CONTEXT_KEY_MAX_BYTES: usize = 64;

/// Maximum bytes of one contextual value.
pub const CONTEXT_VALUE_MAX_BYTES: usize = 256;

/// Maximum bytes of the primary failure message.
pub const MESSAGE_MAX_BYTES: usize = 4096;

/// Maximum bytes of one cause detail.
pub const CAUSE_DETAIL_MAX_BYTES: usize = 512;

/// Maximum number of links a cause chain retains.
pub const CAUSE_CHAIN_MAX_LINKS: usize = 8;

/// Maximum bytes of the corrective action.
pub const FIX_MAX_BYTES: usize = 1024;

/// Maximum number of notes a record retains.
pub const NOTES_MAX: usize = 8;

/// Maximum bytes of one note.
pub const NOTE_MAX_BYTES: usize = 512;

/// Suffix appended to a value the bound truncated.
pub const TRUNCATION_MARKER: &str = "...";

/// Replacement written wherever redaction removes host-identifying material.
pub const REDACTION_PLACEHOLDER: &str = "<redacted>";

/// Contextual keys whose values name the host rather than the failure.
///
/// Redaction replaces the value under one of these keys outright, because the
/// key alone already states what the value would have said.
pub const REDACTED_CONTEXT_KEYS: &[&str] = &[
    "home",
    "host",
    "hostname",
    "source_path",
    "token",
    "user",
    "username",
    "workspace_root",
];

/// Bound one value to `max_bytes`, cutting on a character boundary.
///
/// A value longer than the bound keeps the longest prefix that leaves room for
/// [`TRUNCATION_MARKER`] and ends on a character boundary, then carries the
/// marker. The result is valid UTF-8 and never exceeds `max_bytes`, so a
/// multi-byte character straddling the bound cannot panic the truncation and
/// cannot push the record past its declared size.
#[must_use]
pub fn bound_value(mut value: String, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value;
    }
    let budget = max_bytes.saturating_sub(TRUNCATION_MARKER.len());
    let mut cut = budget;
    while cut > 0 && !value.is_char_boundary(cut) {
        cut -= 1;
    }
    value.truncate(cut);
    value.push_str(TRUNCATION_MARKER);
    value
}

pub(crate) fn deserialize_cow_static<'de, D>(deserializer: D) -> Result<Cow<'static, str>, D::Error>
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

const fn current_schema_version() -> u32 {
    DIAGNOSTIC_SCHEMA_VERSION
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

/// Stable, machine-readable diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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

    fn redacted(&self) -> Self {
        let mut copy = self.clone();
        copy.path = copy.path.map(|path| redact_paths(&path));
        copy
    }
}

/// Why a serialized diagnostic record was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DiagnosticDecodeError {
    /// The encoded record is larger than the schema registry permits.
    #[error(
        "diagnostic record is {got} bytes, limit is {limit}. Fix: bound the producing fields before encoding"
    )]
    TooLarge {
        /// Encoded length.
        got: usize,
        /// Declared limit.
        limit: usize,
    },
    /// The record declares a schema version this build does not implement.
    #[error(
        "diagnostic record declares schema version {got}, this build implements {expected}. Fix: re-render the record with the current compiler"
    )]
    VersionSkew {
        /// Version the record declares.
        got: u32,
        /// Version this build implements.
        expected: u32,
    },
    /// The bytes are not a diagnostic record.
    #[error("diagnostic record is malformed: {detail}. Fix: re-render the record")]
    Malformed {
        /// Deterministic decode detail.
        detail: String,
    },
}

/// Serializable diagnostic shared by compiler, AOT, runtime, and drivers.
///
/// Construct through [`Diagnostic::error`], [`Diagnostic::warning`],
/// [`Diagnostic::note`], or [`Diagnostic::emission_error`] and the `with_*`
/// builders. The type is `#[non_exhaustive]` so a field added here reaches every
/// producer through the builders instead of the subset of struct literals
/// someone remembered to update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Diagnostic {
    /// Record schema version this diagnostic was produced under.
    #[serde(default = "current_schema_version")]
    pub schema_version: u32,
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
    /// Complete structured cause chain from the boundary to the root cause.
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

    /// Construct an error diagnostic from an owned code.
    ///
    /// Used by a registry whose codes are data rather than literals.
    #[must_use]
    pub fn error_with_code(code: DiagnosticCode, message: impl Into<Cow<'static, str>>) -> Self {
        let mut diagnostic = Self::new(Severity::Error, "", message);
        diagnostic.code = code;
        diagnostic
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
            schema_version: DIAGNOSTIC_SCHEMA_VERSION,
            severity,
            code: DiagnosticCode::new(code),
            stage: DiagnosticStage::Validate,
            compiler_level: None,
            message: bound_cow(message.into(), MESSAGE_MAX_BYTES),
            location: None,
            artifact_id: None,
            target: None,
            device: None,
            suggested_fix: None,
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

    /// Set the severity.
    #[must_use]
    pub const fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Attach an artifact identifier.
    #[must_use]
    pub fn with_artifact_id(mut self, artifact_id: impl Into<String>) -> Self {
        self.artifact_id = Some(bound_value(artifact_id.into(), CONTEXT_VALUE_MAX_BYTES));
        self
    }

    /// Attach a target identity.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(bound_value(target.into(), CONTEXT_VALUE_MAX_BYTES));
        self
    }

    /// Attach a device identity.
    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(bound_value(device.into(), CONTEXT_VALUE_MAX_BYTES));
        self
    }

    /// Attach a bounded contextual key-value pair.
    ///
    /// The pair is dropped once [`CONTEXT_MAX_PAIRS`] pairs are present. The key
    /// is bound to [`CONTEXT_KEY_MAX_BYTES`] and the value to
    /// [`CONTEXT_VALUE_MAX_BYTES`], each cut on a character boundary.
    #[must_use]
    pub fn with_context_value(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.context_values.len() < CONTEXT_MAX_PAIRS {
            self.context_values.push((
                bound_value(key.into(), CONTEXT_KEY_MAX_BYTES),
                bound_value(value.into(), CONTEXT_VALUE_MAX_BYTES),
            ));
        }
        self
    }

    /// Attach multiple bounded contextual key-value pairs.
    #[must_use]
    pub fn with_context_values(
        mut self,
        values: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        for (key, value) in values {
            self = self.with_context_value(key, value);
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
        self.suggested_fix = Some(bound_cow(fix.into(), FIX_MAX_BYTES));
        self
    }

    /// Append one classified link to the cause chain.
    ///
    /// The chain is ordered from the boundary that reported the failure to the
    /// root cause, and stops growing at [`CAUSE_CHAIN_MAX_LINKS`].
    #[must_use]
    pub fn with_cause(
        self,
        kind: CauseKind,
        subject: impl Into<Cow<'static, str>>,
        detail: impl Into<String>,
    ) -> Self {
        self.with_cause_link(DiagnosticCause::new(kind, subject, detail))
    }

    /// Append one already-built cause link.
    #[must_use]
    pub fn with_cause_link(mut self, cause: DiagnosticCause) -> Self {
        if self.cause_chain.len() < CAUSE_CHAIN_MAX_LINKS {
            self.cause_chain.push(cause);
        }
        self
    }

    /// Append every link of another chain, preserving order.
    #[must_use]
    pub fn with_cause_chain(mut self, chain: impl IntoIterator<Item = DiagnosticCause>) -> Self {
        for cause in chain {
            self = self.with_cause_link(cause);
        }
        self
    }

    /// Adopt `source` as the cause of this diagnostic.
    ///
    /// The source's own classified chain is appended after this diagnostic's
    /// links, so a chain that crosses three owners still reads root-ward from
    /// the boundary that reported it. This is how a crate preserves a cause it
    /// cannot name the type of: the owning crate projects its error into a
    /// record, and every crate above it keeps every link.
    #[must_use]
    pub fn caused_by(mut self, source: &Self) -> Self {
        let boundary = DiagnosticCause {
            kind: source
                .cause_chain
                .first()
                .map_or(CauseKind::InternalInvariant, |first| first.kind),
            subject: Cow::Owned(source.code.as_str().to_owned()),
            detail: bound_value(source.message.to_string(), CAUSE_DETAIL_MAX_BYTES),
        };
        self = self.with_cause_link(boundary);
        for link in &source.cause_chain {
            self = self.with_cause_link(link.clone());
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
        if self.notes.len() < NOTES_MAX {
            self.notes.push(bound_cow(note.into(), NOTE_MAX_BYTES));
        }
        self
    }

    /// Attach multiple informational notes.
    #[must_use]
    pub fn with_notes(
        mut self,
        notes: impl IntoIterator<Item = impl Into<Cow<'static, str>>>,
    ) -> Self {
        for note in notes {
            self = self.with_note(note);
        }
        self
    }

    /// Primary structured cause, the link closest to the reporting boundary.
    #[must_use]
    pub fn cause(&self) -> Option<&DiagnosticCause> {
        self.cause_chain.first()
    }

    /// Root structured cause, the deepest preserved link.
    #[must_use]
    pub fn root_cause(&self) -> Option<&DiagnosticCause> {
        self.cause_chain.last()
    }

    /// Whether any preserved link carries `kind`.
    ///
    /// This is the typed replacement for searching rendered text.
    #[must_use]
    pub fn caused_by_kind(&self, kind: CauseKind) -> bool {
        self.cause_chain.iter().any(|cause| cause.kind == kind)
    }

    /// Render a deterministic rustc-style diagnostic.
    #[must_use]
    pub fn render_human(&self) -> String {
        let mut output = String::with_capacity(256);
        let _ = write!(
            output,
            "{}[{}]({}): {}",
            self.severity.label(),
            self.code,
            self.stage.label(),
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
        for cause in &self.cause_chain {
            let _ = write!(
                output,
                "\n  = cause[{}/{}]: {}",
                cause.kind.label(),
                cause.subject,
                cause.detail
            );
        }
        let _ = write!(output, "\n  = retry: {}", self.retry.label());
        for (key, value) in &self.context_values {
            let _ = write!(output, "\n  = context `{key}`: {value}");
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

    /// Canonical serialized bytes of this record.
    ///
    /// Field order is the declaration order of the struct and every collection
    /// keeps insertion order, so two encodings of one record are byte-identical
    /// in any process. This is the exact byte sequence every surface renders,
    /// stores, and digests.
    ///
    /// # Panics
    ///
    /// Panics if serialization fails, which requires a field whose `Serialize`
    /// implementation returns an error. Every field is an owned string, integer,
    /// or enum.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect(
            "a diagnostic holds only owned strings, integers, and enums. \
             Fix: a field added to Diagnostic serializes fallibly; make it data \
             or give it a Serialize impl that cannot fail",
        )
    }

    /// Serialize this diagnostic as canonical JSON.
    ///
    /// # Panics
    ///
    /// Panics under the same condition as [`Self::canonical_bytes`].
    #[must_use]
    pub fn to_json(&self) -> String {
        String::from_utf8(self.canonical_bytes())
            .expect("serde_json emits UTF-8. Fix: none; this cannot fail")
    }

    /// Stable identity of this record, for certificates and cache keys.
    ///
    /// The digest covers the schema domain separator followed by the canonical
    /// bytes, so a record cannot be confused with another schema's payload and a
    /// version bump changes the identity.
    #[must_use]
    pub fn record_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DIAGNOSTIC_DOMAIN_SEPARATOR.as_bytes());
        hasher.update(b"\0");
        hasher.update(&self.canonical_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Lowercase hexadecimal [`Self::record_digest`].
    #[must_use]
    pub fn record_digest_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.record_digest() {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    /// Decode a record, refusing a stale version or an oversized payload.
    ///
    /// A record that declares a version this build does not implement is
    /// rejected rather than read partially, because a partially read record
    /// silently loses whichever fields the newer version added.
    ///
    /// # Errors
    ///
    /// Returns [`DiagnosticDecodeError`] when the payload exceeds
    /// [`RECORD_MAX_BYTES`], declares another schema version, or is malformed.
    pub fn from_json(encoded: &str) -> Result<Self, DiagnosticDecodeError> {
        if encoded.len() > RECORD_MAX_BYTES {
            return Err(DiagnosticDecodeError::TooLarge {
                got: encoded.len(),
                limit: RECORD_MAX_BYTES,
            });
        }
        let decoded: Self =
            serde_json::from_str(encoded).map_err(|error| DiagnosticDecodeError::Malformed {
                detail: bound_value(error.to_string(), CAUSE_DETAIL_MAX_BYTES),
            })?;
        if decoded.schema_version != DIAGNOSTIC_SCHEMA_VERSION {
            return Err(DiagnosticDecodeError::VersionSkew {
                got: decoded.schema_version,
                expected: DIAGNOSTIC_SCHEMA_VERSION,
            });
        }
        Ok(decoded)
    }

    /// A copy with host-identifying material removed.
    ///
    /// Redaction is a pure function of the record: an absolute filesystem path
    /// keeps only its final component behind [`REDACTION_PLACEHOLDER`], and a
    /// contextual value under a key in [`REDACTED_CONTEXT_KEYS`] is replaced
    /// outright. Nothing else changes, so a redacted record still classifies and
    /// still digests to one stable identity.
    #[must_use]
    pub fn redacted(&self) -> Self {
        let mut copy = self.clone();
        copy.message = Cow::Owned(redact_paths(&self.message));
        copy.location = self.location.as_ref().map(OpLocation::redacted);
        copy.suggested_fix = self
            .suggested_fix
            .as_ref()
            .map(|fix| Cow::Owned(redact_paths(fix)));
        for cause in &mut copy.cause_chain {
            cause.detail = redact_paths(&cause.detail);
        }
        for (key, value) in &mut copy.context_values {
            if REDACTED_CONTEXT_KEYS.contains(&key.as_str()) {
                *value = REDACTION_PLACEHOLDER.to_owned();
            } else {
                *value = redact_paths(value);
            }
        }
        copy.notes = self
            .notes
            .iter()
            .map(|note| Cow::Owned(redact_paths(note)))
            .collect();
        copy
    }
}

fn bound_cow(value: Cow<'static, str>, max_bytes: usize) -> Cow<'static, str> {
    if value.len() <= max_bytes {
        return value;
    }
    Cow::Owned(bound_value(value.into_owned(), max_bytes))
}

/// Replace every absolute filesystem path in `text` with its final component.
///
/// A token is an absolute path when it starts with `/` and holds a second `/`,
/// or when its second and third bytes are `:\` or `:/`. Both forms keep the
/// final component so the diagnostic still names the file, and lose the
/// directories above it, which are the part that names the host.
fn redact_paths(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut first = true;
    for token in text.split(' ') {
        if !first {
            out.push(' ');
        }
        first = false;
        match absolute_path_tail(token) {
            Some(tail) => {
                out.push_str(REDACTION_PLACEHOLDER);
                out.push('/');
                out.push_str(tail);
            }
            None => out.push_str(token),
        }
    }
    out
}

fn absolute_path_tail(token: &str) -> Option<&str> {
    let trimmed = token.trim_matches(|c| c == '`' || c == '"' || c == ',' || c == '.');
    let bytes = trimmed.as_bytes();
    let unix = bytes.first() == Some(&b'/') && trimmed.matches('/').count() >= 2;
    let windows = bytes.len() > 2
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    if !unix && !windows {
        return None;
    }
    trimmed
        .rsplit(|c| c == '/' || c == '\\')
        .find(|part| !part.is_empty())
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str(&self.render_human())
    }
}
