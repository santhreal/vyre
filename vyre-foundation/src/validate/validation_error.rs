//! Structured validation issues for vyre IR programs.

use core::fmt;
use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticStage, OpLocation, RetryClass, Severity,
};
use super::catalog::{ValidationRule, VALIDATION_RULES};

/// Stable validation rule identity.
///
/// Codes are explicit at every emission site. New validator rules use a new
/// identity rather than encoding ownership in prose.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ValidationCode(Cow<'static, str>);

impl<'de> Deserialize<'de> for ValidationCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let code = String::deserialize(deserializer)?;
        let validation_code = Self(Cow::Owned(code));
        if validation_code.phase().is_none() {
            return Err(serde::de::Error::custom(format!(
                "unknown validation code `{validation_code}`"
            )));
        }
        Ok(validation_code)
    }
}

impl ValidationCode {
    /// Backend capability rejected an operation used by the program.
    pub const V056: Self = Self(Cow::Borrowed("V056"));

    /// Construct a stable rule identity.
    #[must_use]
    pub(crate) const fn new(code: &'static str) -> Self {
        Self(Cow::Borrowed(code))
    }

    /// Return the stable code spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Iterate every registered validation rule and its sole owning phase.
    ///
    /// The registry is the source for diagnostics tooling and for the
    /// generated catalog. A rule must appear in
    /// [`crate::validate::catalog::VALIDATION_RULES`] before a code naming it
    /// can deserialize.
    pub fn registered() -> impl ExactSizeIterator<Item = (&'static str, ValidationPhase)> + Clone {
        VALIDATION_RULES.iter().map(|rule| (rule.code, rule.phase))
    }

    /// Return the sole validator phase allowed to emit this rule.
    #[must_use]
    pub fn phase(&self) -> Option<ValidationPhase> {
        self.rule().map(|rule| rule.phase)
    }

    /// Return the invariant this rule enforces.
    #[must_use]
    pub fn invariant(&self) -> Option<&'static str> {
        self.rule().map(|rule| rule.invariant)
    }

    /// Return the correction this rule offers to the program's author.
    ///
    /// This is the rule-level correction. An emitted [`ValidationError`]
    /// carries a corrective action naming the offending buffer or binding.
    #[must_use]
    pub fn corrective_action(&self) -> Option<&'static str> {
        self.rule().map(|rule| rule.corrective_action)
    }

    fn rule(&self) -> Option<&'static ValidationRule> {
        VALIDATION_RULES
            .iter()
            .find(|rule| rule.code == self.as_str())
    }
}

impl fmt::Display for ValidationCode {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str(&self.0)
    }
}

/// Validator phase that owns a rule emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ValidationPhase {
    /// Program header, workgroup, and buffer declarations.
    Program,
    /// Node structure, scope, and control flow.
    Node,
    /// Expression structure and call validation.
    Expression,
    /// Static type rules.
    Type,
    /// Memory access and ordering rules.
    Memory,
    /// Backend capability-sensitive validation.
    Capability,
    /// Whole-program composition and fusion rules.
    Composition,
    /// Resource and recursion bounds.
    Limits,
}

impl ValidationPhase {
    /// Stable phase spelling used by structured causes and traces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Program => "program",
            Self::Node => "node",
            Self::Expression => "expression",
            Self::Type => "type",
            Self::Memory => "memory",
            Self::Capability => "capability",
            Self::Composition => "composition",
            Self::Limits => "limits",
        }
    }
}

/// Typed location within the validated program.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ValidationLocation {
    /// The complete program.
    Program,
    /// One workgroup axis.
    WorkgroupAxis(u8),
    /// One declared buffer.
    Buffer(Cow<'static, str>),
    /// One node in pre-order traversal.
    Node(u32),
    /// One expression owned by a node.
    Expression {
        /// Pre-order node identity.
        node: u32,
        /// Expression depth below the node.
        depth: u32,
    },
    /// One operand of an expression or call.
    Operand {
        /// Pre-order node identity.
        node: u32,
        /// Zero-based operand index.
        operand: u32,
    },
    /// One issue within a deterministic validator traversal.
    Traversal {
        /// Zero-based node or issue order within the owning validation phase.
        ordinal: u64,
    },
    /// One registered semantic operation.
    Operation(Cow<'static, str>),
}

impl ValidationLocation {
    pub(crate) fn diagnostic_location(&self) -> OpLocation {
        match self {
            Self::Program => OpLocation::op("program"),
            Self::WorkgroupAxis(axis) => {
                OpLocation::op("program.workgroup_size").with_operand(u32::from(*axis))
            }
            Self::Buffer(name) => OpLocation::op("program.buffer").with_attr(name.clone()),
            Self::Node(node) => OpLocation::op("program.node").with_graph_node(*node),
            Self::Expression { node, depth } => OpLocation::op("program.expression")
                .with_graph_node(*node)
                .with_operand(*depth),
            Self::Operand { node, operand } => OpLocation::op("program.expression")
                .with_graph_node(*node)
                .with_operand(*operand),
            Self::Traversal { ordinal } => OpLocation::op("program.validation")
                .with_graph_node(u32::try_from(*ordinal).unwrap_or(u32::MAX)),
            Self::Operation(op_id) => OpLocation::op(op_id.clone()),
        }
    }
}

/// One trace record produced at the shared validation issue choke point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationTraceEvent {
    /// Emitted rule identity.
    pub code: ValidationCode,
    /// Rule-owning validator phase.
    pub phase: ValidationPhase,
    /// Typed program location.
    pub location: ValidationLocation,
}

/// A structured validation issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationError {
    /// Stable validation rule identity.
    code: ValidationCode,
    /// Rule-owning validator phase.
    phase: ValidationPhase,
    /// Typed program location.
    location: ValidationLocation,
    /// Deterministic cause detail without a code or corrective-action prefix.
    cause: Cow<'static, str>,
    /// Corrective action.
    corrective_action: Cow<'static, str>,
    /// Retry policy.
    retry: RetryClass,
}

#[derive(Deserialize)]
struct ValidationErrorWire {
    code: ValidationCode,
    phase: ValidationPhase,
    location: ValidationLocation,
    cause: Cow<'static, str>,
    corrective_action: Cow<'static, str>,
    retry: RetryClass,
}

impl<'de> Deserialize<'de> for ValidationError {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ValidationErrorWire::deserialize(deserializer)?;
        if wire.code.phase() != Some(wire.phase) {
            return Err(serde::de::Error::custom(format!(
                "validation rule {} belongs to phase {:?}, not {:?}",
                wire.code,
                wire.code.phase(),
                wire.phase
            )));
        }
        if wire.retry != RetryClass::Never {
            return Err(serde::de::Error::custom(format!(
                "validation rule {} has invalid retry class {:?}",
                wire.code, wire.retry
            )));
        }
        Ok(Self {
            code: wire.code,
            phase: wire.phase,
            location: wire.location,
            cause: wire.cause,
            corrective_action: wire.corrective_action,
            retry: wire.retry,
        })
    }
}

impl ValidationError {
    /// Construct one validation issue at the shared choke point.
    #[must_use]
    pub(crate) fn new(
        code: ValidationCode,
        phase: ValidationPhase,
        location: ValidationLocation,
        cause: impl Into<Cow<'static, str>>,
        corrective_action: impl Into<Cow<'static, str>>,
    ) -> Self {
        assert_eq!(
            code.phase(),
            Some(phase),
            "validation rule {code} emitted from the wrong phase"
        );
        Self {
            code,
            phase,
            location,
            cause: cause.into(),
            corrective_action: corrective_action.into(),
            retry: RetryClass::Never,
        }
    }

    /// Build an unsupported-operation diagnostic for backend capability checks.
    #[must_use]
    pub fn unsupported_op(backend: &'static str, op_id: &Arc<str>, node_index: usize) -> Self {
        Self::new(
            ValidationCode::V056,
            ValidationPhase::Capability,
            ValidationLocation::Operation(Cow::Owned(op_id.to_string())),
            format!(
                "backend `{backend}` does not support operation `{op_id}` at node {node_index}"
            ),
            format!(
                "choose a backend whose capability set includes this operation, lower the program through a supported backend pipeline, or register an implementation for `{op_id}`"
            ),
        )
    }

    /// Stable rule identity.
    #[must_use]
    pub fn code(&self) -> &ValidationCode {
        &self.code
    }

    /// Rule-owning validation phase.
    #[must_use]
    pub const fn phase(&self) -> ValidationPhase {
        self.phase
    }

    /// Typed program location.
    #[must_use]
    pub const fn location(&self) -> &ValidationLocation {
        &self.location
    }

    /// Deterministic cause detail.
    #[must_use]
    pub fn cause(&self) -> &str {
        &self.cause
    }

    /// Corrective action.
    #[must_use]
    pub fn corrective_action(&self) -> &str {
        &self.corrective_action
    }

    /// Retry policy.
    #[must_use]
    pub const fn retry(&self) -> RetryClass {
        self.retry
    }

    pub(crate) fn set_location(&mut self, location: ValidationLocation) {
        self.location = location;
    }

    /// Render the stable human-readable issue detail.
    #[must_use]
    pub fn message(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}: {}. Fix: {}",
            self.code, self.cause, self.corrective_action
        ))
    }

    /// Return the trace event for this emission.
    #[must_use]
    pub fn trace_event(&self) -> ValidationTraceEvent {
        ValidationTraceEvent {
            code: self.code.clone(),
            phase: self.phase,
            location: self.location.clone(),
        }
    }

    /// Project the issue into the shared diagnostic protocol.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        Diagnostic {
            severity: Severity::Error,
            code: DiagnosticCode::from_owned(self.code.as_str().to_string()),
            stage: DiagnosticStage::Validate,
            message: self.cause.clone(),
            location: Some(self.location.diagnostic_location()),
            suggested_fix: Some(self.corrective_action.clone()),
            cause: Some(crate::diagnostics::DiagnosticCause {
                kind: self.phase.as_str().to_string(),
                detail: self.cause.to_string(),
            }),
            retry: self.retry,
            doc_url: Some(Cow::Owned(format!(
                "https://docs.vyre.dev/validator-errors#{}",
                self.code.as_str().to_ascii_lowercase()
            ))),
        }
    }
}

impl From<&ValidationError> for Diagnostic {
    fn from(issue: &ValidationError) -> Self {
        issue.diagnostic()
    }
}

impl From<ValidationError> for Diagnostic {
    fn from(issue: ValidationError) -> Self {
        issue.diagnostic()
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "vyre IR validation: {}", self.message())
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn issue() -> ValidationError {
        ValidationError::new(
            ValidationCode::new("V028"),
            ValidationPhase::Type,
            ValidationLocation::Operand {
                node: 7,
                operand: 1,
            },
            "Fma operand has type i32, expected f32",
            "cast the operand to f32",
        )
    }

    #[test]
    fn every_validation_family_is_enforced_at_the_shared_choke_point() {
        let cases = [
            ("V105", ValidationPhase::Program),
            ("V112", ValidationPhase::Node),
            ("V012", ValidationPhase::Expression),
            ("V084", ValidationPhase::Type),
            ("V057", ValidationPhase::Memory),
            ("V056", ValidationPhase::Capability),
            ("V115", ValidationPhase::Composition),
            ("V018", ValidationPhase::Limits),
        ];
        for (code, phase) in cases {
            let issue = ValidationError::new(
                ValidationCode::new(code),
                phase,
                ValidationLocation::Program,
                "family mutation",
                "restore the owning phase",
            );
            assert_eq!(issue.code.phase(), Some(phase));
        }
    }

    #[test]
    #[should_panic(expected = "emitted from the wrong phase")]
    fn phase_mutation_fails_at_the_shared_choke_point() {
        let _ = ValidationError::new(
            ValidationCode::new("V105"),
            ValidationPhase::Node,
            ValidationLocation::Program,
            "mutated rule owner",
            "restore the program phase",
        );
    }

    #[test]
    fn typed_issue_projects_without_parsing_prose() {
        let issue = issue();
        assert_eq!(issue.code().as_str(), "V028");
        assert_eq!(
            issue.message(),
            "V028: Fma operand has type i32, expected f32. Fix: cast the operand to f32"
        );
        assert_eq!(issue.trace_event().phase, ValidationPhase::Type);

        let diagnostic = issue.diagnostic();
        assert_eq!(diagnostic.code.as_str(), "V028");
        assert_eq!(diagnostic.stage, DiagnosticStage::Validate);
        assert_eq!(diagnostic.retry, RetryClass::Never);
        assert_eq!(
            diagnostic
                .location
                .as_ref()
                .and_then(|location| location.graph_node),
            Some(7)
        );
        assert_eq!(
            diagnostic.suggested_fix.as_deref(),
            Some("cast the operand to f32")
        );
        assert_eq!(
            diagnostic.cause.as_ref().map(|cause| cause.kind.as_str()),
            Some("type")
        );
    }

    #[test]
    fn serialization_preserves_every_issue_field() {
        let issue = issue();
        let encoded = serde_json::to_vec(&issue).expect("validation issue must serialize");
        let decoded: ValidationError =
            serde_json::from_slice(&encoded).expect("validation issue must deserialize");
        assert_eq!(decoded, issue);
        assert_eq!(decoded.diagnostic(), issue.diagnostic());
    }

    #[test]
    fn deserialization_rejects_unknown_rule_identity() {
        let encoded = serde_json::to_value(issue()).expect("issue must serialize");
        let mut mutated = encoded;
        mutated["code"] = serde_json::Value::String(format!("V{}", 999));
        let error = serde_json::from_value::<ValidationError>(mutated)
            .expect_err("unknown validation rule must fail closed");
        assert!(error.to_string().contains("unknown validation code"));
    }

    #[test]
    fn deserialization_rejects_phase_mutation() {
        let encoded = serde_json::to_value(issue()).expect("issue must serialize");
        let mut mutated = encoded;
        mutated["phase"] = serde_json::Value::String("node".to_string());
        let error = serde_json::from_value::<ValidationError>(mutated)
            .expect_err("phase mutation must fail closed");
        assert!(error.to_string().contains("belongs to phase"));
    }

    #[test]
    fn unsupported_op_has_typed_capability_identity() {
        let issue = ValidationError::unsupported_op("backend-a", &Arc::from("math::fma"), 3);
        assert_eq!(issue.code().as_str(), "V056");
        assert_eq!(issue.phase(), ValidationPhase::Capability);
        assert!(issue.message().contains("backend-a"));
        assert!(issue.message().contains("math::fma"));
        assert!(issue.message().contains("3"));
        assert!(issue.message().contains("Fix:"));
    }
}
