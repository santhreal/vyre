//! Program to canonical neutral artifact plus attached target payload.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CompilerLevel, Diagnostic, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
};
use vyre_foundation::IrError;
use vyre_megakernel::{
    ArtifactEnvelope, TargetCompileError, TargetCompiler, ValidatedCompileRequest,
};

use crate::artifact::{registration, TargetId};

/// Errors returned by [`compile`].
#[derive(Debug, Error)]
pub enum CompileError {
    /// The chosen target has no linked compiler facet.
    #[error(
        "vyre-aot: target `{0}` has no linked target compiler. Fix: link the concrete driver crate that registers this target."
    )]
    TargetNotEnabled(TargetId),

    /// Frontend call expansion failed.
    #[error("vyre-aot: frontend Program preparation failed: {0}")]
    ProgramPreparation(#[source] IrError),

    /// The Program cannot be represented accurately in the canonical graph.
    #[error("vyre-aot: artifact graph rejected Program: {0}")]
    ArtifactLayout(String),

    /// The selected target compiler rejected the canonical artifact.
    #[error("vyre-aot: target compiler rejected artifact: {0}")]
    TargetCompilation(#[source] TargetCompileError),
    /// Canonical artifact construction or payload association failed.
    #[error("vyre-aot: canonical artifact stage `{stage}` failed: {source}")]
    CanonicalArtifact {
        /// AOT stage that failed.
        stage: &'static str,
        /// Structured canonical artifact error, including its exact field path.
        #[source]
        source: vyre_megakernel::CompileError,
    },
}

impl CompileError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::TargetNotEnabled(target) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("AOT001_TARGET_NOT_ENABLED"),
                stage: DiagnosticStage::Admit,
                compiler_level: Some(CompilerLevel::ToolingEvidence),
                message: format!("target `{target}` has no linked target compiler").into(),
                location: None,
                artifact_id: None,
                target: Some(target.as_str().to_string()),
                device: None,
                suggested_fix: Some(
                    "link the concrete driver crate that registers this target".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unregistered_target".to_string(),
                    detail: format!("target `{target}`"),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unregistered_target".to_string(),
                    detail: format!("target `{target}`"),
                }],
                retry: RetryClass::Never,
                context_values: vec![("target".to_string(), target.as_str().to_string())],
                doc_url: None,
                notes: Vec::new(),
            },
            Self::ProgramPreparation(ir_err) => {
                let mut diag = ir_err.diagnostic();
                diag.notes
                    .push("during AOT frontend Program preparation".into());
                diag
            }
            Self::ArtifactLayout(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("AOT002_ARTIFACT_LAYOUT"),
                stage: DiagnosticStage::Plan,
                compiler_level: Some(CompilerLevel::Optimizer),
                message: format!("artifact graph rejected Program: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "ensure Program satisfies canonical graph invariants before AOT compilation"
                        .into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "artifact_layout".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "artifact_layout".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::TargetCompilation(compile_err) => {
                let mut diag = compile_err.diagnostic();
                diag.notes.push("during AOT target compilation".into());
                diag
            }
            Self::CanonicalArtifact { stage, source } => {
                let mut diag = source.diagnostic.clone();
                diag.notes
                    .push(format!("during AOT stage `{stage}`").into());
                diag
            }
        }
    }
}

/// Compile one validated compiler request through the canonical graph compiler and a registered target facet.
pub fn compile(
    request: &ValidatedCompileRequest,
    target: TargetId,
) -> Result<ArtifactEnvelope, CompileError> {
    compile_request(request, target)
}

/// Compile one validated compiler request through the canonical graph compiler and a registered target facet.
pub fn compile_request(
    request: &ValidatedCompileRequest,
    target: TargetId,
) -> Result<ArtifactEnvelope, CompileError> {
    let artifact =
        vyre_megakernel::compile(request).map_err(|source| CompileError::CanonicalArtifact {
            stage: "canonical-compile",
            source,
        })?;
    let compiler = registered_target_compiler(&target)?;
    vyre_megakernel::attach_target(artifact, compiler.as_ref())
        .map_err(CompileError::TargetCompilation)
}

fn registered_target_compiler(target: &TargetId) -> Result<Box<dyn TargetCompiler>, CompileError> {
    registration(target)
        .map_err(|_| CompileError::TargetNotEnabled(target.clone()))?
        .target_compiler()
        .map_err(|_| CompileError::TargetNotEnabled(target.clone()))
}
vyre_foundation::diagnostic_conversions!(CompileError, diagnostic);
