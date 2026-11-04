//! Program to canonical neutral artifact plus attached target payload.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, RetryClass,
};
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
            Self::TargetNotEnabled(target) => Diagnostic::error(
                "AOT001_TARGET_NOT_ENABLED",
                format!("target `{target}` has no linked target compiler"),
            )
            .with_stage(DiagnosticStage::Admit)
            .with_compiler_level(CompilerLevel::ToolingEvidence)
            .with_target(target.as_str())
            .with_fix("link the concrete driver crate that registers this target")
            .with_cause(
                CauseKind::Configuration,
                "unregistered_target",
                format!("target `{target}`"),
            )
            .with_retry(RetryClass::Never)
            .with_context_value("target", target.as_str()),
            Self::TargetCompilation(compile_err) => compile_err
                .diagnostic()
                .with_note("during AOT target compilation"),
            Self::CanonicalArtifact { stage, source } => source
                .diagnostic
                .clone()
                .with_note(format!("during AOT stage `{stage}`")),
        }
    }
}

/// Compile one validated compiler request through the canonical graph compiler
/// and a registered target facet.
///
/// The request is the caller's, whole: its graph, external facts, device facts,
/// objective and search budget reach the canonical compiler unchanged, and this
/// crate states none of them. An ahead-of-time compile and a direct one over
/// the same request therefore produce one artifact identity.
pub fn compile(
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
