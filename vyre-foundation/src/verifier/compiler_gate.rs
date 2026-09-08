//! Compiler entry gate requiring certified [`Verified<SemanticModule>`].
//!
//! Enforces the semantic assurance invariant: unverified syntax cannot reach compilation.

use thiserror::Error;
use super::certificate::VerificationCertificate;
use super::module::SemanticModule;
use super::verified::Verified;

/// Compilation error produced when attempting to compile invalid or unverified IR.
#[derive(Debug, Error)]
pub enum CompileError {
    /// Attempted compilation of an unverified module without certification.
    #[error("UnverifiedSyntaxRejected: module `{0}` has not been verified by DeclarativeVerifier. Fix: invoke DeclarativeVerifier::verify(&module) prior to compilation.")]
    UnverifiedSyntaxRejected(String),
    /// Module contains no executable program or entry points.
    #[error("EmptyModule: module `{0}` contains no executable program or entry points")]
    EmptyModule(String),
    /// Backend emission or lowering failure.
    #[error("LoweringError: {0}")]
    LoweringError(String),
}

/// Compiled semantic artifact certified by verification.
#[derive(Clone, Debug)]
pub struct CompiledSemanticArtifact {
    /// Module name.
    pub name: String,
    /// Copy of the verification certificate that authorized compilation.
    pub certificate: VerificationCertificate,
    /// Number of verified invariants proven prior to compilation.
    pub certified_invariant_count: usize,
}

/// Compilation entry point that strictly requires [`Verified<SemanticModule>`].
pub struct SemanticCompiler;

impl SemanticCompiler {
    /// Compile a certified semantic module into a compiled artifact.
    ///
    /// The function signature statically requires a [`Verified<SemanticModule>`],
    /// making it impossible for unverified syntax to reach compilation directly.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError::EmptyModule`] if the module has no executable content.
    pub fn compile(
        verified: &Verified<SemanticModule>,
    ) -> Result<CompiledSemanticArtifact, CompileError> {
        let module = verified.as_inner();
        let cert = verified.certificate();

        if module.program.is_none() && module.types.is_empty() && module.shape_constraints.is_empty() {
            return Err(CompileError::EmptyModule(module.name.clone()));
        }

        Ok(CompiledSemanticArtifact {
            name: module.name.clone(),
            certificate: cert.clone(),
            certified_invariant_count: cert.invariant_count(),
        })
    }

    /// Explicit unverified entry point that demonstrates rejection of uncertified syntax.
    ///
    /// # Errors
    ///
    /// Always returns [`CompileError::UnverifiedSyntaxRejected`] unless verified first.
    pub fn compile_unverified(
        module: &SemanticModule,
    ) -> Result<CompiledSemanticArtifact, CompileError> {
        Err(CompileError::UnverifiedSyntaxRejected(module.name.clone()))
    }
}
