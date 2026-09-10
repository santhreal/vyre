//! Single declarative semantic verifier and proof certificate system.
//!
//! Provides [`DeclarativeVerifier`](crate::verifier::DeclarativeVerifier),
//! [`VerificationCertificate`](crate::verifier::VerificationCertificate),
//! [`Verified`](crate::verifier::Verified),
//! [`SemanticModule`](crate::verifier::SemanticModule), and
//! [`SemanticCompiler`](crate::verifier::SemanticCompiler).

pub(crate) mod certificate;
pub(crate) mod compiler_gate;
pub(crate) mod declarative;
pub(crate) mod module;
pub(crate) mod stages;
pub(crate) mod verified;

pub use certificate::{
    CheckedInvariant, InvariantCategory, ReplayError, ResourceBounds, VerificationCertificate,
};
pub use compiler_gate::{CompileError, CompiledSemanticArtifact, SemanticCompiler};
pub use declarative::{DeclarativeVerifier, VerificationError};
pub use module::SemanticModule;
pub use stages::{LoweredStage, OptimizedStage, StageKind, VerifiedIrStage, VerifiedStageWrapper};
pub use verified::Verified;
