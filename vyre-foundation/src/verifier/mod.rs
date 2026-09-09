//! Single declarative semantic verifier and proof certificate system (Row 104).
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
pub(crate) mod verified;

pub use certificate::{CheckedInvariant, InvariantCategory, ResourceBounds, VerificationCertificate};
pub use compiler_gate::{CompileError, CompiledSemanticArtifact, SemanticCompiler};
pub use declarative::{DeclarativeVerifier, VerificationError};
pub use module::SemanticModule;
pub use verified::Verified;
