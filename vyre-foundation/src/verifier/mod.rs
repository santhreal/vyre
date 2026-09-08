//! Single declarative semantic verifier and proof certificate system (Row 104).
//!
//! Provides [`DeclarativeVerifier`], [`VerificationCertificate`], [`Verified`],
//! [`SemanticModule`], and [`SemanticCompiler`].

pub mod certificate;
pub mod compiler_gate;
pub mod declarative;
pub mod module;
pub mod verified;

pub use certificate::{CheckedInvariant, InvariantCategory, ResourceBounds, VerificationCertificate};
pub use compiler_gate::{CompileError, CompiledSemanticArtifact, SemanticCompiler};
pub use declarative::{DeclarativeVerifier, VerificationError};
pub use module::SemanticModule;
pub use verified::Verified;
