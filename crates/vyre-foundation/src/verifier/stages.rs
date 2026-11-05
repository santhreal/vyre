//! Typed verified stage pipeline wrappers and lowering verifier seams.
//!
//! Enforces that stage transforms consume and return typed verified stages,
//! explicitly discharging or preserving obligations, without mutating cached
//! validation bits on the semantic value.

use super::certificate::VerificationCertificate;
use super::module::SemanticModule;
use super::verified::Verified;
use std::marker::PhantomData;

/// Marker trait for compilation and lowering pipeline stages.
pub trait StageKind: 'static + Send + Sync {
    /// Name of the pipeline stage.
    fn name() -> &'static str;
}

/// Initial verified IR stage directly from the declarative verifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VerifiedIrStage;
impl StageKind for VerifiedIrStage {
    fn name() -> &'static str {
        "VerifiedIr"
    }
}

/// Optimized semantic IR stage after transformation passes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OptimizedStage;
impl StageKind for OptimizedStage {
    fn name() -> &'static str {
        "Optimized"
    }
}

/// Lowered IR stage prepared for backend target emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LoweredStage;
impl StageKind for LoweredStage {
    fn name() -> &'static str {
        "Lowered"
    }
}

/// Strongly-typed verified stage wrapper holding a certified artifact.
#[derive(Clone, Debug)]
pub struct VerifiedStageWrapper<S: StageKind, T> {
    inner: T,
    certificate: VerificationCertificate,
    _stage: PhantomData<S>,
}

impl<S: StageKind, T> VerifiedStageWrapper<S, T> {
    /// Create a new typed stage wrapper with its certificate.
    #[must_use]
    pub fn new(inner: T, certificate: VerificationCertificate) -> Self {
        Self {
            inner,
            certificate,
            _stage: PhantomData,
        }
    }

    /// Access the verification certificate attached to this stage.
    #[must_use]
    pub fn certificate(&self) -> &VerificationCertificate {
        &self.certificate
    }

    /// Reference to the underlying stage value.
    #[must_use]
    pub fn as_inner(&self) -> &T {
        &self.inner
    }

    /// Consume this stage wrapper and extract the inner value.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Transition to a successor stage while preserving the certificate.
    #[must_use]
    pub fn transition_to<Next: StageKind, U>(
        self,
        transform: impl FnOnce(T) -> U,
    ) -> VerifiedStageWrapper<Next, U> {
        let new_inner = transform(self.inner);
        VerifiedStageWrapper {
            inner: new_inner,
            certificate: self.certificate,
            _stage: PhantomData,
        }
    }
}

impl From<Verified<SemanticModule>> for VerifiedStageWrapper<VerifiedIrStage, SemanticModule> {
    fn from(verified: Verified<SemanticModule>) -> Self {
        let cert = verified.certificate().clone();
        let module = verified.into_inner();
        Self::new(module, cert)
    }
}
