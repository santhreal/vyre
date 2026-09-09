//! Strongly-typed verified artifact wrapper.
//!
//! [`Verified<T>`] can only be constructed by the declarative verifier.
//! Compilation and backend lowering entry points require a `&Verified<SemanticModule>`
//! to ensure unverified syntax can never reach compilation.

use std::ops::Deref;
use super::certificate::{ReplayError, VerificationCertificate};
use super::module::SemanticModule;
/// Type-safe proof wrapper witnessing that `T` has passed semantic verification.
///
/// Cannot be constructed directly: must be obtained via
/// [`DeclarativeVerifier::verify`](super::DeclarativeVerifier::verify).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Verified<T> {
    pub(crate) inner: T,
    pub(crate) certificate: VerificationCertificate,
}

impl<T> Verified<T> {
    /// Internal constructor used only by the declarative verifier.
    pub(crate) fn new_certified(inner: T, certificate: VerificationCertificate) -> Self {
        Self { inner, certificate }
    }

    /// Access the verification certificate attached to this artifact.
    #[must_use]
    pub fn certificate(&self) -> &VerificationCertificate {
        &self.certificate
    }

    /// Reference to the underlying semantic data.
    #[must_use]
    pub fn as_inner(&self) -> &T {
        &self.inner
    }

    /// Consume the wrapper and extract the underlying verified value.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }
}
impl Verified<SemanticModule> {
    /// Independent lightweight proof replay of this verified module's certificate.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] if the certificate is invalid or has been tampered with.
    pub fn replay_proof(&self) -> Result<(), ReplayError> {
        self.certificate.replay_proof(&self.inner)
    }
}


impl<T> Deref for Verified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
