//! Structured artifact-session recovery policy.

use vyre_driver::backend::ErrorCode;
use vyre_driver::{BackendError, DeviceIdentity};

use crate::artifact_admission::{ArtifactSession, ArtifactSessionError};

/// Stable runtime retry class derived from machine-readable backend errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryClass {
    /// The acquired device generation is invalid and requires rematerialization.
    DeviceLoss,
    /// Resource pressure may be retried without changing the artifact instance.
    TransientResource,
    /// Compilation, artifact, or capability failure cannot be retried unchanged.
    Permanent,
    /// The backend has not supplied a structured recovery class.
    Unclassified,
}

/// Classify a backend failure without parsing human-readable text.
#[must_use]
pub fn classify_backend_error(error: &BackendError) -> RecoveryClass {
    match error.code() {
        ErrorCode::DeviceLost => RecoveryClass::DeviceLoss,
        ErrorCode::DeviceOutOfMemory | ErrorCode::PoisonedLock => RecoveryClass::TransientResource,
        ErrorCode::UnsupportedFeature
        | ErrorCode::KernelCompileFailed
        | ErrorCode::InvalidProgram
        | ErrorCode::CooperativeResidencyExceeded => RecoveryClass::Permanent,
        ErrorCode::DispatchFailed | ErrorCode::Unknown => RecoveryClass::Unclassified,
        _ => RecoveryClass::Unclassified,
    }
}

/// Rematerialize an authenticated artifact only for a structured device-loss failure.
///
/// This function never invokes semantic optimization, lowering, or target compilation.
///
/// # Errors
///
/// Returns the original backend failure for any non-device-loss class. Returns the
/// admission or materialization failure when device reacquisition fails.
pub fn recover_artifact_session(
    session: &ArtifactSession,
    failure: BackendError,
) -> Result<DeviceIdentity, ArtifactSessionError> {
    if classify_backend_error(&failure) != RecoveryClass::DeviceLoss {
        return Err(failure.into());
    }
    session.rematerialize()
}
