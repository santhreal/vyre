//! Structured artifact-session recovery policy.

use vyre_driver::ErrorCode;
use vyre_driver::{BackendError, DeviceIdentity};
use vyre_foundation::diagnostics::RetryClass;

use crate::artifact_admission::{ArtifactSession, ArtifactSessionError};

/// Classify a backend failure into the shared workflow retry protocol.

/// Classify a backend failure without parsing human-readable text.
#[must_use]
pub fn classify_backend_error(error: &BackendError) -> RetryClass {
    match error.code() {
        ErrorCode::DeviceLost => RetryClass::NewDevice,
        ErrorCode::DeviceOutOfMemory | ErrorCode::PoisonedLock => RetryClass::SameDevice,
        ErrorCode::UnsupportedFeature
        | ErrorCode::KernelCompileFailed
        | ErrorCode::InvalidProgram
        | ErrorCode::CooperativeResidencyExceeded
        | ErrorCode::DispatchFailed
        | ErrorCode::Unknown => RetryClass::Never,
        _ => RetryClass::Never,
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
    if classify_backend_error(&failure) != RetryClass::NewDevice {
        return Err(failure.into());
    }
    session.rematerialize()
}
