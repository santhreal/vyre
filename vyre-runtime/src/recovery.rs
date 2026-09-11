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
        // An aborted request is dead: the caller submits the next generation
        // rather than retrying this one.
        | ErrorCode::ExecutionAborted
        | ErrorCode::Unknown => RetryClass::Never,
        // `ErrorCode` is non-exhaustive, so the compiler cannot force a
        // decision here. `backend_error_classification_exhaustive_closure` in
        // `vyre-runtime/tests/session_state_machine_contracts.rs` walks
        // `ErrorCode::ALL` and fails until a new code is listed above.
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
    let expected_artifact = session.artifact()?;
    let identity = session.rematerialize()?;
    let recovered_artifact = session.artifact()?;
    if recovered_artifact != expected_artifact {
        return Err(ArtifactSessionError::State(format!(
            "Fix: recovered session serves mismatched artifact identity: expected `{expected_artifact}`, actual `{recovered_artifact}`"
        )));
    }
    Ok(identity)
}

/// Recover an artifact session while validating against a caller-specified expected artifact identity.
///
/// # Errors
///
/// Returns [`ArtifactSessionError::State`] refusing by name if the session before or after
/// recovery disagrees with `expected_artifact`.
pub fn recover_session_with_expected_identity(
    session: &ArtifactSession,
    expected_artifact: &vyre_megakernel::Digest,
) -> Result<DeviceIdentity, ArtifactSessionError> {
    let current_artifact = session.artifact()?;
    if &current_artifact != expected_artifact {
        return Err(ArtifactSessionError::State(format!(
            "Fix: pre-recovery session serves artifact `{current_artifact}`, expected `{expected_artifact}`"
        )));
    }
    let identity = session.rematerialize()?;
    let recovered_artifact = session.artifact()?;
    if &recovered_artifact != expected_artifact {
        return Err(ArtifactSessionError::State(format!(
            "Fix: rematerialized session serves artifact `{recovered_artifact}`, expected `{expected_artifact}`"
        )));
    }
    Ok(identity)
}
