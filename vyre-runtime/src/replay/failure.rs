//! Failure evidence carried in a replay record's tail words.
//!
//! A replay that only knows a slot was published cannot distinguish a run that
//! completed from one that lost the device on the last epoch. The tail words
//! carry the recovery class, the stable backend error code and a digest of the
//! outputs observed at failure, so a later replay diffs against what the
//! original run actually saw rather than against silence.

use vyre_driver::backend::BackendError;
use vyre_foundation::diagnostics::RetryClass;

use super::output_digest;
use crate::recovery::classify_backend_error;

/// Backend/runtime failure class encoded into the replay record tail.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplayFailureClass {
    /// No failure evidence was recorded for this published slot.
    #[default]
    None,
    /// Backend context, adapter, or compiled-pipeline state was lost or stale.
    DeviceLoss,
    /// Queue/resource pressure that can be retried without recompilation.
    TransientQueue,
    /// Program/lowering/kernel-source failure that should not be retried as-is.
    ProgramBug,
    /// Failure did not match a known automated recovery class.
    Unclassified,
}

impl ReplayFailureClass {
    const NONE: u32 = 0;
    const DEVICE_LOSS: u32 = 1;
    const TRANSIENT_QUEUE: u32 = 2;
    const PROGRAM_BUG: u32 = 3;
    const UNCLASSIFIED: u32 = 4;

    pub(super) const fn encode(self) -> u32 {
        match self {
            Self::None => Self::NONE,
            Self::DeviceLoss => Self::DEVICE_LOSS,
            Self::TransientQueue => Self::TRANSIENT_QUEUE,
            Self::ProgramBug => Self::PROGRAM_BUG,
            Self::Unclassified => Self::UNCLASSIFIED,
        }
    }

    /// Any word outside the declared set decodes to `Unclassified` rather than
    /// erroring: a log written by a newer build is still readable, and an
    /// unrecognized class is not a reason to discard the record around it.
    const fn decode(raw: u32) -> Self {
        match raw {
            Self::NONE => Self::None,
            Self::DEVICE_LOSS => Self::DeviceLoss,
            Self::TRANSIENT_QUEUE => Self::TransientQueue,
            Self::PROGRAM_BUG => Self::ProgramBug,
            Self::UNCLASSIFIED => Self::Unclassified,
            _ => Self::Unclassified,
        }
    }

    const fn from_retry_class(class: RetryClass) -> Self {
        match class {
            RetryClass::NewDevice => Self::DeviceLoss,
            RetryClass::SameDevice => Self::TransientQueue,
            RetryClass::Never | RetryClass::RecompileSource => Self::ProgramBug,
            _ => Self::Unclassified,
        }
    }
}

/// Failure evidence captured in a replay record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayFailureEvidence {
    /// Terminal or observed ring status word for the failed slot.
    pub slot_status: u32,
    /// Recovery-oriented failure class.
    pub failure_class: ReplayFailureClass,
    /// Stable backend error code. Zero means no backend error was known.
    pub backend_error_code: u32,
    /// Stable digest over output bytes observed before/at failure.
    pub output_digest: u64,
}

impl ReplayFailureEvidence {
    /// Build replay failure evidence from a backend error and observed output bytes.
    #[must_use]
    pub fn from_backend_error(slot_status: u32, error: &BackendError, output_bytes: &[u8]) -> Self {
        Self {
            slot_status,
            failure_class: ReplayFailureClass::from_retry_class(classify_backend_error(error)),
            backend_error_code: error.code().stable_id(),
            output_digest: output_digest(output_bytes),
        }
    }

    /// Decode the four tail words, or `None` when all four are zero.
    ///
    /// All-zero is how a record with no failure is written, so it must not
    /// decode to evidence claiming class `None` with a zero code: a caller
    /// cannot tell that from a real failure whose fields happened to be zero.
    pub(super) fn from_words(
        slot_status: u32,
        failure_class: u32,
        backend_error_code: u32,
        output_digest: u64,
    ) -> Option<Self> {
        if slot_status == 0 && failure_class == 0 && backend_error_code == 0 && output_digest == 0 {
            return None;
        }
        Some(Self {
            slot_status,
            failure_class: ReplayFailureClass::decode(failure_class),
            backend_error_code,
            output_digest,
        })
    }
}
