//! Persistent resident-work-queue execution over authenticated artifact sessions.

use std::collections::BTreeMap;

use vyre_driver::{BackendError, BackendRegistration, Completion, DeviceIdentity};
use vyre_megakernel::{ArtifactValueId, Digest};

use crate::artifact_admission::{ArtifactSession, ArtifactSessionError, RetainedArtifactSession};
use crate::recovery::{classify_backend_error, RecoveryClass};

/// Host-visible resident work-queue state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQueueState {
    /// Queue control words.
    pub control: Vec<u8>,
    /// Packed work-slot ring.
    pub ring: Vec<u8>,
    /// Device debug-log storage.
    pub debug_log: Vec<u8>,
    /// Runtime IO request/completion queue.
    pub io_queue: Vec<u8>,
}

/// Completed resident work-queue update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQueueCompletion {
    /// Updated retained queue state.
    pub state: ResidentQueueState,
    /// Backend-measured device duration when available.
    pub device_ns: Option<u64>,
}

#[derive(Clone, Copy)]
struct ResidentQueueAbi {
    control: ArtifactValueId,
    ring: ArtifactValueId,
    debug_log: ArtifactValueId,
    io_queue: ArtifactValueId,
}

impl ResidentQueueAbi {
    fn resolve(session: &ArtifactSession) -> Result<Self, ArtifactSessionError> {
        Ok(Self {
            control: session.resource("control")?,
            ring: session.resource("ring_buffer")?,
            debug_log: session.resource("debug_log")?,
            io_queue: session.resource("io_queue")?,
        })
    }

    fn bind(self, state: ResidentQueueState) -> BTreeMap<ArtifactValueId, Vec<u8>> {
        BTreeMap::from([
            (self.control, state.control),
            (self.ring, state.ring),
            (self.debug_log, state.debug_log),
            (self.io_queue, state.io_queue),
        ])
    }

    fn completion(
        self,
        completion: Completion,
    ) -> Result<ResidentQueueCompletion, ArtifactSessionError> {
        let mut retained = completion.retained;
        let mut take = |value: ArtifactValueId,
                        name: &str|
         -> Result<Vec<u8>, ArtifactSessionError> {
            retained.remove(&value).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: persistent artifact completion must return retained queue resource `{name}`."
                    ),
                }
                .into()
            })
        };
        let state = ResidentQueueState {
            control: take(self.control, "control")?,
            ring: take(self.ring, "ring_buffer")?,
            debug_log: take(self.debug_log, "debug_log")?,
            io_queue: take(self.io_queue, "io_queue")?,
        };
        if !retained.is_empty() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: persistent queue artifacts must expose exactly control, ring_buffer, debug_log, and io_queue as retained values.".to_string(),
            }
            .into());
        }
        Ok(ResidentQueueCompletion {
            state,
            device_ns: completion.device_ns,
        })
    }
}

/// Authenticated persistent queue session over one immutable artifact identity.
pub struct PersistentExecutor {
    session: RetainedArtifactSession,
    abi: ResidentQueueAbi,
}

impl PersistentExecutor {
    /// Authenticate and materialize an artifact envelope, then initialize retained queue state.
    pub fn from_bytes(
        registration: &'static BackendRegistration,
        envelope_bytes: &[u8],
        initial: ResidentQueueState,
    ) -> Result<Self, ArtifactSessionError> {
        let session = ArtifactSession::from_bytes(registration, envelope_bytes)?;
        let abi = ResidentQueueAbi::resolve(&session)?;
        let session = RetainedArtifactSession::new(session, abi.bind(initial))?;
        Ok(Self { session, abi })
    }

    /// Neutral artifact identity preserved across every queue update and recovery.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        self.session.artifact()
    }

    /// Current acquired device generation.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.device()
    }

    /// Replace the host queue mirror, submit once, and return the updated retained state.
    pub fn submit_and_wait(
        &self,
        state: ResidentQueueState,
    ) -> Result<ResidentQueueCompletion, ArtifactSessionError> {
        self.session.replace_retained(self.abi.bind(state))?;
        let bindings = self.session.bindings()?;
        let completion = self.session.submit_and_wait(bindings)?;
        self.abi.completion(completion)
    }

    /// Rematerialize authenticated bytes after a structured device-loss failure.
    ///
    /// # Errors
    ///
    /// Returns the original backend failure for every non-device-loss class.
    pub fn recover(&self, failure: BackendError) -> Result<DeviceIdentity, ArtifactSessionError> {
        if classify_backend_error(&failure) != RecoveryClass::DeviceLoss {
            return Err(failure.into());
        }
        self.session.rematerialize()
    }
}
