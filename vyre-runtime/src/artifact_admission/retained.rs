use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use vyre_driver::{BackendError, BindingSet, BoundResource, Completion, DeviceIdentity};
use vyre_megakernel::{ArtifactValueId, Digest};

use super::session::{ArtifactSession, ArtifactSessionError};

/// Runtime-owned retained binding policy over one immutable [`ArtifactSession`].
pub struct RetainedArtifactSession {
    session: ArtifactSession,
    retained_values: BTreeSet<ArtifactValueId>,
    retained: Mutex<BTreeMap<ArtifactValueId, Vec<u8>>>,
}

impl RetainedArtifactSession {
    /// Create retained policy state and require every retained ABI value initially.
    pub fn new(
        session: ArtifactSession,
        initial: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Self, ArtifactSessionError> {
        let retained_values = session.retained_values()?;
        if initial.keys().copied().collect::<BTreeSet<_>>() != retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: initialize exactly every retained artifact value before creating a retained session.".to_string(),
            }
            .into());
        }
        Ok(Self {
            session,
            retained_values,
            retained: Mutex::new(initial),
        })
    }

    /// Neutral artifact identity shared with ephemeral sessions.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        self.session.artifact()
    }

    /// Current immutable device generation identity.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.device()
    }

    /// Build empty transient bindings for the shared neutral artifact.
    pub fn bindings(&self) -> Result<BindingSet, ArtifactSessionError> {
        self.session.bindings()
    }

    /// Reacquire a device and rematerialize the authenticated artifact bytes.
    pub fn rematerialize(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.rematerialize()
    }

    /// Replace runtime-owned retained bytes before the next submission.
    ///
    /// # Errors
    ///
    /// Returns an error unless the update covers exactly every retained ABI value.
    pub fn replace_retained(
        &self,
        values: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<(), ArtifactSessionError> {
        if values.keys().copied().collect::<BTreeSet<_>>() != self.retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: replace exactly every retained artifact value.".to_string(),
            }
            .into());
        }
        *self
            .retained
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))? = values;
        Ok(())
    }

    /// Submit transient bindings, merge retained state, and atomically retain completion state.
    pub fn submit_and_wait(
        &self,
        mut bindings: BindingSet,
    ) -> Result<Completion, ArtifactSessionError> {
        if bindings.artifact() != self.session.artifact()? {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: retained session bindings must name the session artifact digest."
                    .to_string(),
            }
            .into());
        }
        {
            let retained = self
                .retained
                .lock()
                .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
            for (value, bytes) in retained.iter() {
                bindings.insert(*value, BoundResource::Host(bytes.clone()));
            }
        }
        let completion = self.session.submit_and_wait(bindings)?;
        if completion.retained.keys().copied().collect::<BTreeSet<_>>() != self.retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: artifact completion must return exactly every retained ABI value."
                    .to_string(),
            }
            .into());
        }
        *self
            .retained
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))? =
            completion.retained.clone();
        Ok(completion)
    }
}
