use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use vyre_driver::{BackendError, BindingSet, BoundResource, Completion, DeviceIdentity};
use vyre_megakernel::{ArtifactValueId, Digest};

use super::session::{ArtifactSession, ArtifactSessionError};

/// Execution phase of a retained artifact session state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedSessionPhase {
    /// Session is ready to accept state replacement or submission.
    Ready,
    /// Submission is actively executing on the device.
    InFlight,
    /// Session encountered an unrecoverable failure and is quarantined.
    Quarantined,
}

/// Typed transition on the retained artifact session state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetainedSessionTransition {
    /// Replace the entire retained state map before submission.
    ReplaceState {
        /// Expected generation number.
        expected_generation: u64,
    },
    /// Begin execution submission.
    BeginSubmission,
    /// Complete submission and advance the retained generation.
    CompleteSubmission {
        /// Updated generation number.
        new_generation: u64,
    },
    /// Quarantine the session following unrecoverable failure.
    Quarantine {
        /// Reason for quarantine.
        reason: String,
    },
}

#[derive(Debug)]
struct RetainedSessionStateMachine {
    generation: u64,
    phase: RetainedSessionPhase,
    values: BTreeMap<ArtifactValueId, Vec<u8>>,
    quarantine_reason: Option<String>,
}

impl RetainedSessionStateMachine {
    fn new(initial: BTreeMap<ArtifactValueId, Vec<u8>>) -> Self {
        Self {
            generation: 1,
            phase: RetainedSessionPhase::Ready,
            values: initial,
            quarantine_reason: None,
        }
    }

    fn apply_transition(
        &mut self,
        transition: RetainedSessionTransition,
    ) -> Result<(), ArtifactSessionError> {
        match (self.phase, transition) {
            (
                RetainedSessionPhase::Ready,
                RetainedSessionTransition::ReplaceState {
                    expected_generation,
                },
            ) => {
                if expected_generation != self.generation {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: generation mismatch in retained state replacement: expected {}, actual {}",
                            expected_generation, self.generation
                        ),
                    }
                    .into());
                }
                Ok(())
            }
            (RetainedSessionPhase::Ready, RetainedSessionTransition::BeginSubmission) => {
                self.phase = RetainedSessionPhase::InFlight;
                Ok(())
            }
            (
                RetainedSessionPhase::InFlight,
                RetainedSessionTransition::CompleteSubmission { new_generation },
            ) => {
                self.phase = RetainedSessionPhase::Ready;
                self.generation = new_generation;
                Ok(())
            }
            (_, RetainedSessionTransition::Quarantine { reason }) => {
                self.phase = RetainedSessionPhase::Quarantined;
                self.quarantine_reason = Some(reason);
                Ok(())
            }
            (current_phase, attempted) => Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: illegal retained session transition {:?} while in phase {:?}",
                    attempted, current_phase
                ),
            }
            .into()),
        }
    }
}

/// Runtime-owned retained binding policy over one immutable [`ArtifactSession`].
pub struct RetainedArtifactSession {
    session: ArtifactSession,
    retained_values: BTreeSet<ArtifactValueId>,
    state_machine: Mutex<RetainedSessionStateMachine>,
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
            state_machine: Mutex::new(RetainedSessionStateMachine::new(initial)),
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

    /// Current retained state machine generation.
    pub fn generation(&self) -> Result<u64, ArtifactSessionError> {
        let sm = self
            .state_machine
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(sm.generation)
    }

    /// Current retained session lifecycle phase.
    pub fn phase(&self) -> Result<RetainedSessionPhase, ArtifactSessionError> {
        let sm = self
            .state_machine
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(sm.phase)
    }

    /// Apply an explicit transition to the retained session state machine.
    pub fn transition(
        &self,
        transition: RetainedSessionTransition,
    ) -> Result<(), ArtifactSessionError> {
        let mut sm = self
            .state_machine
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        sm.apply_transition(transition)
    }

    /// Reacquire a device and rematerialize the authenticated artifact bytes.
    pub fn rematerialize(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.rematerialize()
    }

    /// Replace runtime-owned retained bytes before the next submission.
    ///
    /// # Errors
    ///
    /// Returns an error unless the update covers exactly every retained ABI value
    /// and the state machine is in the `Ready` phase.
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
        let mut sm = self
            .state_machine
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let expected_generation = sm.generation;
        sm.apply_transition(RetainedSessionTransition::ReplaceState {
            expected_generation,
        })?;
        sm.values = values;
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
        let mut sm = self
            .state_machine
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        sm.apply_transition(RetainedSessionTransition::BeginSubmission)?;
        for (value, bytes) in sm.values.iter() {
            bindings.insert(*value, BoundResource::Host(bytes.clone()));
        }
        let current_gen = sm.generation;
        let submission_result = self.session.submit_and_wait(bindings);
        let completion = match submission_result {
            Ok(comp) => comp,
            Err(error) => {
                let _ = sm.apply_transition(RetainedSessionTransition::Quarantine {
                    reason: error.to_string(),
                });
                return Err(error);
            }
        };
        if completion.retained.keys().copied().collect::<BTreeSet<_>>() != self.retained_values {
            let _ = sm.apply_transition(RetainedSessionTransition::Quarantine {
                reason: "missing retained output keys in completion".to_string(),
            });
            return Err(BackendError::InvalidProgram {
                fix: "Fix: artifact completion must return exactly every retained ABI value."
                    .to_string(),
            }
            .into());
        }
        sm.values = completion.retained.clone();
        sm.apply_transition(RetainedSessionTransition::CompleteSubmission {
            new_generation: current_gen.wrapping_add(1).max(1),
        })?;
        Ok(completion)
    }
}
