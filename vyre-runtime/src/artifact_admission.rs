use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, RwLock};

use thiserror::Error;
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileError, Diagnostic, TargetCompileError, TargetPayload,
    TargetPayloadFormat, ValidatedCompileRequest,
};

use crate::pipeline_cache::{PipelineCacheStore, PipelineFingerprint};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, DeviceIdentity, Submission,
};
use vyre_megakernel::{ArtifactValueId, Digest, ResourceLifetime};

/// Failure to authenticate an artifact envelope or select its exact required payload.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("artifact admission rejected: {source}")]
pub struct ArtifactAdmissionError {
    #[source]
    source: CompileError,
}

impl ArtifactAdmissionError {
    /// Canonical structured diagnostic produced while decoding or selecting the payload.
    #[must_use]
    pub const fn diagnostic(&self) -> &Diagnostic {
        &self.source.diagnostic
    }

    /// Recover the canonical error without flattening its diagnostic context.
    #[must_use]
    pub fn into_compile_error(self) -> CompileError {
        self.source
    }
}

impl From<CompileError> for ArtifactAdmissionError {
    fn from(source: CompileError) -> Self {
        Self { source }
    }
}

/// Authenticated canonical envelope with one caller-selected exact payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedArtifact {
    envelope: ArtifactEnvelope,
    target_payload_index: usize,
}

impl AdmittedArtifact {
    /// Borrow the authenticated canonical envelope.
    #[must_use]
    pub const fn envelope(&self) -> &ArtifactEnvelope {
        &self.envelope
    }

    /// Borrow the canonical backend-neutral artifact.
    #[must_use]
    pub const fn neutral(&self) -> &Artifact {
        self.envelope.neutral()
    }

    /// Borrow the exact target payload selected during admission.
    #[must_use]
    pub fn target_payload(&self) -> &TargetPayload {
        &self.envelope.target_payloads()[self.target_payload_index]
    }

    /// Consume the admission result and recover its owned canonical envelope.
    #[must_use]
    pub fn into_envelope(self) -> ArtifactEnvelope {
        self.envelope
    }
}

/// Decode and authenticate canonical envelope bytes, then require one exact payload format.
pub fn admit_artifact(
    envelope_bytes: &[u8],
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let envelope = ArtifactEnvelope::from_bytes(envelope_bytes)?;
    admit_envelope(envelope, required_format)
}

/// Authenticate an already-decoded canonical envelope and require one exact payload format.
///
/// Prefer this when a producer such as AOT packaging has already decoded the envelope
/// and only the exact target-format selection remains.
pub fn admit_envelope(
    envelope: ArtifactEnvelope,
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let target_payload_index = envelope.require_target_payload_index(required_format)?;
    Ok(AdmittedArtifact {
        envelope,
        target_payload_index,
    })
}

/// Load verified cache payload bytes and admit them as a canonical envelope.
///
/// `DiskCache` / `PipelineCacheStore` are format-agnostic blob stores. AOT
/// writes `ArtifactEnvelope` bytes as the payload (plus the store's
/// own BLAKE3 footer, stripped by [`PipelineCacheStore::get`]). Callers that
/// treat a cache hit as executable MUST run this helper (or
/// [`admit_artifact`] on the payload) before dispatch. A miss is `Ok(None)`.
///
/// # Errors
///
/// Returns [`ArtifactAdmissionError`] when payload bytes are present but are
/// not an authentic envelope with the required target format.
pub fn admit_cached_artifact(
    store: &dyn PipelineCacheStore,
    fingerprint: &PipelineFingerprint,
    required_format: &TargetPayloadFormat,
) -> Result<Option<AdmittedArtifact>, ArtifactAdmissionError> {
    let Some(payload) = store.get(fingerprint) else {
        return Ok(None);
    };
    admit_artifact(&payload, required_format).map(Some)
}

/// Runtime materialization or submission failure with structured admission preserved.
#[derive(Debug, Error)]
pub enum ArtifactSessionError {
    /// Canonical envelope or target-format admission failed.
    #[error(transparent)]
    Admission(#[from] ArtifactAdmissionError),
    /// Neutral compilation or target payload construction failed.
    #[error(transparent)]
    Compile(#[from] CompileError),
    /// The registered target compiler rejected the selected artifact.
    #[error(transparent)]
    Target(#[from] TargetCompileError),
    /// Registered device materialization or submission failed.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Runtime lifecycle state was poisoned by a panic while locked.
    #[error("artifact session state is poisoned: {0}. Fix: discard and rebuild the session")]
    State(String),
}

struct MaterializedArtifact {
    admitted: AdmittedArtifact,
    _materializer: Box<dyn ArtifactMaterializer>,
    instance: Box<dyn ArtifactInstance>,
}

/// Authenticated immutable artifact materialized on one registered device generation.
pub struct ArtifactSession {
    registration: &'static BackendRegistration,
    state: RwLock<MaterializedArtifact>,
}

impl ArtifactSession {
    /// Compile one validated request, attach the registered target payload, and
    /// materialize the authenticated artifact.
    pub fn compile(
        registration: &'static BackendRegistration,
        request: &ValidatedCompileRequest,
    ) -> Result<Self, ArtifactSessionError> {
        let artifact = vyre_megakernel::compile(request)?;
        let compiler = registration.target_compiler()?;
        let envelope = vyre_megakernel::attach_target(artifact, compiler.as_ref())?;
        Self::from_envelope(registration, envelope)
    }

    /// Admit one already-decoded canonical envelope and materialize its exact target bytes.
    pub fn from_envelope(
        registration: &'static BackendRegistration,
        envelope: ArtifactEnvelope,
    ) -> Result<Self, ArtifactSessionError> {
        let materializer = registration.materializer()?;
        let admitted = admit_envelope(envelope, materializer.device().target_format())?;
        let instance = materializer.materialize(admitted.neutral(), admitted.target_payload())?;
        validate_instance(&admitted, materializer.as_ref(), instance.as_ref())?;
        Ok(Self {
            registration,
            state: RwLock::new(MaterializedArtifact {
                admitted,
                _materializer: materializer,
                instance,
            }),
        })
    }

    /// Authenticate canonical envelope bytes and materialize the exact device format.
    pub fn from_bytes(
        registration: &'static BackendRegistration,
        envelope_bytes: &[u8],
    ) -> Result<Self, ArtifactSessionError> {
        let envelope =
            ArtifactEnvelope::from_bytes(envelope_bytes).map_err(ArtifactAdmissionError::from)?;
        Self::from_envelope(registration, envelope)
    }

    /// Neutral artifact identity shared by every session and device generation.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.admitted.neutral().digest())
    }
    /// Exact authenticated target payload identity materialized by this session.
    pub fn payload(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.admitted.target_payload().digest())
    }

    /// Current immutable device generation identity.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.instance.device().clone())
    }

    /// Build an empty typed binding set for this exact artifact.
    pub fn bindings(&self) -> Result<BindingSet, ArtifactSessionError> {
        Ok(BindingSet::new(self.artifact()?))
    }

    /// Submit typed bindings without exposing the materialized native instance.
    pub fn submit(
        &self,
        bindings: BindingSet,
    ) -> Result<Box<dyn Submission>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.instance.submit(bindings)?)
    }

    /// Submit and wait for typed completion/readback.
    pub fn submit_and_wait(
        &self,
        bindings: BindingSet,
    ) -> Result<Completion, ArtifactSessionError> {
        Ok(self.submit(bindings)?.wait()?)
    }

    /// Reacquire the registered device and rematerialize authenticated target bytes.
    ///
    /// This path never invokes the target compiler, semantic optimizer, or lowering.
    pub fn rematerialize(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        let mut state = self
            .state
            .write()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let materializer = self.registration.materializer()?;
        let admitted = admit_envelope(
            state.admitted.envelope().clone(),
            materializer.device().target_format(),
        )?;
        let instance = materializer.materialize(admitted.neutral(), admitted.target_payload())?;
        validate_instance(&admitted, materializer.as_ref(), instance.as_ref())?;
        let identity = instance.device().clone();
        *state = MaterializedArtifact {
            admitted,
            _materializer: materializer,
            instance,
        };
        Ok(identity)
    }

    /// Resolve one canonical artifact ABI value by its stable resource name.
    pub fn resource(&self, name: &str) -> Result<ArtifactValueId, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        state
            .admitted
            .neutral()
            .resources()
            .iter()
            .find(|resource| resource.name == name)
            .map(|resource| resource.value)
            .ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: artifact ABI does not declare required runtime resource `{name}`."
                    ),
                }
                .into()
            })
    }

    fn retained_values(&self) -> Result<BTreeSet<ArtifactValueId>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state
            .admitted
            .neutral()
            .resources()
            .iter()
            .filter(|resource| resource.lifetime == ResourceLifetime::Retained)
            .map(|resource| resource.value)
            .collect())
    }
}

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

fn validate_instance(
    admitted: &AdmittedArtifact,
    materializer: &dyn ArtifactMaterializer,
    instance: &dyn ArtifactInstance,
) -> Result<(), BackendError> {
    if instance.artifact() != admitted.neutral().digest()
        || instance.payload() != admitted.target_payload().digest()
        || instance.device() != materializer.device().identity()
    {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: materialized instance identities must exactly match the admitted artifact, target payload, and acquired device generation.".to_string(),
        });
    }
    Ok(())
}
