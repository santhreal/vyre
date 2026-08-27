//! The artifact instance and submission a fixture materializer hands back.
//!
//! An instance reports three identity fields and completes one submission.
//! Every fixture backend states the same accessors and the same one-shot
//! submission, and differs only in what it does with the bindings it is given.
//! That difference is a closure here, so a method added to `ArtifactInstance`
//! is answered once instead of in every suite that fakes a device.

// Every test binary compiles this module on its own, so a fixture a given suite
// does not ask for is unused in that binary.
#![allow(dead_code)]

use vyre_driver::{
    ArtifactInstance, BackendError, BindingSet, Completion, DeviceIdentity, Submission,
};
use vyre_megakernel::{Artifact, Digest, EmittedResources, TargetPayload};

/// What a fixture instance does with one submission's bindings.
type SubmitFn = dyn Fn(Digest, BindingSet) -> Result<Completion, BackendError> + Send + Sync;

/// A materialized instance whose submission behavior is supplied per fixture.
pub(crate) struct FixtureInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    submit: Box<SubmitFn>,
}

impl FixtureInstance {
    /// An instance identified by `artifact` and `payload` on `device`.
    ///
    /// `submit` receives the artifact digest this instance reports, so a
    /// fixture completion names the same artifact the bindings were validated
    /// against without capturing it separately.
    pub(crate) fn submitting(
        artifact: &Artifact,
        payload: &TargetPayload,
        device: &DeviceIdentity,
        submit: impl Fn(Digest, BindingSet) -> Result<Completion, BackendError> + Send + Sync + 'static,
    ) -> Box<dyn ArtifactInstance> {
        Box::new(Self {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: device.clone(),
            submit: Box::new(submit),
        })
    }

    /// An instance that completes with no outputs and no retained values.
    pub(crate) fn neutral(
        artifact: &Artifact,
        payload: &TargetPayload,
        device: &DeviceIdentity,
    ) -> Box<dyn ArtifactInstance> {
        Self::submitting(artifact, payload, device, |artifact, _| {
            Ok(completion(artifact))
        })
    }
}

impl ArtifactInstance for FixtureInstance {
    fn artifact(&self) -> Digest {
        self.artifact
    }

    fn payload(&self) -> Digest {
        self.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        Ok(Box::new(FixtureSubmission(Some((self.submit)(
            self.artifact,
            bindings,
        )?))))
    }

    fn emitted_resources(&self) -> Result<Vec<EmittedResources>, BackendError> {
        Ok(vec![EmittedResources::default()])
    }
}

/// An empty completion against `artifact`, ready to be extended per fixture.
pub(crate) fn completion(artifact: Digest) -> Completion {
    Completion {
        artifact,
        outputs: std::collections::BTreeMap::new(),
        retained: std::collections::BTreeMap::new(),
        device_ns: None,
    }
}

/// A submission that is ready at once and consumable exactly once.
struct FixtureSubmission(Option<Completion>);

impl Submission for FixtureSubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.0.take().ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: consume a fixture submission once.".to_string(),
        })
    }
}
