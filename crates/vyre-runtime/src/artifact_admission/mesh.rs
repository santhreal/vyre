//! Coordinated submission of one artifact across the device mesh its schedule
//! placed it on.
//!
//! The placement is a compile decision: the artifact states which devices run
//! it, how each region is cut, and which transfers move bytes between them. This
//! module submits that topology and nothing else. It does not choose a
//! partition, does not re-route a transfer, and does not run a mesh placement on
//! fewer devices than it was compiled for.
//!
//! A device that rejects its submission ends the submission. There is no host
//! path for the work it held: the bytes were cut for that device, so completing
//! elsewhere would compute a different program.

use std::collections::BTreeMap;
use std::sync::Arc;

use thiserror::Error;
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, Completion, PeerTopology,
    Submission,
};
use vyre_megakernel::allocation::DeviceSlot;
use vyre_megakernel::ArtifactEnvelope;

use super::{admit_envelope, AdmittedArtifact, ArtifactAdmissionError};

/// Failure to bind, route, materialize, or submit one mesh placement.
#[derive(Debug, Error)]
pub enum MeshSessionError {
    /// Canonical envelope or per-device payload admission failed.
    #[error(transparent)]
    Admission(#[from] ArtifactAdmissionError),
    /// A registered device rejected materialization.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// No device was supplied.
    #[error("a mesh submission needs at least one acquired device. Fix: supply one materializer per device the artifact is submitted to")]
    NoDevice,
    /// Supplied devices disagree on the payload format they consume.
    #[error("device {device} consumes target format {found} and device {anchor} consumes {expected}. Fix: submit a mesh on devices of one target format, or compile one artifact per format")]
    MixedFormats {
        /// Device whose format differs.
        device: u16,
        /// Format that device consumes.
        found: String,
        /// First supplied device.
        anchor: u16,
        /// Format the first device consumes.
        expected: String,
    },
    /// The artifact is submitted to a device the caller does not hold.
    #[error("the artifact is submitted to device {device} and no materializer was supplied for it. Fix: acquire every device the placement names, or recompile the graph for the devices you hold")]
    MissingDevice {
        /// Submission device with no supplied materializer.
        device: u16,
    },
    /// A device was supplied that the placement does not use.
    #[error("device {device} was supplied and this artifact places no work on it. Fix: supply exactly the devices the artifact topology names")]
    UnplacedDevice {
        /// Supplied device the topology omits.
        device: u16,
    },
    /// A recorded transfer has no direct peer route.
    #[error("the topology moves {bytes} bytes from device {from} to device {to} and the peer topology reports no direct link. Fix: enable peer access between those devices, or recompile for a mesh whose links the driver authenticates")]
    UnroutableTransfer {
        /// Source device of the transfer.
        from: u16,
        /// Destination device of the transfer.
        to: u16,
        /// Bytes the transfer moves.
        bytes: u64,
    },
    /// No bindings were supplied for a device the artifact runs on.
    #[error("device {device} runs part of this artifact and no bindings were supplied for it. Fix: bind every device the artifact is submitted to")]
    MissingBindings {
        /// Device with no bindings.
        device: u16,
    },
    /// One device of the mesh failed. The submission is over.
    #[error("device {device} failed its part of the mesh submission: {source}. Fix: report the failure to the caller; the shards this device held have no host path")]
    DeviceFailure {
        /// Device that failed.
        device: u16,
        /// Backend failure the device reported.
        #[source]
        source: BackendError,
    },
}

/// One artifact materialized on every device its placement names.
pub struct MeshSession {
    admitted: AdmittedArtifact,
    instances: BTreeMap<DeviceSlot, Box<dyn ArtifactInstance>>,
}

impl MeshSession {
    /// Admit one envelope and materialize it on exactly the devices its topology
    /// names.
    ///
    /// # Errors
    ///
    /// Returns when no device is supplied, the supplied devices consume
    /// different target formats, the supplied set is not the set the topology
    /// names, a recorded transfer has no direct peer route, a per-device payload
    /// is absent, or a device rejects materialization.
    pub fn new(
        envelope: ArtifactEnvelope,
        devices: Vec<(DeviceSlot, Arc<dyn ArtifactMaterializer>)>,
        peers: &PeerTopology,
    ) -> Result<Self, MeshSessionError> {
        let (anchor_slot, anchor) = devices.first().ok_or(MeshSessionError::NoDevice)?;
        let format = anchor.device().target_format().clone();
        for (slot, materializer) in &devices {
            let supplied = materializer.device().target_format();
            if *supplied != format {
                return Err(MeshSessionError::MixedFormats {
                    device: slot.0,
                    found: supplied.identity().to_string(),
                    anchor: anchor_slot.0,
                    expected: format.identity().to_string(),
                });
            }
        }
        let admitted = admit_envelope(envelope, &format)?;
        let placed = admitted.submission_devices();
        for device in &placed {
            if !devices.iter().any(|(slot, _)| slot == device) {
                return Err(MeshSessionError::MissingDevice { device: device.0 });
            }
        }
        for (slot, _) in &devices {
            if !placed.contains(slot) {
                return Err(MeshSessionError::UnplacedDevice { device: slot.0 });
            }
        }
        for transfer in &admitted.neutral().topology().transfers {
            if !peers
                .capability(u32::from(transfer.from.0), u32::from(transfer.to.0))
                .is_direct()
            {
                return Err(MeshSessionError::UnroutableTransfer {
                    from: transfer.from.0,
                    to: transfer.to.0,
                    bytes: transfer.bytes,
                });
            }
        }
        let mut instances = BTreeMap::new();
        for (slot, materializer) in devices {
            let payload = admitted.target_payload_for_device(slot)?;
            let instance = materializer
                .materialize(admitted.neutral(), payload)
                .map_err(|source| MeshSessionError::DeviceFailure {
                    device: slot.0,
                    source,
                })?;
            if instance.artifact() != admitted.neutral().digest()
                || instance.payload() != payload.digest()
                || instance.device() != materializer.device().identity()
            {
                return Err(MeshSessionError::Backend(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: the instance materialized for device {} must name the admitted artifact, that device's payload, and the acquired device generation.",
                        slot.0
                    ),
                }));
            }
            instances.insert(slot, instance);
        }
        Ok(Self {
            admitted,
            instances,
        })
    }

    /// Devices this session submits to, in slot order.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceSlot> {
        self.instances.keys().copied().collect()
    }

    /// Submission stages the placement records.
    #[must_use]
    pub fn stage_count(&self) -> u32 {
        self.admitted.neutral().topology().stage_count()
    }

    /// Build an empty typed binding set for one device of this placement.
    ///
    /// # Errors
    ///
    /// Returns when the placement does not name `device`.
    pub fn bindings(&self, device: DeviceSlot) -> Result<BindingSet, MeshSessionError> {
        if !self.instances.contains_key(&device) {
            return Err(MeshSessionError::UnplacedDevice { device: device.0 });
        }
        Ok(BindingSet::new(self.admitted.neutral().digest()))
    }

    /// Submit every device's bindings as one coordinated placement.
    ///
    /// # Errors
    ///
    /// Returns when a device of the placement has no bindings, bindings name a
    /// device the placement omits, or a device rejects its submission.
    pub fn submit(
        &self,
        mut bindings: BTreeMap<DeviceSlot, BindingSet>,
    ) -> Result<MeshSubmission, MeshSessionError> {
        let mut submissions = Vec::with_capacity(self.instances.len());
        for (slot, instance) in &self.instances {
            let bound = bindings
                .remove(slot)
                .ok_or(MeshSessionError::MissingBindings { device: slot.0 })?;
            let submission =
                instance
                    .submit(bound)
                    .map_err(|source| MeshSessionError::DeviceFailure {
                        device: slot.0,
                        source,
                    })?;
            submissions.push((*slot, submission));
        }
        if let Some(slot) = bindings.keys().next() {
            return Err(MeshSessionError::UnplacedDevice { device: slot.0 });
        }
        Ok(MeshSubmission { submissions })
    }
}

/// Every in-flight device submission of one coordinated placement.
pub struct MeshSubmission {
    submissions: Vec<(DeviceSlot, Box<dyn Submission>)>,
}

impl MeshSubmission {
    /// Devices with work in flight, in slot order.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceSlot> {
        self.submissions.iter().map(|(slot, _)| *slot).collect()
    }

    /// Whether every device has completed.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.submissions
            .iter()
            .all(|(_, submission)| submission.is_ready())
    }

    /// Wait for every device and return its typed completion.
    ///
    /// # Errors
    ///
    /// Returns the first device failure, naming the device. The remaining
    /// devices are not retried and their shards have no host path.
    pub fn wait(self) -> Result<Vec<(DeviceSlot, Completion)>, MeshSessionError> {
        let mut completions = Vec::with_capacity(self.submissions.len());
        for (slot, submission) in self.submissions {
            let completion =
                submission
                    .wait()
                    .map_err(|source| MeshSessionError::DeviceFailure {
                        device: slot.0,
                        source,
                    })?;
            completions.push((slot, completion));
        }
        Ok(completions)
    }
}
