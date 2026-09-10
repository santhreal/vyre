//! External resource admission, layout/usage transition execution, and timeline synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides runtime admission for external memory handles and executes selected
//! transition schedules with guaranteed zero host copies and fine-grained timeline synchronization.
//!
//! The admitted resource table, its dependent indexes, the ceiling every
//! admission is bounded by and device-loss invalidation belong to
//! [`ExternalResourceRegistry`]. What is runtime here is the pre-allocation
//! authentication a record passes, the lease an admission mints, and an error
//! vocabulary that separates an absent record from an invalidated one.

use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use vyre_driver::{
    AdmissionState, AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind,
    ExternalResourceRegistry, ImageDimensions, ImageFormat, ResourceAbiError,
    ResourcePermittedUsages, ResourceTransitionSchedule, TransitionExecutionReport,
};

/// The subsystem every poison report from this registry names as the owner.
const OWNER: &str = "runtime external resource admission";

/// Global lease counter for admitted external resources.
static NEXT_LEASE_ID: AtomicU64 = AtomicU64::new(1);

/// Authenticated lease for an admitted external resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedExternalResourceLease {
    /// Unique resource identifier.
    pub resource_id: u64,
    /// Lease identifier.
    pub lease_id: u64,
    /// Format and element layout.
    pub format: ImageFormat,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes (aligned to hardware boundary).
    pub row_pitch_bytes: u32,
    /// Whether this resource operates with zero host copies.
    pub is_zero_copy: bool,
}

/// Errors occurring during external resource admission or transition execution.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ExternalAdmissionError {
    /// Unsupported format and external memory kind combination rejected before allocation.
    #[error("unsupported zero-copy import combination for resource {resource_id}: format {format:?} with memory kind {memory_kind:?}")]
    InvalidCombination {
        /// Resource ID.
        resource_id: u64,
        /// Image format rejected.
        format: ImageFormat,
        /// External memory kind rejected.
        memory_kind: ExternalMemoryKind,
    },
    /// Resource dimensions or row pitch alignment is invalid.
    #[error("invalid row pitch {provided} for resource {resource_id}, minimum aligned required {required}")]
    InvalidPitch {
        /// Resource ID.
        resource_id: u64,
        /// Provided pitch in bytes.
        provided: u32,
        /// Required aligned pitch in bytes.
        required: u32,
    },
    /// Resource dimensions are zero or invalid.
    #[error("invalid dimensions {dimensions:?} for resource {resource_id}")]
    InvalidDimensions {
        /// Resource ID.
        resource_id: u64,
        /// Provided dimensions.
        dimensions: ImageDimensions,
    },
    /// Requested usage flag is not permitted on this resource.
    #[error("requested usage {requested:?} is not permitted on resource {resource_id}")]
    UsageNotPermitted {
        /// Resource ID.
        resource_id: u64,
        /// Permitted usages.
        requested: ResourcePermittedUsages,
    },
    /// Resource has been invalidated due to device loss.
    #[error("resource {resource_id} is invalidated due to device loss")]
    ResourceInvalidated {
        /// Resource ID.
        resource_id: u64,
    },
    /// Stale frame access detected (generation counter mismatch).
    #[error(
        "stale generation access on resource {resource_id}: expected {expected}, actual {actual}"
    )]
    GenerationMismatch {
        /// Resource ID.
        resource_id: u64,
        /// Expected generation.
        expected: u64,
        /// Actual generation.
        actual: u64,
    },
    /// Referenced resource was not found.
    #[error("resource {resource_id} not found")]
    ResourceNotFound {
        /// Resource ID.
        resource_id: u64,
    },
}

impl From<ResourceAbiError> for ExternalAdmissionError {
    fn from(err: ResourceAbiError) -> Self {
        match err {
            ResourceAbiError::ResourceInvalidated { resource_id } => {
                Self::ResourceInvalidated { resource_id }
            }
            ResourceAbiError::GenerationMismatch {
                resource_id,
                expected,
                actual,
            } => Self::GenerationMismatch {
                resource_id,
                expected,
                actual,
            },
            ResourceAbiError::UsageNotPermitted {
                resource_id,
                requested_usage,
            } => Self::UsageNotPermitted {
                resource_id,
                requested: requested_usage,
            },
            ResourceAbiError::UnsupportedZeroCopyNegotiation {
                resource_id,
                format,
                memory_kind,
            } => Self::InvalidCombination {
                resource_id,
                format,
                memory_kind,
            },
            ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id,
                provided_pitch,
                required_pitch,
            } => Self::InvalidPitch {
                resource_id,
                provided: provided_pitch,
                required: required_pitch,
            },
            ResourceAbiError::DeviceLoss { device_id } => Self::ResourceInvalidated {
                resource_id: device_id,
            },
        }
    }
}

/// Manager for external graphics resource admission, lifecycle, and schedule synchronization.
#[derive(Debug)]
pub struct ExternalResourceAdmissionManager {
    device_id: u64,
    registry: ExternalResourceRegistry,
}

impl ExternalResourceAdmissionManager {
    /// Create a new external resource admission manager for `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            registry: ExternalResourceRegistry::new(OWNER),
        }
    }

    /// Return the device ID managed by this admission manager.
    #[must_use]
    pub fn device_id(&self) -> u64 {
        self.device_id
    }

    /// The admitted resources of this device, their dependent views and their
    /// dependent artifacts.
    #[must_use]
    pub fn registry(&self) -> &ExternalResourceRegistry {
        &self.registry
    }

    /// Report what the registry holds under `resource_id` in this manager's
    /// error vocabulary.
    fn require_valid(&self, resource_id: u64) -> Result<(), ExternalAdmissionError> {
        match self.registry.admission(resource_id) {
            AdmissionState::Valid => Ok(()),
            AdmissionState::Invalidated => {
                Err(ExternalAdmissionError::ResourceInvalidated { resource_id })
            }
            AdmissionState::Absent => Err(ExternalAdmissionError::ResourceNotFound { resource_id }),
        }
    }

    /// Authenticate and admit an external resource before allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalAdmissionError::InvalidCombination`] if the format/memory combination is unsupported,
    /// [`ExternalAdmissionError::InvalidPitch`] if pitch is unaligned, or
    /// [`ExternalAdmissionError::UsageNotPermitted`] if external import is not permitted.
    pub fn admit_external_resource(
        &self,
        record: AdmittedResourceRecord,
    ) -> Result<AdmittedExternalResourceLease, ExternalAdmissionError> {
        // 1. Validate spatial dimensions
        if !record.dimensions.is_valid() {
            return Err(ExternalAdmissionError::InvalidDimensions {
                resource_id: record.resource_id,
                dimensions: record.dimensions,
            });
        }

        // 2. Validate pitch alignment
        let min_pitch = record
            .dimensions
            .width
            .saturating_mul(record.format.bytes_per_pixel());
        let required_pitch = (min_pitch + 255) & !255;
        if record.row_pitch_bytes < min_pitch || (record.row_pitch_bytes & 255) != 0 {
            return Err(ExternalAdmissionError::InvalidPitch {
                resource_id: record.resource_id,
                provided: record.row_pitch_bytes,
                required: required_pitch,
            });
        }

        // 3. Validate usage flags
        if !record
            .permitted_usages
            .contains(ResourcePermittedUsages::EXTERNAL_IMPORT)
        {
            return Err(ExternalAdmissionError::UsageNotPermitted {
                resource_id: record.resource_id,
                requested: ResourcePermittedUsages::EXTERNAL_IMPORT,
            });
        }

        // 4. Authenticate memory kind and format combination before allocation
        if let vyre_driver::ResourceProvenance::ExternalImport { memory_kind, .. } =
            record.provenance
        {
            let rejected = match memory_kind {
                ExternalMemoryKind::DmaBuf
                | ExternalMemoryKind::OpaqueFd
                | ExternalMemoryKind::HostAllocation => record.format.is_depth_stencil(),
                ExternalMemoryKind::Win32Nt | ExternalMemoryKind::Win32Kmt => {
                    record.format.is_planar_video()
                        && record.format != ImageFormat::Yuv420SemiPlanar
                }
                ExternalMemoryKind::MetalSharedResource => record.format.is_planar_video(),
            };
            if rejected {
                return Err(ExternalAdmissionError::InvalidCombination {
                    resource_id: record.resource_id,
                    format: record.format,
                    memory_kind,
                });
            }
        }

        let lease = AdmittedExternalResourceLease {
            resource_id: record.resource_id,
            lease_id: NEXT_LEASE_ID.fetch_add(1, Ordering::Relaxed),
            format: record.format,
            dimensions: record.dimensions,
            row_pitch_bytes: record.row_pitch_bytes,
            is_zero_copy: record.is_zero_copy,
        };
        self.registry.admit_resource(record)?;
        Ok(lease)
    }

    /// Register a dependent view on an admitted resource.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalAdmissionError::ResourceNotFound`] or [`ExternalAdmissionError::ResourceInvalidated`].
    pub fn register_dependent_view(
        &self,
        resource_id: u64,
        view_id: u64,
    ) -> Result<(), ExternalAdmissionError> {
        self.require_valid(resource_id)?;
        self.registry
            .register_dependent_view(resource_id, view_id)?;
        Ok(())
    }

    /// Register a dependent pipeline on an admitted resource.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalAdmissionError::ResourceNotFound`] or [`ExternalAdmissionError::ResourceInvalidated`].
    pub fn register_dependent_pipeline(
        &self,
        resource_id: u64,
        pipeline_id: u64,
    ) -> Result<(), ExternalAdmissionError> {
        self.require_valid(resource_id)?;
        self.registry
            .register_dependent_artifact(resource_id, pipeline_id)?;
        Ok(())
    }

    /// Execute a transition schedule exactly, recording transitions and timeline sync without copies or device-wide waits.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalAdmissionError::ResourceNotFound`] or [`ExternalAdmissionError::ResourceInvalidated`].
    pub fn execute_transition_schedule(
        &self,
        schedule: &ResourceTransitionSchedule,
    ) -> Result<TransitionExecutionReport, ExternalAdmissionError> {
        for (resource_id, _) in &schedule.transitions {
            self.require_valid(*resource_id)?;
        }
        Ok(TransitionExecutionReport::execute_exact(schedule))
    }

    /// Mutate resource and advance generation, verifying expected generation matches.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalAdmissionError::GenerationMismatch`] if stale generation access is detected.
    pub fn mutate_resource(
        &self,
        resource_id: u64,
        expected_gen: u64,
    ) -> Result<u64, ExternalAdmissionError> {
        let advanced = self
            .registry
            .mutate(resource_id, |record| {
                record.advance_generation(expected_gen)
            })
            .ok_or(ExternalAdmissionError::ResourceNotFound { resource_id })?;
        Ok(advanced?)
    }

    /// Invalidate all resources, views, and dependent pipelines upon device loss.
    pub fn invalidate_device_loss(&self) -> DeviceLossInvalidationReport {
        self.registry.invalidate_on_device_loss(self.device_id)
    }

    /// Look up an admitted resource record.
    #[must_use]
    pub fn query_resource(&self, resource_id: u64) -> Option<AdmittedResourceRecord> {
        self.registry.get_resource(resource_id)
    }
}

impl crate::StateOwnerRecovery for ExternalResourceAdmissionManager {
    fn failure_domain(&self) -> crate::FailureDomain {
        crate::FailureDomain::DeviceContext
    }

    fn recovery_class(&self) -> crate::RecoveryClass {
        crate::RecoveryClass::DeviceContextFatal
    }
}
