//! External resource admission, layout/usage transition execution, and timeline synchronization (Row 111).
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides runtime admission for external memory handles and executes selected
//! transition schedules with guaranteed zero host copies and fine-grained timeline synchronization.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use thiserror::Error;

use vyre_driver::{
    AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind, ImageDimensions,
    ImageFormat, ResourceAbiError, ResourcePermittedUsages, ResourceTransitionSchedule,
    TransitionExecutionReport,
};
use vyre_foundation::failure_domain::{reclaim_poisoned_read, reclaim_poisoned_write};

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "runtime external resource admission";

/// The record of every external memory handle the device currently holds.
const RESOURCE_TABLE: &str = "the admitted external resource table";

/// Which views depend on each admitted resource.
const VIEW_INDEX: &str = "the dependent view index";

/// Which pipelines depend on each admitted resource.
const PIPELINE_INDEX: &str = "the dependent pipeline index";

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
    resources: RwLock<HashMap<u64, AdmittedResourceRecord>>,
    dependent_views: RwLock<HashMap<u64, HashSet<u64>>>,
    dependent_pipelines: RwLock<HashMap<u64, HashSet<u64>>>,
}

impl ExternalResourceAdmissionManager {
    /// Create a new external resource admission manager for `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            resources: RwLock::new(HashMap::new()),
            dependent_views: RwLock::new(HashMap::new()),
            dependent_pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Return the device ID managed by this admission manager.
    #[must_use]
    pub fn device_id(&self) -> u64 {
        self.device_id
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
            match memory_kind {
                ExternalMemoryKind::DmaBuf | ExternalMemoryKind::OpaqueFd => {
                    if record.format.is_depth_stencil() {
                        return Err(ExternalAdmissionError::InvalidCombination {
                            resource_id: record.resource_id,
                            format: record.format,
                            memory_kind,
                        });
                    }
                }
                ExternalMemoryKind::Win32Nt | ExternalMemoryKind::Win32Kmt => {
                    if record.format.is_planar_video()
                        && record.format != ImageFormat::Yuv420SemiPlanar
                    {
                        return Err(ExternalAdmissionError::InvalidCombination {
                            resource_id: record.resource_id,
                            format: record.format,
                            memory_kind,
                        });
                    }
                }
                ExternalMemoryKind::MetalSharedResource => {
                    if record.format.is_planar_video() {
                        return Err(ExternalAdmissionError::InvalidCombination {
                            resource_id: record.resource_id,
                            format: record.format,
                            memory_kind,
                        });
                    }
                }
                ExternalMemoryKind::HostAllocation => {
                    if record.format.is_depth_stencil() {
                        return Err(ExternalAdmissionError::InvalidCombination {
                            resource_id: record.resource_id,
                            format: record.format,
                            memory_kind,
                        });
                    }
                }
            }
        }

        let lease_id = NEXT_LEASE_ID.fetch_add(1, Ordering::Relaxed);
        let resource_id = record.resource_id;
        let format = record.format;
        let dimensions = record.dimensions;
        let row_pitch_bytes = record.row_pitch_bytes;
        let is_zero_copy = record.is_zero_copy;

        let mut map = reclaim_poisoned_write(&self.resources, OWNER, RESOURCE_TABLE);
        map.insert(resource_id, record);

        Ok(AdmittedExternalResourceLease {
            resource_id,
            lease_id,
            format,
            dimensions,
            row_pitch_bytes,
            is_zero_copy,
        })
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
        let map = reclaim_poisoned_read(&self.resources, OWNER, RESOURCE_TABLE);
        let record = map
            .get(&resource_id)
            .ok_or(ExternalAdmissionError::ResourceNotFound { resource_id })?;
        if !record.is_valid {
            return Err(ExternalAdmissionError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut views = reclaim_poisoned_write(&self.dependent_views, OWNER, VIEW_INDEX);
        views.entry(resource_id).or_default().insert(view_id);
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
        let map = reclaim_poisoned_read(&self.resources, OWNER, RESOURCE_TABLE);
        let record = map
            .get(&resource_id)
            .ok_or(ExternalAdmissionError::ResourceNotFound { resource_id })?;
        if !record.is_valid {
            return Err(ExternalAdmissionError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut pipelines =
            reclaim_poisoned_write(&self.dependent_pipelines, OWNER, PIPELINE_INDEX);
        pipelines
            .entry(resource_id)
            .or_default()
            .insert(pipeline_id);
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
        let map = reclaim_poisoned_write(&self.resources, OWNER, RESOURCE_TABLE);

        // 1. Verify every referenced resource is valid
        for (resource_id, _) in &schedule.transitions {
            let record = map
                .get(resource_id)
                .ok_or(ExternalAdmissionError::ResourceNotFound {
                    resource_id: *resource_id,
                })?;
            if !record.is_valid {
                return Err(ExternalAdmissionError::ResourceInvalidated {
                    resource_id: *resource_id,
                });
            }
        }

        // 2. Execute transitions exactly
        let report = TransitionExecutionReport::execute_exact(schedule);
        Ok(report)
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
        let mut map = reclaim_poisoned_write(&self.resources, OWNER, RESOURCE_TABLE);
        let record = map
            .get_mut(&resource_id)
            .ok_or(ExternalAdmissionError::ResourceNotFound { resource_id })?;
        let next_gen = record.advance_generation(expected_gen)?;
        Ok(next_gen)
    }

    /// Invalidate all resources, views, and dependent pipelines upon device loss.
    pub fn invalidate_device_loss(&self) -> DeviceLossInvalidationReport {
        let mut report = DeviceLossInvalidationReport {
            device_id: self.device_id,
            invalidated_resources: Vec::new(),
            invalidated_views: Vec::new(),
            invalidated_artifacts: Vec::new(),
        };

        let mut map = reclaim_poisoned_write(&self.resources, OWNER, RESOURCE_TABLE);
        let mut views = reclaim_poisoned_write(&self.dependent_views, OWNER, VIEW_INDEX);
        let mut pipelines =
            reclaim_poisoned_write(&self.dependent_pipelines, OWNER, PIPELINE_INDEX);

        for (res_id, record) in map.iter_mut() {
            record.invalidate_on_device_loss();
            report.invalidated_resources.push(*res_id);

            if let Some(view_set) = views.remove(res_id) {
                report.invalidated_views.extend(view_set);
            }
            if let Some(pipe_set) = pipelines.remove(res_id) {
                report.invalidated_artifacts.extend(pipe_set);
            }
        }

        report.invalidated_resources.sort_unstable();
        report.invalidated_views.sort_unstable();
        report.invalidated_artifacts.sort_unstable();
        report
    }

    /// Look up an admitted resource record.
    #[must_use]
    pub fn query_resource(&self, resource_id: u64) -> Option<AdmittedResourceRecord> {
        let map = reclaim_poisoned_read(&self.resources, OWNER, RESOURCE_TABLE);
        map.get(&resource_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use vyre_driver::{
        ColorInterpretation, ResourceLayoutState, ResourceUsageTransition, TimelineSyncProtocol,
    };

    const RESOURCE_ID: u64 = 7001;

    fn manager_with_one_resource() -> Arc<ExternalResourceAdmissionManager> {
        let manager = Arc::new(ExternalResourceAdmissionManager::new(9));
        let record = AdmittedResourceRecord::new_external_import_2d(
            RESOURCE_ID,
            9,
            ImageFormat::Rgba8Unorm,
            ColorInterpretation::Srgb,
            64,
            64,
            64 * 4,
            ResourcePermittedUsages::SAMPLED.union(ResourcePermittedUsages::TRANSFER_DST),
            ExternalMemoryKind::DmaBuf,
            0x7001_C001,
            TimelineSyncProtocol::TimelineSemaphore {
                timeline_id: 1,
                wait_value: 0,
                signal_value: 1,
            },
        );
        manager
            .admit_external_resource(record)
            .expect("admit the fixture resource");
        manager
    }

    /// Poison one of the manager's locks from a thread that panics holding it.
    fn poison<T: Send + Sync + 'static>(
        manager: Arc<ExternalResourceAdmissionManager>,
        pick: fn(&ExternalResourceAdmissionManager) -> &RwLock<T>,
    ) {
        let joined = std::thread::spawn(move || {
            let _guard = pick(&manager).write().expect("lock is not yet poisoned");
            panic!("a panic holding an external admission lock");
        })
        .join();
        assert!(joined.is_err(), "the poisoning thread must have panicked");
    }

    /// Closes the class "a poisoned lock reported as an invalidated resource".
    ///
    /// `AdmittedResourceRecord` is the only record of an external handle the
    /// device holds, so discarding the table on a poison report leaks every
    /// handle in it and tells the caller the resource died when only a guard
    /// did. Every lock-touching method clears the poison and continues, so the
    /// three locks are poisoned first and then each method is exercised.
    ///
    /// A `ResourceInvalidated` result here means poison is being reported as
    /// device loss again. It does not prove the table is internally consistent
    /// after an arbitrary panic; consistency comes from every mutation between
    /// the guard and the panic being a single map insert or a generation bump.
    #[test]
    fn external_admission_recovers_every_poisoned_lock_instead_of_invalidating_resources() {
        let manager = manager_with_one_resource();

        poison(Arc::clone(&manager), |m| &m.resources);
        poison(Arc::clone(&manager), |m| &m.dependent_views);
        poison(Arc::clone(&manager), |m| &m.dependent_pipelines);

        assert!(
            manager.resources.is_poisoned()
                && manager.dependent_views.is_poisoned()
                && manager.dependent_pipelines.is_poisoned(),
            "the fixture must leave all three locks poisoned"
        );

        manager
            .register_dependent_view(RESOURCE_ID, 11)
            .expect("Fix: register_dependent_view must clear the poison and continue");
        manager
            .register_dependent_pipeline(RESOURCE_ID, 12)
            .expect("Fix: register_dependent_pipeline must clear the poison and continue");

        let mut schedule = ResourceTransitionSchedule::new();
        schedule.add_transition(
            RESOURCE_ID,
            ResourceUsageTransition::to_sampled(ResourceLayoutState::General),
        );
        manager
            .execute_transition_schedule(&schedule)
            .expect("Fix: execute_transition_schedule must clear the poison and continue");

        assert_eq!(
            manager
                .mutate_resource(RESOURCE_ID, 1)
                .expect("Fix: mutate_resource must clear the poison and continue"),
            2,
            "the generation must advance across a recovered poison"
        );

        let record = manager
            .query_resource(RESOURCE_ID)
            .expect("Fix: query_resource must clear the poison and continue");
        assert!(
            record.is_valid,
            "a recovered poison must not mark the resource invalid"
        );

        let report = manager.invalidate_device_loss();
        assert_eq!(
            report.invalidated_resources,
            vec![RESOURCE_ID],
            "device loss must still see the resource the recovered table holds"
        );
        assert_eq!(report.invalidated_views, vec![11]);
        assert_eq!(report.invalidated_artifacts, vec![12]);
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
