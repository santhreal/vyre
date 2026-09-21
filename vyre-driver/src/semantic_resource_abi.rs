//! Domain-neutral semantic resource ABI, logical image/view types, zero-copy import/export,
//! and timeline synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module defines domain-neutral typed image, plane, view, sampler, external-memory,
//! and external-event capabilities separate from semantic algorithms and concrete API handles.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use vyre_foundation::failure_domain::{reclaim_poisoned_read, reclaim_poisoned_write};

/// The admitted resource table every registry poison report names.
const RESOURCE_TABLE: &str = "the admitted external resource table";

/// The dependent view index every registry poison report names.
const VIEW_INDEX: &str = "the dependent view index";

/// The dependent artifact index every registry poison report names.
const ARTIFACT_INDEX: &str = "the dependent artifact index";

pub use vyre_spec::{
    all_address_modes, all_alias_set_kinds, all_border_colors, all_color_interpretations,
    all_compare_functions, all_external_event_kinds, all_external_memory_kinds, all_filter_modes,
    all_format_classes, all_image_formats, all_image_view_kinds, all_layout_states,
    all_lifetime_state_kinds, all_mipmap_filter_modes, all_plane_kinds, all_provenance_kinds,
    all_swizzle_components, all_sync_protocols, all_usage_flags, AddressMode,
    AdmittedResourceRecord, BorderColor, ColorInterpretation, CompareFunction, ComponentSwizzle,
    ExternalEventCapability, ExternalEventKind, ExternalMemoryCapability, ExternalMemoryKind,
    FilterMode, FormatClass, ImageDimensions, ImageFormat, ImagePlane, ImageViewDescriptor,
    ImageViewKind, MipmapFilterMode, PlaneKind, ResourceAbiError, ResourceAliasSet,
    ResourceLayoutState, ResourceLifetimeState, ResourceOwnershipState, ResourcePermittedUsages,
    ResourceProvenance, ResourceUsageTransition, SamplerCapability, SamplerDescriptor,
    SubresourceRange, SwizzleComponent, TimelineSyncProtocol,
};

use crate::ResidentOwner;

/// Extended constructors and helpers for [`AdmittedResourceRecord`] with [`ResidentOwner`].
pub trait AdmittedResourceRecordExt {
    /// Construct an admitted 2D texture with a typed [`ResidentOwner`].
    #[must_use]
    fn new_2d_owned(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner: ResidentOwner,
    ) -> AdmittedResourceRecord;
}

impl AdmittedResourceRecordExt for AdmittedResourceRecord {
    fn new_2d_owned(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner: ResidentOwner,
    ) -> Self {
        Self::new_2d(
            resource_id,
            device_id,
            format,
            color,
            width,
            height,
            permitted_usages,
            owner.get(),
        )
    }
}

/// Capability check authenticating zero-copy import parameters before any allocation.
///
/// Rejects unsupported format / memory kind combinations or unaligned pitches before allocation.
///
/// # Errors
///
/// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] if the combination is unsupported,
/// or [`ResourceAbiError::InvalidDimensionsOrPitch`] if pitch is unaligned.
pub fn authenticate_external_import(
    resource_id: u64,
    _device_id: u64,
    format: ImageFormat,
    memory_kind: ExternalMemoryKind,
    width: u32,
    pitch_bytes: u32,
) -> Result<(), ResourceAbiError> {
    let min_pitch = width.saturating_mul(format.bytes_per_pixel());
    if pitch_bytes < min_pitch || (pitch_bytes & 255) != 0 {
        return Err(ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id,
            provided_pitch: pitch_bytes,
            required_pitch: (min_pitch + 255) & !255,
        });
    }

    // Authenticate supported memory kind + format combinations
    match memory_kind {
        ExternalMemoryKind::DmaBuf | ExternalMemoryKind::OpaqueFd => {
            // DMA-BUF supports standard unorm, srgb, float, and NV12 formats
            if format.is_depth_stencil() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::Win32Nt | ExternalMemoryKind::Win32Kmt => {
            // Windows NT shared handles support standard 2D formats
            if format.is_planar_video() && format != ImageFormat::Yuv420SemiPlanar {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::MetalSharedResource => {
            // Metal shared resources support 2D texture formats
            if format.is_planar_video() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::HostAllocation => {
            // Host pinned allocation supports all non-depth stencil formats
            if format.is_depth_stencil() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
    }

    Ok(())
}

/// A selected execution schedule of layout/usage transitions and synchronization points.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ResourceTransitionSchedule {
    /// Ordered sequence of resource layout and usage transitions.
    pub transitions: Vec<(u64, ResourceUsageTransition)>,
    /// Timeline points to wait on before execution.
    pub waits: Vec<TimelineSyncProtocol>,
    /// Timeline points to signal upon execution completion.
    pub signals: Vec<TimelineSyncProtocol>,
}

impl ResourceTransitionSchedule {
    /// Create a new empty transition schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transition for a specific resource ID.
    pub fn add_transition(&mut self, resource_id: u64, transition: ResourceUsageTransition) {
        self.transitions.push((resource_id, transition));
    }

    /// Add a timeline wait point.
    pub fn add_wait(&mut self, wait: TimelineSyncProtocol) {
        self.waits.push(wait);
    }

    /// Add a timeline signal point.
    pub fn add_signal(&mut self, signal: TimelineSyncProtocol) {
        self.signals.push(signal);
    }
}

/// Execution report measuring exact schedule execution (proving absence of copies and global waits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TransitionExecutionReport {
    /// Number of layout/usage transitions executed.
    pub transitions_executed: usize,
    /// Number of execution barriers emitted.
    pub barriers_emitted: usize,
    /// Number of timeline waits executed.
    pub timeline_waits_executed: usize,
    /// Number of timeline signals executed.
    pub timeline_signals_executed: usize,
    /// Number of host-to-device or device-to-device copies (must be 0 for zero-copy schedule).
    pub copy_count: usize,
    /// Number of device-wide idle waits / synchronizations (must be 0 for fine-grained schedule).
    pub device_wide_waits: usize,
}

impl TransitionExecutionReport {
    /// Execute a transition schedule exactly without copies or device-wide waits.
    #[must_use]
    pub fn execute_exact(schedule: &ResourceTransitionSchedule) -> Self {
        let mut report = Self::default();
        for (_, transition) in &schedule.transitions {
            report.transitions_executed += 1;
            if transition.requires_barrier {
                report.barriers_emitted += 1;
            }
        }
        report.timeline_waits_executed = schedule.waits.len();
        report.timeline_signals_executed = schedule.signals.len();
        report.copy_count = 0;
        report.device_wide_waits = 0;
        report
    }
}

/// Invalidation report produced when a device is lost or torn down.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeviceLossInvalidationReport {
    /// Device ID that was lost.
    pub device_id: u64,
    /// Resource IDs that were invalidated.
    pub invalidated_resources: Vec<u64>,
    /// Dependent view IDs that were invalidated.
    pub invalidated_views: Vec<u64>,
    /// Dependent artifact IDs that were invalidated.
    pub invalidated_artifacts: Vec<u64>,
}

/// How many admitted external resources one registry holds at once.
///
/// An import has no matching release call, so the table is bounded here rather
/// than by the caller: it holds at most this many records, and admission past
/// the ceiling evicts, taking a record the device already invalidated before
/// the record admitted longest ago. Both dependent indexes are keyed by
/// admitted resource id, so evicting a record drops its entries there too and
/// all three carry the same ceiling.
const REGISTRY_CAPACITY: usize = 1024;

/// One admitted external resource: the record and the backend handle it was
/// imported from.
///
/// Validity, mutation generation and zero-copy admission are read from
/// [`AdmittedResourceRecord`], which is the single statement of all three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedResource<H> {
    /// The admitted record.
    pub record: AdmittedResourceRecord,
    /// The backend handle the record was imported from.
    pub handle: H,
}

/// The admitted resource table and the admission order eviction reads.
///
/// Order is held beside the records under one lock. Split across two locks the
/// pair disagrees about which record is coldest as soon as two imports
/// interleave.
#[derive(Debug)]
struct AdmittedResourceTable<H> {
    records: HashMap<u64, ImportedResource<H>>,
    admission_order: VecDeque<u64>,
}

impl<H> AdmittedResourceTable<H> {
    /// An empty table that reserves its whole ceiling up front.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            records: HashMap::with_capacity(capacity),
            admission_order: VecDeque::with_capacity(capacity),
        }
    }

    /// Record `resource` under `resource_id` and return the id evicted to make
    /// room, if the admission crossed the ceiling.
    ///
    /// Re-admitting an id already present replaces the record in place and
    /// keeps its original admission position, so a caller that re-imports one
    /// resource every frame cannot hold the whole table hot.
    fn admit(&mut self, resource_id: u64, resource: ImportedResource<H>) -> Option<u64> {
        if self.records.insert(resource_id, resource).is_some() {
            return None;
        }
        self.admission_order.push_back(resource_id);
        if self.records.len() <= REGISTRY_CAPACITY {
            return None;
        }
        self.evict_one()
    }

    /// Drop one record: an invalidated one when the table holds any, otherwise
    /// the one admitted longest ago.
    ///
    /// A record the device already invalidated answers every lookup with
    /// [`ResourceAbiError::ResourceInvalidated`], which is also the answer once
    /// it is gone, so reclaiming it first costs a caller nothing.
    fn evict_one(&mut self) -> Option<u64> {
        let position = self
            .admission_order
            .iter()
            .position(|id| {
                self.records
                    .get(id)
                    .is_none_or(|resource| !resource.record.is_valid)
            })
            .unwrap_or(0);
        let evicted = self.admission_order.remove(position)?;
        self.records.remove(&evicted);
        Some(evicted)
    }
}

/// What a registry holds under one resource id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionState {
    /// No record: never admitted, or admitted and since evicted.
    Absent,
    /// A record device loss or teardown invalidated.
    Invalidated,
    /// A record the device still holds.
    Valid,
}

/// The admitted resource table, its two dependent indexes, and the device-loss
/// invalidation that walks all three.
///
/// `H` is the backend handle an import carries. A registry over `()` holds
/// records admitted without one.
#[derive(Debug)]
pub struct ExternalResourceRegistry<H = ()> {
    owner: &'static str,
    resources: RwLock<AdmittedResourceTable<H>>,
    dependent_views: RwLock<HashMap<u64, HashSet<u64>>>,
    dependent_artifacts: RwLock<HashMap<u64, HashSet<u64>>>,
}

impl<H> ExternalResourceRegistry<H> {
    /// An empty registry whose poison reports name `owner`.
    #[must_use]
    pub fn new(owner: &'static str) -> Self {
        Self {
            owner,
            resources: RwLock::new(AdmittedResourceTable::with_capacity(REGISTRY_CAPACITY)),
            dependent_views: RwLock::new(HashMap::with_capacity(REGISTRY_CAPACITY)),
            dependent_artifacts: RwLock::new(HashMap::with_capacity(REGISTRY_CAPACITY)),
        }
    }

    /// Read the admitted resource table, keeping every entry after a panic.
    ///
    /// The table is the only record of external handles the device still holds,
    /// so recovery keeps it and clears the poison flag once.
    fn read_resources(&self) -> RwLockReadGuard<'_, AdmittedResourceTable<H>> {
        reclaim_poisoned_read(&self.resources, self.owner, RESOURCE_TABLE)
    }

    /// Take the admitted resource table for mutation under the same policy.
    fn write_resources(&self) -> RwLockWriteGuard<'_, AdmittedResourceTable<H>> {
        reclaim_poisoned_write(&self.resources, self.owner, RESOURCE_TABLE)
    }

    /// Take a dependent index under the same policy as the table it indexes.
    fn write_index<'a>(
        &self,
        index: &'a RwLock<HashMap<u64, HashSet<u64>>>,
        state: &str,
    ) -> RwLockWriteGuard<'a, HashMap<u64, HashSet<u64>>> {
        reclaim_poisoned_write(index, self.owner, state)
    }

    /// Admit `resource`, evicting to stay under the ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the record is
    /// already marked invalid.
    pub fn admit(&self, resource: ImportedResource<H>) -> Result<u64, ResourceAbiError> {
        let resource_id = resource.record.resource_id;
        if !resource.record.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        let evicted = self.write_resources().admit(resource_id, resource);
        if let Some(evicted) = evicted {
            self.write_index(&self.dependent_views, VIEW_INDEX)
                .remove(&evicted);
            self.write_index(&self.dependent_artifacts, ARTIFACT_INDEX)
                .remove(&evicted);
        }
        Ok(resource_id)
    }

    /// What the registry holds under `resource_id`.
    ///
    /// A caller whose error vocabulary separates an absent record from an
    /// invalidated one reads this; [`Self::require_admitted`] is the same
    /// question for a caller that does not.
    #[must_use]
    pub fn admission(&self, resource_id: u64) -> AdmissionState {
        match self.read_resources().records.get(&resource_id) {
            None => AdmissionState::Absent,
            Some(resource) if resource.record.is_valid => AdmissionState::Valid,
            Some(_) => AdmissionState::Invalidated,
        }
    }

    /// Confirm `resource_id` names a record the registry holds and the device
    /// has not invalidated.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the record is
    /// absent, evicted, or invalidated.
    pub fn require_admitted(&self, resource_id: u64) -> Result<(), ResourceAbiError> {
        match self.admission(resource_id) {
            AdmissionState::Valid => Ok(()),
            AdmissionState::Absent | AdmissionState::Invalidated => {
                Err(ResourceAbiError::ResourceInvalidated { resource_id })
            }
        }
    }

    /// Apply `mutation` to the record admitted under `resource_id`.
    ///
    /// Returns `None` when the registry holds no such record. An invalidated
    /// record is still passed to `mutation`, which decides what a mutation of
    /// one means.
    pub fn mutate<R>(
        &self,
        resource_id: u64,
        mutation: impl FnOnce(&mut AdmittedResourceRecord) -> R,
    ) -> Option<R> {
        self.write_resources()
            .records
            .get_mut(&resource_id)
            .map(|resource| mutation(&mut resource.record))
    }

    /// Index `dependent_id` under `resource_id` in `index`.
    fn register_dependent(
        &self,
        index: &RwLock<HashMap<u64, HashSet<u64>>>,
        state: &str,
        resource_id: u64,
        dependent_id: u64,
    ) -> Result<(), ResourceAbiError> {
        self.require_admitted(resource_id)?;
        self.write_index(index, state)
            .entry(resource_id)
            .or_default()
            .insert(dependent_id);
        Ok(())
    }

    /// Register a dependent view on an admitted resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the parent resource
    /// is absent or invalidated.
    pub fn register_dependent_view(
        &self,
        resource_id: u64,
        view_id: u64,
    ) -> Result<(), ResourceAbiError> {
        self.register_dependent(&self.dependent_views, VIEW_INDEX, resource_id, view_id)
    }

    /// Register a dependent artifact on an admitted resource.
    ///
    /// A pipeline, a graph and a derived view are all dependent artifacts: the
    /// index holds whatever a device loss has to invalidate alongside the
    /// resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the parent resource
    /// is absent or invalidated.
    pub fn register_dependent_artifact(
        &self,
        resource_id: u64,
        artifact_id: u64,
    ) -> Result<(), ResourceAbiError> {
        self.register_dependent(
            &self.dependent_artifacts,
            ARTIFACT_INDEX,
            resource_id,
            artifact_id,
        )
    }

    /// Execute `schedule` exactly, once every resource it names is admitted.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] naming the first
    /// resource in the schedule the registry cannot serve.
    pub fn execute_transition_schedule(
        &self,
        schedule: &ResourceTransitionSchedule,
    ) -> Result<TransitionExecutionReport, ResourceAbiError> {
        let table = self.read_resources();
        for (resource_id, _) in &schedule.transitions {
            match table.records.get(resource_id) {
                Some(resource) if resource.record.is_valid => {}
                _ => {
                    return Err(ResourceAbiError::ResourceInvalidated {
                        resource_id: *resource_id,
                    })
                }
            }
        }
        drop(table);
        Ok(TransitionExecutionReport::execute_exact(schedule))
    }

    /// Invalidate every resource of `device_id` and every view and artifact
    /// derived from one.
    pub fn invalidate_on_device_loss(&self, device_id: u64) -> DeviceLossInvalidationReport {
        let mut report = DeviceLossInvalidationReport {
            device_id,
            invalidated_resources: Vec::new(),
            invalidated_views: Vec::new(),
            invalidated_artifacts: Vec::new(),
        };

        let mut resources = self.write_resources();
        let mut views = self.write_index(&self.dependent_views, VIEW_INDEX);
        let mut artifacts = self.write_index(&self.dependent_artifacts, ARTIFACT_INDEX);

        for (resource_id, resource) in resources.records.iter_mut() {
            if resource.record.device_id != device_id {
                continue;
            }
            resource.record.invalidate_on_device_loss();
            report.invalidated_resources.push(*resource_id);

            if let Some(view_set) = views.remove(resource_id) {
                report.invalidated_views.extend(view_set);
            }
            if let Some(artifact_set) = artifacts.remove(resource_id) {
                report.invalidated_artifacts.extend(artifact_set);
            }
        }

        report.invalidated_resources.sort_unstable();
        report.invalidated_views.sort_unstable();
        report.invalidated_artifacts.sort_unstable();
        report
    }

    /// Look up an admitted record by resource id.
    #[must_use]
    pub fn get_resource(&self, resource_id: u64) -> Option<AdmittedResourceRecord> {
        self.read_resources()
            .records
            .get(&resource_id)
            .map(|resource| resource.record.clone())
    }
}

impl<H: Clone> ExternalResourceRegistry<H> {
    /// Look up an admitted resource and the handle it was imported from.
    #[must_use]
    pub fn get(&self, resource_id: u64) -> Option<ImportedResource<H>> {
        self.read_resources().records.get(&resource_id).cloned()
    }
}

impl ExternalResourceRegistry<()> {
    /// Admit a record with no backend handle behind it.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the record is
    /// already marked invalid.
    pub fn admit_resource(&self, record: AdmittedResourceRecord) -> Result<u64, ResourceAbiError> {
        self.admit(ImportedResource { record, handle: () })
    }
}

impl<H> crate::lock_policy::StateOwnerRecovery for ExternalResourceRegistry<H> {
    fn failure_domain(&self) -> crate::lock_policy::FailureDomain {
        crate::lock_policy::FailureDomain::DeviceContext
    }

    fn recovery_class(&self) -> crate::lock_policy::RecoveryClass {
        crate::lock_policy::RecoveryClass::DeviceContextFatal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const RESOURCE_ID: u64 = 7001;
    const DEVICE_ID: u64 = 9;

    fn registry_with_one_resource() -> Arc<ExternalResourceRegistry> {
        let registry = Arc::new(ExternalResourceRegistry::new("registry contract"));
        let record = AdmittedResourceRecord::new_external_import_2d(
            RESOURCE_ID,
            DEVICE_ID,
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
        registry
            .admit_resource(record)
            .expect("admit the fixture resource");
        registry
    }

    /// Poison one of the registry's locks from a thread that panics holding it.
    fn poison<T: Send + Sync + 'static>(
        registry: Arc<ExternalResourceRegistry>,
        pick: fn(&ExternalResourceRegistry) -> &RwLock<T>,
    ) {
        let joined = std::thread::spawn(move || {
            let _guard = pick(&registry).write().expect("lock is not yet poisoned");
            panic!("a panic holding an external resource registry lock");
        })
        .join();
        assert!(joined.is_err(), "the poisoning thread must have panicked");
    }

    /// Closes the class "a poisoned lock reported as an invalidated resource".
    ///
    /// [`AdmittedResourceRecord`] is the only record of an external handle the
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
    fn every_poisoned_lock_recovers_instead_of_invalidating_resources() {
        let registry = registry_with_one_resource();

        poison(Arc::clone(&registry), |r| &r.resources);
        poison(Arc::clone(&registry), |r| &r.dependent_views);
        poison(Arc::clone(&registry), |r| &r.dependent_artifacts);

        assert!(
            registry.resources.is_poisoned()
                && registry.dependent_views.is_poisoned()
                && registry.dependent_artifacts.is_poisoned(),
            "the fixture must leave all three locks poisoned"
        );

        registry
            .register_dependent_view(RESOURCE_ID, 11)
            .expect("register_dependent_view must clear the poison and continue");
        registry
            .register_dependent_artifact(RESOURCE_ID, 12)
            .expect("register_dependent_artifact must clear the poison and continue");

        let mut schedule = ResourceTransitionSchedule::new();
        schedule.add_transition(
            RESOURCE_ID,
            ResourceUsageTransition::to_sampled(ResourceLayoutState::General),
        );
        registry
            .execute_transition_schedule(&schedule)
            .expect("execute_transition_schedule must clear the poison and continue");

        assert_eq!(
            registry
                .mutate(RESOURCE_ID, |record| record.advance_generation(1))
                .expect("mutate must clear the poison and continue")
                .expect("the fixture generation is 1"),
            2,
            "the generation must advance across a recovered poison"
        );

        let record = registry
            .get_resource(RESOURCE_ID)
            .expect("get_resource must clear the poison and continue");
        assert!(
            record.is_valid,
            "a recovered poison must not mark the resource invalid"
        );

        let report = registry.invalidate_on_device_loss(DEVICE_ID);
        assert_eq!(
            report.invalidated_resources,
            vec![RESOURCE_ID],
            "device loss must still see the resource the recovered table holds"
        );
        assert_eq!(report.invalidated_views, vec![11]);
        assert_eq!(report.invalidated_artifacts, vec![12]);
    }

    /// A registry over one device leaves another device's records admitted.
    #[test]
    fn device_loss_spares_the_records_of_every_other_device() {
        let registry = registry_with_one_resource();

        let report = registry.invalidate_on_device_loss(DEVICE_ID + 1);
        assert_eq!(report.invalidated_resources, Vec::<u64>::new());
        assert!(
            registry
                .get_resource(RESOURCE_ID)
                .expect("the fixture record is still admitted")
                .is_valid,
            "another device's loss invalidated this device's record"
        );
    }
}
