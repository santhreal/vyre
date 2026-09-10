//! Concrete WGPU zero-copy external resource import, export, and timeline synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides WGPU HAL and external memory handle import, texture view derivation,
//! and timeline synchronization across presentation and compute boundaries.
//!
//! An import has no matching release call, so the imported-resource table is
//! bounded here rather than by the caller: it holds at most
//! [`REGISTRY_CAPACITY`] records, and admission past that ceiling evicts,
//! taking a record the device already invalidated before the record admitted
//! longest ago. The two dependent indexes are keyed by admitted resource id,
//! so evicting a record drops its entries there too and both indexes carry the
//! same ceiling.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::RwLock;

use vyre_foundation::failure_domain::reclaim_poisoned_write;

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "wgpu backend external resource registry";

/// How many imported external resources one device registry holds at once.
///
/// Every record names a live dma-buf descriptor, NT handle, pinned host
/// allocation or HAL texture. The ceiling covers a swapchain, the per-frame
/// imports a compositor drives through it, and a wide margin above both.
const REGISTRY_CAPACITY: usize = 1024;

use vyre_driver::{
    AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind, ImageDimensions,
    ImageFormat, ResourceAbiError, ResourcePermittedUsages, ResourceTransitionSchedule,
    TimelineSyncProtocol, TransitionExecutionReport,
};

/// WGPU external memory handle representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WgpuExternalMemoryHandle {
    /// Linux DMA-BUF file descriptor.
    DmaBuf(i32),
    /// Windows NT shared handle.
    Win32Nt(usize),
    /// Host pinned buffer allocation.
    HostBuffer {
        /// Virtual address.
        ptr: usize,
        /// Size in bytes.
        byte_size: u64,
    },
    /// WGPU HAL external texture handle.
    HalTexture(usize),
}

/// Descriptor for importing external memory into the WGPU driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuExternalMemoryDescriptor {
    /// Resource ID.
    pub resource_id: u64,
    /// Image format.
    pub format: ImageFormat,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes.
    pub row_pitch_bytes: u32,
    /// External memory handle.
    pub handle: WgpuExternalMemoryHandle,
    /// Permitted usages.
    pub permitted_usages: ResourcePermittedUsages,
    /// Timeline synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
}

/// An admitted zero-copy imported WGPU resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuImportedResource {
    /// Associated resource record.
    pub record: AdmittedResourceRecord,
    /// External memory handle.
    pub handle: WgpuExternalMemoryHandle,
    /// Whether this resource operates with zero host copies.
    pub is_zero_copy: bool,
    /// Current mutation generation.
    pub generation: u64,
    /// Device validity flag.
    pub is_valid: bool,
}

/// The imported-resource table and the admission order eviction reads.
///
/// Order is held beside the records under one lock. Split across two locks the
/// pair disagrees about which record is coldest as soon as two imports
/// interleave.
#[derive(Debug)]
struct ImportedResourceTable {
    records: HashMap<u64, WgpuImportedResource>,
    admission_order: VecDeque<u64>,
}

impl ImportedResourceTable {
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
    /// Re-importing an id already present replaces the record in place and
    /// keeps its original admission position, so a caller that re-imports one
    /// resource every frame cannot hold the whole table hot.
    fn admit(&mut self, resource_id: u64, resource: WgpuImportedResource) -> Option<u64> {
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
            .position(|id| self.records.get(id).is_none_or(|record| !record.is_valid))
            .unwrap_or(0);
        let evicted = self.admission_order.remove(position)?;
        self.records.remove(&evicted);
        Some(evicted)
    }
}

/// WGPU concrete external resource importer and synchronization engine.
#[derive(Debug)]
pub struct WgpuExternalResourceImporter {
    device_id: u64,
    imported_resources: RwLock<ImportedResourceTable>,
    dependent_views: RwLock<HashMap<u64, HashSet<u64>>>,
    dependent_pipelines: RwLock<HashMap<u64, HashSet<u64>>>,
}

impl WgpuExternalResourceImporter {
    /// Create a new importer for WGPU `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            imported_resources: RwLock::new(ImportedResourceTable::with_capacity(
                REGISTRY_CAPACITY,
            )),
            dependent_views: RwLock::new(HashMap::with_capacity(REGISTRY_CAPACITY)),
            dependent_pipelines: RwLock::new(HashMap::with_capacity(REGISTRY_CAPACITY)),
        }
    }

    /// Pre-allocation capability check: reject unsupported combinations before any allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] if the combination is unsupported,
    /// or [`ResourceAbiError::InvalidDimensionsOrPitch`] if pitch is unaligned.
    pub fn authenticate_import(
        &self,
        descriptor: &WgpuExternalMemoryDescriptor,
    ) -> Result<(), ResourceAbiError> {
        // 1. Validate spatial dimensions
        if !descriptor.dimensions.is_valid() {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch: 256,
            });
        }

        // 2. Validate pitch alignment (WGPU COPY_BYTES_PER_ROW_ALIGNMENT requires 256-byte boundary)
        let min_pitch = descriptor
            .dimensions
            .width
            .saturating_mul(descriptor.format.bytes_per_pixel());
        let required_pitch = (min_pitch + 255) & !255;
        if descriptor.row_pitch_bytes < min_pitch || (descriptor.row_pitch_bytes & 255) != 0 {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch,
            });
        }

        // 3. Authenticate format and usage compatibility (DepthStencil cannot be bound as STORAGE_WRITE)
        if descriptor.format.is_depth_stencil()
            && descriptor
                .permitted_usages
                .contains(ResourcePermittedUsages::STORAGE_WRITE)
        {
            let memory_kind = match descriptor.handle {
                WgpuExternalMemoryHandle::DmaBuf(_) => ExternalMemoryKind::DmaBuf,
                WgpuExternalMemoryHandle::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
                WgpuExternalMemoryHandle::HostBuffer { .. } => ExternalMemoryKind::HostAllocation,
                WgpuExternalMemoryHandle::HalTexture(_) => ExternalMemoryKind::OpaqueFd,
            };
            return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                resource_id: descriptor.resource_id,
                format: descriptor.format,
                memory_kind,
            });
        }

        Ok(())
    }

    /// Import external memory into WGPU with zero host copies.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError`] if pre-allocation authentication fails.
    pub fn import_external_resource(
        &self,
        descriptor: WgpuExternalMemoryDescriptor,
    ) -> Result<AdmittedResourceRecord, ResourceAbiError> {
        self.authenticate_import(&descriptor)?;

        let memory_kind = match descriptor.handle {
            WgpuExternalMemoryHandle::DmaBuf(_) => ExternalMemoryKind::DmaBuf,
            WgpuExternalMemoryHandle::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            WgpuExternalMemoryHandle::HostBuffer { .. } => ExternalMemoryKind::HostAllocation,
            WgpuExternalMemoryHandle::HalTexture(_) => ExternalMemoryKind::OpaqueFd,
        };

        let handle_tag = match descriptor.handle {
            WgpuExternalMemoryHandle::DmaBuf(fd) => fd as u64,
            WgpuExternalMemoryHandle::Win32Nt(handle) => handle as u64,
            WgpuExternalMemoryHandle::HostBuffer { ptr, .. } => ptr as u64,
            WgpuExternalMemoryHandle::HalTexture(tex) => tex as u64,
        };

        let record = AdmittedResourceRecord::new_external_import_2d(
            descriptor.resource_id,
            self.device_id,
            descriptor.format,
            vyre_driver::ColorInterpretation::Srgb,
            descriptor.dimensions.width,
            descriptor.dimensions.height,
            descriptor.row_pitch_bytes,
            descriptor.permitted_usages,
            memory_kind,
            handle_tag,
            descriptor.sync_protocol,
        );

        let imported = WgpuImportedResource {
            record: record.clone(),
            handle: descriptor.handle,
            is_zero_copy: true,
            generation: 1,
            is_valid: true,
        };

        let evicted = {
            let mut table = match self.imported_resources.write() {
                Ok(g) => g,
                Err(_) => {
                    return Err(ResourceAbiError::ResourceInvalidated {
                        resource_id: descriptor.resource_id,
                    })
                }
            };
            table.admit(descriptor.resource_id, imported)
        };
        if let Some(evicted) = evicted {
            reclaim_poisoned_write(&self.dependent_views, OWNER, "the dependent view index")
                .remove(&evicted);
            reclaim_poisoned_write(
                &self.dependent_pipelines,
                OWNER,
                "the dependent pipeline index",
            )
            .remove(&evicted);
        }

        Ok(record)
    }

    /// Register a dependent view on an imported resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the resource is invalidated or not found.
    pub fn register_dependent_view(
        &self,
        resource_id: u64,
        view_id: u64,
    ) -> Result<(), ResourceAbiError> {
        let map = match self.imported_resources.read() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        let resource = map
            .records
            .get(&resource_id)
            .ok_or(ResourceAbiError::ResourceInvalidated { resource_id })?;
        if !resource.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut views = match self.dependent_views.write() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        views.entry(resource_id).or_default().insert(view_id);
        Ok(())
    }

    /// Register a dependent pipeline on an imported resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the resource is invalidated or not found.
    pub fn register_dependent_pipeline(
        &self,
        resource_id: u64,
        pipeline_id: u64,
    ) -> Result<(), ResourceAbiError> {
        let map = match self.imported_resources.read() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        let resource = map
            .records
            .get(&resource_id)
            .ok_or(ResourceAbiError::ResourceInvalidated { resource_id })?;
        if !resource.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut pipelines = match self.dependent_pipelines.write() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        pipelines
            .entry(resource_id)
            .or_default()
            .insert(pipeline_id);
        Ok(())
    }

    /// Execute a transition schedule with exact timeline semaphore waits/signals.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if any resource is invalid.
    pub fn execute_transition_schedule(
        &self,
        schedule: &ResourceTransitionSchedule,
    ) -> Result<TransitionExecutionReport, ResourceAbiError> {
        let map = match self.imported_resources.read() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id: 0 }),
        };
        for (resource_id, _) in &schedule.transitions {
            let res = map
                .records
                .get(resource_id)
                .ok_or(ResourceAbiError::ResourceInvalidated {
                    resource_id: *resource_id,
                })?;
            if !res.is_valid {
                return Err(ResourceAbiError::ResourceInvalidated {
                    resource_id: *resource_id,
                });
            }
        }

        let report = TransitionExecutionReport::execute_exact(schedule);
        Ok(report)
    }

    /// Invalidate all resources, views, and dependent pipelines on device loss.
    pub fn invalidate_on_device_loss(&self) -> DeviceLossInvalidationReport {
        let mut report = DeviceLossInvalidationReport {
            device_id: self.device_id,
            invalidated_resources: Vec::new(),
            invalidated_views: Vec::new(),
            invalidated_artifacts: Vec::new(),
        };

        let mut map = reclaim_poisoned_write(
            &self.imported_resources,
            OWNER,
            "the imported external resource table",
        );
        let mut views =
            reclaim_poisoned_write(&self.dependent_views, OWNER, "the dependent view index");
        let mut pipelines = reclaim_poisoned_write(
            &self.dependent_pipelines,
            OWNER,
            "the dependent pipeline index",
        );

        for (res_id, res) in map.records.iter_mut() {
            res.is_valid = false;
            res.record.invalidate_on_device_loss();
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
}
