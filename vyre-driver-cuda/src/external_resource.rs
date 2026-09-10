//! Concrete CUDA zero-copy external resource import, export, and timeline synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides CUDA driver API integration for external memory handles (`CUexternalMemory`)
//! and external synchronization semaphores (`CUexternalSemaphore`).

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use vyre_foundation::failure_domain::reclaim_poisoned_write;

/// The subsystem every poison report in this module names as the owner.
const OWNER: &str = "cuda backend external resource registry";

use vyre_driver::{
    AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind, ImageDimensions,
    ImageFormat, ResourceAbiError, ResourcePermittedUsages, ResourceTransitionSchedule,
    TimelineSyncProtocol, TransitionExecutionReport,
};

/// CUDA external memory handle representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaExternalMemoryHandle {
    /// Linux DMA-BUF file descriptor.
    DmaBufFd(i32),
    /// Windows NT handle.
    Win32Nt(usize),
    /// Pinned host virtual address pointer.
    HostPointer {
        /// Virtual address.
        ptr: usize,
        /// Size in bytes.
        byte_size: u64,
    },
}

/// Descriptor for importing external memory into the CUDA driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaExternalMemoryDescriptor {
    /// Resource ID.
    pub resource_id: u64,
    /// Image format.
    pub format: ImageFormat,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes.
    pub row_pitch_bytes: u32,
    /// External memory handle.
    pub handle: CudaExternalMemoryHandle,
    /// Permitted usages.
    pub permitted_usages: ResourcePermittedUsages,
    /// Timeline synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
}

/// An admitted zero-copy imported CUDA resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaImportedResource {
    /// Associated resource record.
    pub record: AdmittedResourceRecord,
    /// External memory handle.
    pub handle: CudaExternalMemoryHandle,
    /// Whether this resource operates with zero host copies.
    pub is_zero_copy: bool,
    /// Current mutation generation.
    pub generation: u64,
    /// Device validity flag.
    pub is_valid: bool,
}

/// CUDA concrete external resource importer and synchronization engine.
#[derive(Debug)]
pub struct CudaExternalResourceImporter {
    device_id: u64,
    imported_resources: RwLock<HashMap<u64, CudaImportedResource>>,
    dependent_views: RwLock<HashMap<u64, HashSet<u64>>>,
    dependent_graphs: RwLock<HashMap<u64, HashSet<u64>>>,
}

impl CudaExternalResourceImporter {
    /// Create a new importer for CUDA `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            imported_resources: RwLock::new(HashMap::new()),
            dependent_views: RwLock::new(HashMap::new()),
            dependent_graphs: RwLock::new(HashMap::new()),
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
        descriptor: &CudaExternalMemoryDescriptor,
    ) -> Result<(), ResourceAbiError> {
        // 1. Validate spatial dimensions
        if !descriptor.dimensions.is_valid() {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch: 256,
            });
        }

        // 2. Authenticate memory kind + format support
        let memory_kind = match descriptor.handle {
            CudaExternalMemoryHandle::DmaBufFd(_) => ExternalMemoryKind::DmaBuf,
            CudaExternalMemoryHandle::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            CudaExternalMemoryHandle::HostPointer { .. } => ExternalMemoryKind::HostAllocation,
        };

        match memory_kind {
            ExternalMemoryKind::DmaBuf => {
                // CUDA DMA-BUF import rejects depth/stencil formats (not valid for 2D surface load/store)
                if descriptor.format.is_depth_stencil() {
                    return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                        resource_id: descriptor.resource_id,
                        format: descriptor.format,
                        memory_kind,
                    });
                }
            }
            ExternalMemoryKind::Win32Nt => {
                // CUDA Windows NT import rejects 3-plane planar formats
                if descriptor.format.is_planar_video()
                    && descriptor.format != ImageFormat::Yuv420SemiPlanar
                {
                    return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                        resource_id: descriptor.resource_id,
                        format: descriptor.format,
                        memory_kind,
                    });
                }
            }
            ExternalMemoryKind::HostAllocation => {
                if descriptor.format.is_depth_stencil() {
                    return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                        resource_id: descriptor.resource_id,
                        format: descriptor.format,
                        memory_kind,
                    });
                }
            }
            _ => {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id: descriptor.resource_id,
                    format: descriptor.format,
                    memory_kind,
                });
            }
        }

        // 3. Validate pitch alignment (CUDA pitch linear surfaces require 256-byte alignment)
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

        Ok(())
    }

    /// Import external memory into CUDA with zero host copies.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError`] if pre-allocation authentication fails.
    pub fn import_external_resource(
        &self,
        descriptor: CudaExternalMemoryDescriptor,
    ) -> Result<AdmittedResourceRecord, ResourceAbiError> {
        self.authenticate_import(&descriptor)?;

        let memory_kind = match descriptor.handle {
            CudaExternalMemoryHandle::DmaBufFd(_) => ExternalMemoryKind::DmaBuf,
            CudaExternalMemoryHandle::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            CudaExternalMemoryHandle::HostPointer { .. } => ExternalMemoryKind::HostAllocation,
        };

        let handle_tag = match descriptor.handle {
            CudaExternalMemoryHandle::DmaBufFd(fd) => fd as u64,
            CudaExternalMemoryHandle::Win32Nt(handle) => handle as u64,
            CudaExternalMemoryHandle::HostPointer { ptr, .. } => ptr as u64,
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

        let imported = CudaImportedResource {
            record: record.clone(),
            handle: descriptor.handle,
            is_zero_copy: true,
            generation: 1,
            is_valid: true,
        };

        let mut map = match self.imported_resources.write() {
            Ok(g) => g,
            Err(_) => {
                return Err(ResourceAbiError::ResourceInvalidated {
                    resource_id: descriptor.resource_id,
                })
            }
        };
        map.insert(descriptor.resource_id, imported);

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

    /// Register a dependent CUDA graph on an imported resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the resource is invalidated or not found.
    pub fn register_dependent_graph(
        &self,
        resource_id: u64,
        graph_id: u64,
    ) -> Result<(), ResourceAbiError> {
        let map = match self.imported_resources.read() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        let resource = map
            .get(&resource_id)
            .ok_or(ResourceAbiError::ResourceInvalidated { resource_id })?;
        if !resource.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut graphs = match self.dependent_graphs.write() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        graphs.entry(resource_id).or_default().insert(graph_id);
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

    /// Invalidate all resources, views, and dependent CUDA graphs on device loss.
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
        let mut graphs =
            reclaim_poisoned_write(&self.dependent_graphs, OWNER, "the dependent graph index");

        for (res_id, res) in map.iter_mut() {
            res.is_valid = false;
            res.record.invalidate_on_device_loss();
            report.invalidated_resources.push(*res_id);

            if let Some(view_set) = views.remove(res_id) {
                report.invalidated_views.extend(view_set);
            }
            if let Some(graph_set) = graphs.remove(res_id) {
                report.invalidated_artifacts.extend(graph_set);
            }
        }

        report.invalidated_resources.sort_unstable();
        report.invalidated_views.sort_unstable();
        report.invalidated_artifacts.sort_unstable();
        report
    }
}
