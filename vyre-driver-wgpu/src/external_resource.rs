//! Concrete WGPU zero-copy external resource import, export, and timeline synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides WGPU HAL and external memory handle import, texture view derivation,
//! and timeline synchronization across presentation and compute boundaries.
//!
//! Admission, the two dependent indexes, the ceiling every import is bounded
//! by and device-loss invalidation belong to
//! [`ExternalResourceRegistry`]. What is WGPU here is the handle an import
//! carries and the combinations authentication rejects.

use vyre_driver::{
    AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind,
    ExternalResourceRegistry, ImageDimensions, ImageFormat, ImportedResource, ResourceAbiError,
    ResourcePermittedUsages, TimelineSyncProtocol,
};

/// The subsystem every poison report from this registry names as the owner.
const OWNER: &str = "wgpu backend external resource registry";

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

impl WgpuExternalMemoryHandle {
    /// The external memory class this handle is imported as.
    fn memory_kind(&self) -> ExternalMemoryKind {
        match self {
            Self::DmaBuf(_) => ExternalMemoryKind::DmaBuf,
            Self::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            Self::HostBuffer { .. } => ExternalMemoryKind::HostAllocation,
            Self::HalTexture(_) => ExternalMemoryKind::OpaqueFd,
        }
    }

    /// The opaque address the record carries as this handle's provenance.
    fn tag(&self) -> u64 {
        match *self {
            Self::DmaBuf(fd) => fd as u64,
            Self::Win32Nt(handle) => handle as u64,
            Self::HostBuffer { ptr, .. } => ptr as u64,
            Self::HalTexture(tex) => tex as u64,
        }
    }
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

/// WGPU concrete external resource importer and synchronization engine.
#[derive(Debug)]
pub struct WgpuExternalResourceImporter {
    device_id: u64,
    registry: ExternalResourceRegistry<WgpuExternalMemoryHandle>,
}

impl WgpuExternalResourceImporter {
    /// Create a new importer for WGPU `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            registry: ExternalResourceRegistry::new(OWNER),
        }
    }

    /// The admitted resources of this device, their dependent views and their
    /// dependent artifacts.
    #[must_use]
    pub fn registry(&self) -> &ExternalResourceRegistry<WgpuExternalMemoryHandle> {
        &self.registry
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
            return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                resource_id: descriptor.resource_id,
                format: descriptor.format,
                memory_kind: descriptor.handle.memory_kind(),
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

        let record = AdmittedResourceRecord::new_external_import_2d(
            descriptor.resource_id,
            self.device_id,
            descriptor.format,
            vyre_driver::ColorInterpretation::Srgb,
            descriptor.dimensions.width,
            descriptor.dimensions.height,
            descriptor.row_pitch_bytes,
            descriptor.permitted_usages,
            descriptor.handle.memory_kind(),
            descriptor.handle.tag(),
            descriptor.sync_protocol,
        );

        self.registry.admit(ImportedResource {
            record: record.clone(),
            handle: descriptor.handle,
        })?;

        Ok(record)
    }

    /// Invalidate every resource of this device, and every view and artifact
    /// derived from one.
    pub fn invalidate_on_device_loss(&self) -> DeviceLossInvalidationReport {
        self.registry.invalidate_on_device_loss(self.device_id)
    }
}
