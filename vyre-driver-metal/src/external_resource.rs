//! Concrete Metal zero-copy external resource import, export, and shared event synchronization.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module provides Metal driver integration for IOSurface shared textures, shared buffers,
//! and `MTLSharedEvent` timeline synchronization.
//!
//! Admission, the two dependent indexes, the ceiling every import is bounded
//! by and device-loss invalidation belong to
//! [`ExternalResourceRegistry`]. What is Metal here is the handle an import
//! carries and the combinations authentication rejects.

use vyre_driver::{
    AdmittedResourceRecord, DeviceLossInvalidationReport, ExternalMemoryKind,
    ExternalResourceRegistry, ImageDimensions, ImageFormat, ImportedResource, ResourceAbiError,
    ResourcePermittedUsages, TimelineSyncProtocol,
};

/// The subsystem every poison report from this registry names as the owner.
const OWNER: &str = "metal backend external resource registry";

/// Metal external memory handle representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalExternalMemoryHandle {
    /// Apple IOSurface handle.
    IOSurface(usize),
    /// Metal shared buffer.
    SharedBuffer {
        /// Virtual address.
        ptr: usize,
        /// Size in bytes.
        byte_size: u64,
    },
    /// Metal shared texture reference.
    SharedTexture(usize),
}

/// Descriptor for importing external memory into the Metal driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalExternalMemoryDescriptor {
    /// Resource ID.
    pub resource_id: u64,
    /// Image format.
    pub format: ImageFormat,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes.
    pub row_pitch_bytes: u32,
    /// External memory handle.
    pub handle: MetalExternalMemoryHandle,
    /// Permitted usages.
    pub permitted_usages: ResourcePermittedUsages,
    /// Timeline synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
}

/// Metal concrete external resource importer and synchronization engine.
#[derive(Debug)]
pub struct MetalExternalResourceImporter {
    device_id: u64,
    registry: ExternalResourceRegistry<MetalExternalMemoryHandle>,
}

impl MetalExternalResourceImporter {
    /// Create a new importer for Metal `device_id`.
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
    pub fn registry(&self) -> &ExternalResourceRegistry<MetalExternalMemoryHandle> {
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
        descriptor: &MetalExternalMemoryDescriptor,
    ) -> Result<(), ResourceAbiError> {
        // 1. Validate spatial dimensions
        if !descriptor.dimensions.is_valid() {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch: 256,
            });
        }

        // 2. Authenticate Metal format support (Metal shared textures reject multi-planar YUV)
        if descriptor.format.is_planar_video() {
            return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                resource_id: descriptor.resource_id,
                format: descriptor.format,
                memory_kind: ExternalMemoryKind::MetalSharedResource,
            });
        }

        // 3. Validate pitch alignment (Metal texture buffer row alignment requires 256-byte boundary)
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

    /// Import external memory into Metal with zero host copies.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError`] if pre-allocation authentication fails.
    pub fn import_external_resource(
        &self,
        descriptor: MetalExternalMemoryDescriptor,
    ) -> Result<AdmittedResourceRecord, ResourceAbiError> {
        self.authenticate_import(&descriptor)?;

        let handle_tag = match descriptor.handle {
            MetalExternalMemoryHandle::IOSurface(surface) => surface as u64,
            MetalExternalMemoryHandle::SharedBuffer { ptr, .. } => ptr as u64,
            MetalExternalMemoryHandle::SharedTexture(tex) => tex as u64,
        };

        let record = AdmittedResourceRecord::new_external_import_2d(
            descriptor.resource_id,
            self.device_id,
            descriptor.format,
            vyre_driver::ColorInterpretation::DisplayP3,
            descriptor.dimensions.width,
            descriptor.dimensions.height,
            descriptor.row_pitch_bytes,
            descriptor.permitted_usages,
            ExternalMemoryKind::MetalSharedResource,
            handle_tag,
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
