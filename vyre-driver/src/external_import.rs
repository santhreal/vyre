//! Backend-neutral external resource import.
//!
//! Three concrete drivers admit an external memory handle the same way: they
//! validate the spatial dimensions, refuse the format and usage combinations
//! the backend cannot import, validate the row pitch, build an admitted record
//! and hand it to [`ExternalResourceRegistry`]. Only the middle step is
//! backend business, so only that step is a concrete driver's to state.
//!
//! A driver supplies an [`ExternalImportPolicy`]: the handle type, how a handle
//! maps to an external memory class and a provenance tag, the color
//! interpretation its records carry, and which combinations it refuses.
//! [`ExternalResourceImporter`] owns everything else, including the order the
//! three checks answer in, which is one order for every backend.

use crate::semantic_resource_abi::{
    AdmittedResourceRecord, ColorInterpretation, DeviceLossInvalidationReport, ExternalMemoryKind,
    ExternalResourceRegistry, ImageDimensions, ImageFormat, ImportedResource, ResourceAbiError,
    ResourcePermittedUsages, TimelineSyncProtocol,
};

/// The pitch every external import is aligned to, in bytes.
///
/// Every target this workspace imports into requires a row of a pitch-linear
/// surface to start on this boundary, so the value is stated once here rather
/// than once per driver.
const PITCH_ALIGNMENT: u32 = 256;

/// What a backend handle states about the import it stands for.
pub trait ExternalImportHandle {
    /// The external memory class this handle is imported as.
    fn memory_kind(&self) -> ExternalMemoryKind;

    /// The opaque address the admitted record carries as this handle's
    /// provenance.
    fn provenance_tag(&self) -> u64;
}

/// One external import request, before any allocation happens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalImportDescriptor<H> {
    /// Resource ID.
    pub resource_id: u64,
    /// Image format.
    pub format: ImageFormat,
    /// Spatial dimensions.
    pub dimensions: ImageDimensions,
    /// Row pitch in bytes.
    pub row_pitch_bytes: u32,
    /// External memory handle.
    pub handle: H,
    /// Permitted usages.
    pub permitted_usages: ResourcePermittedUsages,
    /// Timeline synchronization protocol.
    pub sync_protocol: TimelineSyncProtocol,
}

impl<H: ExternalImportHandle> ExternalImportDescriptor<H> {
    /// The refusal naming this descriptor's combination as unsupported.
    ///
    /// Every backend policy answers a combination it cannot import with this,
    /// so the three fields a caller reads to identify the refused import are
    /// filled in one place.
    #[must_use]
    pub fn unsupported_combination(&self) -> ResourceAbiError {
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: self.resource_id,
            format: self.format,
            memory_kind: self.handle.memory_kind(),
        }
    }
}

/// The backend half of external resource import.
pub trait ExternalImportPolicy {
    /// The handle an import of this backend carries.
    type Handle: ExternalImportHandle;

    /// The subsystem every poison report from this backend's registry names as
    /// the owner.
    const OWNER: &'static str;

    /// The color interpretation this backend's admitted records carry.
    const COLOR: ColorInterpretation;

    /// The external memory class the admitted record carries.
    ///
    /// The default reads the handle. A backend that imports every handle as one
    /// class overrides this and states that class instead.
    fn admitted_memory_kind(handle: &Self::Handle) -> ExternalMemoryKind {
        handle.memory_kind()
    }

    /// Refuse a format, usage and handle combination this backend cannot
    /// import.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] naming the
    /// refused combination.
    fn refuse_unsupported(
        descriptor: &ExternalImportDescriptor<Self::Handle>,
    ) -> Result<(), ResourceAbiError>;
}

/// Bounded external resource import for one device of one backend.
///
/// Admission, the two dependent indexes, the ceiling every import is bounded by
/// and device-loss invalidation belong to [`ExternalResourceRegistry`].
#[derive(Debug)]
pub struct ExternalResourceImporter<P: ExternalImportPolicy> {
    device_id: u64,
    registry: ExternalResourceRegistry<P::Handle>,
}

impl<P: ExternalImportPolicy> ExternalResourceImporter<P> {
    /// Create a new importer for `device_id`.
    #[must_use]
    pub fn new(device_id: u64) -> Self {
        Self {
            device_id,
            registry: ExternalResourceRegistry::new(P::OWNER),
        }
    }

    /// The device this importer admits resources for.
    #[must_use]
    pub fn device_id(&self) -> u64 {
        self.device_id
    }

    /// The admitted resources of this device, their dependent views and their
    /// dependent artifacts.
    #[must_use]
    pub fn registry(&self) -> &ExternalResourceRegistry<P::Handle> {
        &self.registry
    }

    /// Pre-allocation capability check: reject an import before any allocation.
    ///
    /// The three checks answer in one order for every backend: dimensions,
    /// then the combination the backend refuses, then pitch. A caller told the
    /// combination is unsupported stops; a caller told only the pitch is wrong
    /// aligns it and is refused a second time for the reason that was true all
    /// along.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] if the
    /// combination is unsupported, or [`ResourceAbiError::InvalidDimensionsOrPitch`]
    /// if the dimensions are degenerate or the pitch is short or unaligned.
    pub fn authenticate_import(
        &self,
        descriptor: &ExternalImportDescriptor<P::Handle>,
    ) -> Result<(), ResourceAbiError> {
        if !descriptor.dimensions.is_valid() {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch: PITCH_ALIGNMENT,
            });
        }

        P::refuse_unsupported(descriptor)?;

        let min_pitch = descriptor
            .dimensions
            .width
            .saturating_mul(descriptor.format.bytes_per_pixel());
        if descriptor.row_pitch_bytes < min_pitch
            || (descriptor.row_pitch_bytes % PITCH_ALIGNMENT) != 0
        {
            return Err(ResourceAbiError::InvalidDimensionsOrPitch {
                resource_id: descriptor.resource_id,
                provided_pitch: descriptor.row_pitch_bytes,
                required_pitch: min_pitch.next_multiple_of(PITCH_ALIGNMENT),
            });
        }

        Ok(())
    }

    /// Import external memory with zero host copies.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError`] if pre-allocation authentication fails, or
    /// if the registry refuses the record.
    pub fn import_external_resource(
        &self,
        descriptor: ExternalImportDescriptor<P::Handle>,
    ) -> Result<AdmittedResourceRecord, ResourceAbiError> {
        self.authenticate_import(&descriptor)?;

        let record = AdmittedResourceRecord::new_external_import_2d(
            descriptor.resource_id,
            self.device_id,
            descriptor.format,
            P::COLOR,
            descriptor.dimensions.width,
            descriptor.dimensions.height,
            descriptor.row_pitch_bytes,
            descriptor.permitted_usages,
            P::admitted_memory_kind(&descriptor.handle),
            descriptor.handle.provenance_tag(),
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
