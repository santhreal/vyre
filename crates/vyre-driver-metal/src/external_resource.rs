//! Concrete Metal zero-copy external resource import.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! Authentication order, admission, the two dependent indexes, the ceiling
//! every import is bounded by and device-loss invalidation belong to
//! [`vyre_driver::external_import`]. What is Metal here is the handle an import
//! carries and the combinations authentication refuses.

use vyre_driver::external_import::{
    ExternalImportDescriptor, ExternalImportHandle, ExternalImportPolicy, ExternalResourceImporter,
};
use vyre_driver::{ColorInterpretation, ExternalMemoryKind, ResourceAbiError};

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

impl ExternalImportHandle for MetalExternalMemoryHandle {
    /// Every Metal handle is imported as one shared-resource class: IOSurface,
    /// shared buffer and shared texture are the same `MTLResource` sharing
    /// mechanism seen through three descriptors.
    fn memory_kind(&self) -> ExternalMemoryKind {
        ExternalMemoryKind::MetalSharedResource
    }

    fn provenance_tag(&self) -> u64 {
        match *self {
            Self::IOSurface(surface) => surface as u64,
            Self::SharedBuffer { ptr, .. } => ptr as u64,
            Self::SharedTexture(tex) => tex as u64,
        }
    }
}

/// What Metal states about an external import beyond the neutral contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalExternalImportPolicy;

impl ExternalImportPolicy for MetalExternalImportPolicy {
    type Handle = MetalExternalMemoryHandle;

    const OWNER: &'static str = "metal backend external resource registry";

    const COLOR: ColorInterpretation = ColorInterpretation::DisplayP3;

    fn refuse_unsupported(
        descriptor: &ExternalImportDescriptor<Self::Handle>,
    ) -> Result<(), ResourceAbiError> {
        // A Metal shared texture has no multi-planar YUV form.
        if descriptor.format.is_planar_video() {
            return Err(descriptor.unsupported_combination());
        }
        Ok(())
    }
}

/// Descriptor for importing external memory into the Metal driver.
pub type MetalExternalMemoryDescriptor = ExternalImportDescriptor<MetalExternalMemoryHandle>;

/// Metal concrete external resource importer and synchronization engine.
pub type MetalExternalResourceImporter = ExternalResourceImporter<MetalExternalImportPolicy>;
