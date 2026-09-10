//! Concrete WGPU zero-copy external resource import.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! Authentication order, admission, the two dependent indexes, the ceiling
//! every import is bounded by and device-loss invalidation belong to
//! [`vyre_driver::external_import`]. What is WGPU here is the handle an import
//! carries and the combinations authentication refuses.

use vyre_driver::external_import::{
    ExternalImportDescriptor, ExternalImportHandle, ExternalImportPolicy, ExternalResourceImporter,
};
use vyre_driver::{
    ColorInterpretation, ExternalMemoryKind, ResourceAbiError, ResourcePermittedUsages,
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

impl ExternalImportHandle for WgpuExternalMemoryHandle {
    fn memory_kind(&self) -> ExternalMemoryKind {
        match self {
            Self::DmaBuf(_) => ExternalMemoryKind::DmaBuf,
            Self::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            Self::HostBuffer { .. } => ExternalMemoryKind::HostAllocation,
            Self::HalTexture(_) => ExternalMemoryKind::OpaqueFd,
        }
    }

    fn provenance_tag(&self) -> u64 {
        match *self {
            Self::DmaBuf(fd) => fd as u64,
            Self::Win32Nt(handle) => handle as u64,
            Self::HostBuffer { ptr, .. } => ptr as u64,
            Self::HalTexture(tex) => tex as u64,
        }
    }
}

/// What WGPU states about an external import beyond the neutral contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuExternalImportPolicy;

impl ExternalImportPolicy for WgpuExternalImportPolicy {
    type Handle = WgpuExternalMemoryHandle;

    const OWNER: &'static str = "wgpu backend external resource registry";

    const COLOR: ColorInterpretation = ColorInterpretation::Srgb;

    fn refuse_unsupported(
        descriptor: &ExternalImportDescriptor<Self::Handle>,
    ) -> Result<(), ResourceAbiError> {
        // A depth/stencil surface has no storage-write binding.
        if descriptor.format.is_depth_stencil()
            && descriptor
                .permitted_usages
                .contains(ResourcePermittedUsages::STORAGE_WRITE)
        {
            return Err(descriptor.unsupported_combination());
        }
        Ok(())
    }
}

/// Descriptor for importing external memory into the WGPU driver.
pub type WgpuExternalMemoryDescriptor = ExternalImportDescriptor<WgpuExternalMemoryHandle>;

/// WGPU concrete external resource importer and synchronization engine.
pub type WgpuExternalResourceImporter = ExternalResourceImporter<WgpuExternalImportPolicy>;
