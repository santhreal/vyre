//! Concrete CUDA zero-copy external resource import.
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! Authentication order, admission, the two dependent indexes, the ceiling
//! every import is bounded by and device-loss invalidation belong to
//! [`vyre_driver::external_import`]. What is CUDA here is the handle an import
//! carries and the combinations authentication refuses.

use vyre_driver::external_import::{
    ExternalImportDescriptor, ExternalImportHandle, ExternalImportPolicy, ExternalResourceImporter,
};
use vyre_driver::{ColorInterpretation, ExternalMemoryKind, ImageFormat, ResourceAbiError};

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

impl ExternalImportHandle for CudaExternalMemoryHandle {
    fn memory_kind(&self) -> ExternalMemoryKind {
        match self {
            Self::DmaBufFd(_) => ExternalMemoryKind::DmaBuf,
            Self::Win32Nt(_) => ExternalMemoryKind::Win32Nt,
            Self::HostPointer { .. } => ExternalMemoryKind::HostAllocation,
        }
    }

    fn provenance_tag(&self) -> u64 {
        match *self {
            Self::DmaBufFd(fd) => fd as u64,
            Self::Win32Nt(handle) => handle as u64,
            Self::HostPointer { ptr, .. } => ptr as u64,
        }
    }
}

/// What CUDA states about an external import beyond the neutral contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaExternalImportPolicy;

impl ExternalImportPolicy for CudaExternalImportPolicy {
    type Handle = CudaExternalMemoryHandle;

    const OWNER: &'static str = "cuda backend external resource registry";

    const COLOR: ColorInterpretation = ColorInterpretation::Srgb;

    fn refuse_unsupported(
        descriptor: &ExternalImportDescriptor<Self::Handle>,
    ) -> Result<(), ResourceAbiError> {
        let refused = match descriptor.handle {
            // A 2D surface load/store has no depth/stencil form.
            CudaExternalMemoryHandle::DmaBufFd(_)
            | CudaExternalMemoryHandle::HostPointer { .. } => descriptor.format.is_depth_stencil(),
            // A Windows NT shared handle has no 3-plane planar form.
            CudaExternalMemoryHandle::Win32Nt(_) => {
                descriptor.format.is_planar_video()
                    && descriptor.format != ImageFormat::Yuv420SemiPlanar
            }
        };
        if refused {
            return Err(descriptor.unsupported_combination());
        }
        Ok(())
    }
}

/// Descriptor for importing external memory into the CUDA driver.
pub type CudaExternalMemoryDescriptor = ExternalImportDescriptor<CudaExternalMemoryHandle>;

/// CUDA concrete external resource importer and synchronization engine.
pub type CudaExternalResourceImporter = ExternalResourceImporter<CudaExternalImportPolicy>;
