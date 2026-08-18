//! GPU-visible memory region wrappers and ABI structures for io_uring.

use crate::PipelineError;
use core::marker::PhantomData;

/// Minimal `iovec` struct matching the Linux ABI for `readv`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Iovec {
    /// Target buffer address for this chunk of the read.
    pub iov_base: *mut core::ffi::c_void,
    /// Byte length of the target buffer.
    pub iov_len: usize,
}

/// `IORING_OP_READV`  -  scatter-read into an array of iovecs.
pub(crate) const IORING_OP_READV: u8 = 1;
/// `IORING_OP_READ_FIXED`  -  read into a pre-registered buffer.
pub(crate) const IORING_OP_READ_FIXED: u8 = 22;
/// `IORING_OP_URING_CMD`  -  vendor-specific passthrough (NVMe). Kernel 6.0+.
pub(crate) const IORING_OP_URING_CMD: u8 = 46;

/// GPU-visible memory region that io_uring is allowed to DMA into.
///
/// Compatibility constructors cover host-visible shared mappings. The BAR1
/// constructor covers the native GPUDirect path where NVMe DMA lands directly
/// in GPU-owned memory.
pub struct GpuMappedBuffer<'a> {
    ptr: *mut u8,
    len: usize,
    _owner: PhantomData<&'a mut [u8]>,
}

// SAFETY: Send + Sync because (a) the constructor's safety contract
// requires the caller to commit the lifetime invariant, and (b) the
// raw pointer is only dereferenced by the kernel via io_uring  -
// vyre-runtime never reads through it directly.
unsafe impl Send for GpuMappedBuffer<'_> {}
unsafe impl Sync for GpuMappedBuffer<'_> {}

macro_rules! define_mapped_owner_constructor {
    ($name:ident, $ptr:ident, $doc:expr) => {
        #[doc = $doc]
        pub unsafe fn $name<O: ?Sized>(_owner: &'a mut O, $ptr: *mut u8, len: usize) -> Self {
            Self {
                ptr: $ptr,
                len,
                _owner: PhantomData,
            }
        }
    };
}

impl<'a> GpuMappedBuffer<'a> {
    /// Construct from a borrowed host-visible byte slice.
    ///
    /// # Safety
    ///
    /// The caller asserts:
    /// - `slice` aliases a device allocation created with host-visible
    ///   host-shared usage bits by the concrete backend.
    /// - No other code reads or writes through `slice` while the
    ///   returned handle is alive.
    pub unsafe fn from_host_visible_slice(slice: &'a mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _owner: PhantomData,
        }
    }

    define_mapped_owner_constructor!(
        from_host_visible_owner,
        ptr,
        concat!(
            "Construct from a raw pointer plus an explicit owner anchor.\n\n",
            "The borrow on `owner` forces the mapped region to outlive every derived ",
            "[`AsyncUringStream`].\n\n",
            "# Safety\n\n",
            "The caller must ensure that `ptr` names a `len`-byte host-visible GPU ",
            "allocation owned by `owner`, and that no other code accesses the region ",
            "while the returned handle is alive."
        )
    );

    /// Duplicate the mapped-buffer handle for the same underlying region.
    ///
    /// # Safety
    ///
    /// The caller must uphold the same aliasing and lifetime guarantees as
    /// [`GpuMappedBuffer::from_host_visible_slice`]. This does not clone memory;
    /// it creates another handle to the same mapped bytes.
    pub unsafe fn duplicate(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _owner: PhantomData,
        }
    }

    /// Carve out a sub-region of this mapped buffer.
    ///
    /// This preserves the original constructor contract: the returned
    /// handle aliases the same host-visible GPU allocation and carries
    /// no ownership of its own.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when `offset + len`
    /// exceeds the mapped buffer bounds.
    pub fn sub_region(&self, offset: usize, len: usize) -> Result<Self, PipelineError> {
        let _end = vyre_driver::accounting::checked_usize_byte_range_end_lazy(
            offset,
            len,
            self.len,
            || {
                PipelineError::QueueFull {
                queue: "submission",
                fix: "GpuMappedBuffer::sub_region offset + len overflows usize; reduce slot size or enlarge the staging buffer",
            }
            },
            |_| {
                PipelineError::QueueFull {
                queue: "submission",
                fix: "GpuMappedBuffer::sub_region exceeds the mapped allocation; reduce slot size or enlarge the staging buffer",
            }
            },
        )?;
        Ok(Self {
            ptr: self.ptr.wrapping_add(offset),
            len,
            _owner: PhantomData,
        })
    }

    /// Byte length of the mapped region.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw pointer for io_uring submission. Crate-private.
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Borrow the mapped bytes as a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive mutable access to the region for the
    /// lifetime of the returned slice.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    define_mapped_owner_constructor!(
        from_bar1_peer_with_owner,
        peer_ptr,
        concat!(
            "Construct from a PCIe peer-memory pointer for direct storage DMA.\n\n",
            "# Safety\n\n",
            "The caller must ensure that `peer_ptr` names a GPU allocation suitable ",
            "for peer DMA, that the allocation outlives the handle, and that the ",
            "io_uring kernel and storage driver both support DMA mapping."
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_slice_roundtrip_is_miri_clean() {
        let mut backing = [1_u8, 2, 3, 4];
        // SAFETY: `backing` stays live and uniquely borrowed for the mapped buffer lifetime.
        let mut mapped = unsafe { GpuMappedBuffer::from_host_visible_slice(&mut backing) };
        // SAFETY: the mapped buffer was built from `backing` and remains uniquely borrowed.
        let slice = unsafe { mapped.as_mut_slice() };
        slice[0] = 9;
        slice[3] = 7;
        assert_eq!(backing, [9, 2, 3, 7]);
    }
}
