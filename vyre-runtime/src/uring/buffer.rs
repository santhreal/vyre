//! GPU-visible memory region wrappers and ABI structures for io_uring.

use super::raw_platform::RawBufferPointer;
use crate::{CounterArithmetic, CounterScope, PipelineError};
use core::marker::PhantomData;

/// Minimal `iovec` struct matching the Linux ABI for `readv`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Iovec {
    /// Starting host-visible address of the buffer.
    pub iov_base: *mut core::ffi::c_void,
    /// Length of the buffer in bytes.
    pub iov_len: usize,
}

/// `IORING_OP_READV`  -  scatter-read into an array of iovecs.
pub(crate) const IORING_OP_READV: u8 = 1;
/// `IORING_OP_READ_FIXED`  -  read into a pre-registered buffer.
pub(crate) const IORING_OP_READ_FIXED: u8 = 22;
/// `IORING_OP_URING_CMD`  -  vendor-specific passthrough (NVMe). Kernel 6.0+.
#[cfg(feature = "uring-cmd-nvme")]
pub(crate) const IORING_OP_URING_CMD: u8 = 46;

/// GPU-visible memory region that io_uring is allowed to DMA into.
///
/// Aliasing safety is a type property: `GpuMappedBuffer` represents an exclusive
/// mutable borrow (`&'a mut [u8]`) of the underlying device allocation.
/// It is `Send` but intentionally `!Sync`, and cannot be duplicated.
#[derive(Debug)]
pub struct GpuMappedBuffer<'a> {
    raw: RawBufferPointer,
    len: usize,
    _owner: PhantomData<&'a mut [u8]>,
}

impl<'a> GpuMappedBuffer<'a> {
    /// Construct from a borrowed host-visible byte slice.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn from_host_visible_slice(slice: &'a mut [u8]) -> Self {
        let len = slice.len();
        // SAFETY: the address comes from a live exclusive borrow of `len`
        // bytes, and the `'a` anchor keeps that borrow alive for as long as
        // this value, so nothing else reaches the range meanwhile.
        let raw = unsafe { RawBufferPointer::new(slice.as_mut_ptr()) };
        Self {
            raw,
            len,
            _owner: PhantomData,
        }
    }

    /// Construct from a host-visible address plus an explicit owner anchor.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `ptr` is the start of `len` host-visible
    /// bytes owned by `_owner`, that the mapping stays live and unaliased for
    /// the borrow of `_owner`, and that no other handle reads or writes the
    /// range. `_owner` ties the region's lifetime to something the borrow
    /// checker can see, but it does not establish that `ptr` names it.
    #[must_use]
    #[allow(unsafe_code)]
    pub unsafe fn from_host_visible_owner<O: ?Sized>(
        _owner: &'a mut O,
        ptr: *mut u8,
        len: usize,
    ) -> Self {
        Self {
            // SAFETY: the caller has upheld that `ptr` names `len` live
            // host-visible bytes that stay mapped and unaliased.
            raw: unsafe { RawBufferPointer::new(ptr) },
            len,
            _owner: PhantomData,
        }
    }

    /// Carve out a sub-region of this mapped buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::CounterOverflow`] when `offset + len` leaves
    /// the host address range, and [`PipelineError::RegionBounds`] when the
    /// range ends past the mapped buffer.
    #[allow(unsafe_code)]
    pub fn sub_region(&self, offset: usize, len: usize) -> Result<Self, PipelineError> {
        let offset_u64 = mapped_byte_count(offset, "GpuMappedBuffer sub-region offset")?;
        let len_u64 = mapped_byte_count(len, "GpuMappedBuffer sub-region length")?;
        let region_len = mapped_byte_count(self.len, "GpuMappedBuffer mapped length")?;
        let _end = vyre_driver::accounting::checked_usize_byte_range_end_lazy(
            offset,
            len,
            self.len,
            || PipelineError::CounterOverflow {
                scope: CounterScope::IoUring,
                counter: "GpuMappedBuffer sub-region end offset",
                arithmetic: CounterArithmetic::Sum,
                lhs: offset_u64,
                rhs: len_u64,
                bits: usize::BITS,
                fix: "reduce the slot size or its offset; their sum leaves the host address range",
            },
            |_| PipelineError::RegionBounds {
                region: "GpuMappedBuffer mapped allocation",
                offset: offset_u64,
                len: len_u64,
                region_len,
                unit: "bytes",
                fix: "reduce the slot size or enlarge the staging buffer",
            },
        )?;
        // SAFETY: the bound check above has established that `offset` is
        // inside this region, so the result names a suffix of the same
        // host-visible range and carries the same obligation over it.
        let raw = unsafe { self.raw.offset(offset) };
        Ok(Self {
            raw,
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
        self.raw.as_ptr()
    }

    /// Borrow the mapped bytes as a mutable slice.
    ///
    /// The exclusive borrow is what keeps a device transfer from writing into
    /// the range while the slice exists: a submission needs the buffer, and a
    /// buffer cannot be submitted while this borrow is outstanding.
    #[allow(unsafe_code)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: construction established that `self.len` bytes from this
        // pointer are mapped, and the exclusive borrow of `self` excludes both
        // another host handle and an in-flight transfer over the range.
        unsafe { self.raw.as_mut_slice(self.len) }
    }

    /// Construct from a PCIe peer-memory address for direct storage DMA.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `peer_ptr` is the start of `len` bytes of
    /// BAR1 peer memory that `_owner` keeps mapped for the borrow, that the
    /// range is registered for peer-to-peer DMA, and that no other handle
    /// reads or writes it.
    #[must_use]
    #[allow(unsafe_code)]
    pub unsafe fn from_bar1_peer_with_owner<O: ?Sized>(
        _owner: &'a mut O,
        peer_ptr: *mut u8,
        len: usize,
    ) -> Self {
        Self {
            // SAFETY: the caller has upheld that `peer_ptr` names `len` live
            // peer-mapped bytes that stay mapped and unaliased.
            raw: unsafe { RawBufferPointer::new(peer_ptr) },
            len,
            _owner: PhantomData,
        }
    }
}

/// Widen a host byte count into the `u64` an error field carries.
fn mapped_byte_count(value: usize, quantity: &'static str) -> Result<u64, PipelineError> {
    u64::try_from(value).map_err(|_| PipelineError::IntegerWidth {
        quantity,
        value: u128::try_from(value).unwrap_or(0),
        bits: 64,
        fix: "keep mapped allocations within 64-bit bounds",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_buffer_sub_region_is_valid() {
        let mut backing = vec![0u8; 64];
        let mut mapped = GpuMappedBuffer::from_host_visible_slice(&mut backing);
        let slice = mapped.as_mut_slice();
        slice[0] = 42;
        assert_eq!(slice[0], 42);

        let sub = mapped.sub_region(8, 16).unwrap();
        assert_eq!(sub.len(), 16);
    }

    #[test]
    fn mapped_buffer_bounds_check() {
        let mut backing = vec![0u8; 32];
        let mapped = GpuMappedBuffer::from_host_visible_slice(&mut backing);
        let err = mapped.sub_region(20, 20).unwrap_err();
        let PipelineError::RegionBounds {
            offset,
            len,
            region_len,
            ..
        } = err
        else {
            panic!("Expected RegionBounds error, got {err:?}");
        };
        assert_eq!(offset, 20);
        assert_eq!(len, 20);
        assert_eq!(region_len, 32);
    }
}
