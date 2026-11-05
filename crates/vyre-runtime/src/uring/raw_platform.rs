//! Platform FFI, mmap, futex, and raw syscall wrappers for io_uring.
//!
//! This is the only file in `vyre-runtime` that carries a file-level
//! `allow(unsafe_code)`. The crate root denies the lint, so an `unsafe` block
//! added to any other module is a compile error unless that single item also
//! carries the allow, beside the comment discharging the obligation.
//!
//! A mapped address never leaves this module. [`MappedRing`] owns the ring
//! descriptor and the three regions mapped from it, and every read or write of
//! those regions is one of its methods, so a borrow cannot outlive the mapping
//! it came from and an offset cannot leave the region it indexes.

#![allow(unsafe_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]

use crate::PipelineError;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};

// ---- io_uring Constants ----
pub(crate) const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
pub(crate) const IORING_SETUP_SQPOLL: u32 = 1 << 1;
pub(crate) const IORING_ENTER_SQ_WAKEUP: u32 = 1 << 1;
pub(crate) const IORING_SQ_NEED_WAKEUP: u32 = 1 << 0;

pub(crate) const IORING_OFF_SQ_RING: u64 = 0;
pub(crate) const IORING_OFF_CQ_RING: u64 = 0x8000000;
pub(crate) const IORING_OFF_SQES: u64 = 0x10000000;

pub(crate) const IORING_REGISTER_BUFFERS: u32 = 0;
pub(crate) const IORING_REGISTER_FILES: u32 = 2;

pub(crate) const IOSQE_FIXED_FILE: u8 = 1 << 0;

// ---- Struct Definitions matching Linux ABI ----

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct io_sqring_offsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub flags: u32,
    pub dropped: u32,
    pub array: u32,
    pub resv1: u32,
    pub resv2: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct io_cqring_offsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub overflow: u32,
    pub cqes: u32,
    pub flags: u32,
    pub resv1: u32,
    pub resv2: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct io_uring_params {
    pub sq_entries: u32,
    pub cq_entries: u32,
    pub flags: u32,
    pub sq_thread_cpu: u32,
    pub sq_thread_idle: u32,
    pub features: u32,
    pub wq_fd: u32,
    pub resv: [u32; 3],
    pub sq_off: io_sqring_offsets,
    pub cq_off: io_cqring_offsets,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct io_uring_sqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    pub user_data_or_off: u64,
    pub addr: u64,
    pub len: u32,
    pub op_flags: u32,
    pub user_data: u64,
    pub buf_index: u16,
    pub personality: u16,
    pub file_index: i32,
    pub addr3: u64,
    pub __pad2: [u64; 1],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct io_uring_cqe {
    pub user_data: u64,
    pub res: i32,
    pub flags: u32,
}

#[repr(C)]
struct futex_waitv {
    val: u64,
    uaddr: u64,
    flags: u32,
    __reserved: u32,
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

const FUTEX2_SIZE_U32: u32 = 0x02;
const SYS_FUTEX_WAITV: libc::c_long = 449;

/// The kernel resources one io_uring occupies: the ring descriptor and the
/// submission ring, completion ring, and SQE table mapped from it.
#[derive(Debug)]
pub(crate) struct MappedRing {
    ring_fd: i32,
    sq_ring_ptr: *mut libc::c_void,
    sq_ring_size: usize,
    cq_ring_ptr: *mut libc::c_void,
    cq_ring_size: usize,
    sqes_ptr: *mut libc::c_void,
    sqes_size: usize,
}

// SAFETY: the caller must uphold that exactly one live `MappedRing` names a
// given descriptor and region set. `new` is the only constructor, it takes the
// descriptor and the three regions by value, and the fields are private, so a
// second owner cannot be formed inside the crate and a move transfers the whole
// set. Ring header words are read and written through `AtomicU32`, and an SQE
// slot is handed out only behind `&mut self`, so a shared reference cannot
// produce two mutable borrows of one slot.
unsafe impl Send for MappedRing {}
unsafe impl Sync for MappedRing {}

impl MappedRing {
    /// Create one io_uring and map its submission ring, completion ring, and
    /// SQE table, reporting the parameters the kernel granted the ring.
    ///
    /// The returned value owns the descriptor and every mapping, and a failure
    /// part-way through releases what it had already taken, so this leaves no
    /// obligation with the caller.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IoUringSyscall`] when `io_uring_setup` or any
    /// of the three mappings is refused, and [`PipelineError::IntegerWidth`]
    /// when the kernel reports a ring the host address space cannot span.
    pub(super) fn setup(
        entries: u32,
        flags: u32,
        sq_thread_idle: u32,
    ) -> Result<(Self, io_uring_params), PipelineError> {
        let mut params = io_uring_params {
            flags,
            sq_thread_idle,
            ..io_uring_params::default()
        };
        let ring_fd = sys_io_uring_setup(entries, &mut params)?;
        match Self::map_regions(ring_fd, &params) {
            Ok(ring) => Ok((ring, params)),
            Err(error) => {
                sys_close_fd(ring_fd);
                Err(error)
            }
        }
    }

    /// Map the three regions of an already-created ring.
    ///
    /// The descriptor remains [`MappedRing::setup`]'s until this returns `Ok`,
    /// which is why every error path here leaves it open for that caller to
    /// close.
    fn map_regions(ring_fd: i32, params: &io_uring_params) -> Result<Self, PipelineError> {
        let sq_ring_span = super::ring::kernel_ring_span_usize(
            params.sq_off.array,
            params.sq_entries,
            mem::size_of::<u32>(),
            "SQ ring",
        )?;
        let cq_ring_span = super::ring::kernel_ring_span_usize(
            params.cq_off.cqes,
            params.cq_entries,
            mem::size_of::<io_uring_cqe>(),
            "CQ ring",
        )?;
        // One mapping carries both rings when the kernel offers the feature, so
        // it has to span the larger of the two.
        let single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
        let (sq_ring_size, cq_ring_size) = if single_mmap {
            let shared = core::cmp::max(sq_ring_span, cq_ring_span);
            (shared, shared)
        } else {
            (sq_ring_span, cq_ring_span)
        };
        let sqes_size = super::ring::kernel_record_span_usize(
            params.sq_entries,
            mem::size_of::<io_uring_sqe>(),
            "SQE table",
        )?;

        let sq_ring_ptr =
            sys_mmap_ring(ring_fd, sq_ring_size, IORING_OFF_SQ_RING, "mmap(sq_ring)")?;
        let cq_ring_ptr = if single_mmap {
            sq_ring_ptr
        } else {
            match sys_mmap_ring(ring_fd, cq_ring_size, IORING_OFF_CQ_RING, "mmap(cq_ring)") {
                Ok(ptr) => ptr,
                Err(error) => {
                    // SAFETY: the address and length are the ones the mapping
                    // above returned, and nothing has borrowed the region.
                    unsafe { sys_munmap(sq_ring_ptr, sq_ring_size) };
                    return Err(error);
                }
            }
        };
        let sqes_ptr = match sys_mmap_ring(ring_fd, sqes_size, IORING_OFF_SQES, "mmap(sqes)") {
            Ok(ptr) => ptr,
            Err(error) => {
                // SAFETY: each address and length pair is the one the matching
                // mapping above returned, nothing has borrowed either region,
                // and a shared mapping is released once.
                unsafe {
                    if !single_mmap {
                        sys_munmap(cq_ring_ptr, cq_ring_size);
                    }
                    sys_munmap(sq_ring_ptr, sq_ring_size);
                }
                return Err(error);
            }
        };

        Ok(Self {
            ring_fd,
            sq_ring_ptr,
            sq_ring_size,
            cq_ring_ptr,
            cq_ring_size,
            sqes_ptr,
            sqes_size,
        })
    }

    /// Enter the ring to submit entries, wait for completions, or both.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IoUringSyscall`] when the kernel refuses the
    /// call or reports a completion count wider than `i32`.
    pub(super) fn enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
    ) -> Result<i32, PipelineError> {
        // SAFETY: `self.ring_fd` is the descriptor `setup` opened and this
        // value still owns, so it is open for the call. The signal set is
        // null, which the kernel reads as "leave the mask alone", so no
        // signal-set memory is passed.
        let res = unsafe {
            libc::syscall(
                libc::SYS_io_uring_enter,
                self.ring_fd,
                to_submit,
                min_complete,
                flags,
                ptr::null::<libc::sigset_t>(),
                0,
            )
        };
        if res < 0 {
            return Err(PipelineError::IoUringSyscall {
                syscall: "io_uring_enter",
                errno: get_errno(),
                fix: "retry on EINTR/EBUSY; check SQPOLL thread health via /proc/<pid>/task/ on ENXIO",
            });
        }
        i32::try_from(res).map_err(|_| PipelineError::IoUringSyscall {
            syscall: "io_uring_enter",
            errno: libc::EOVERFLOW,
            fix: "io_uring_enter returned a completion count outside i32; check libc/kernel ABI bindings",
        })
    }

    /// Register a fixed buffer array with `IORING_REGISTER_BUFFERS`.
    ///
    /// # Safety
    ///
    /// The caller must uphold that every range in `iovecs` stays mapped and
    /// writable, and is not accessed by the host, until the ring is torn down.
    /// The kernel writes transfer results directly into those ranges after
    /// this call returns.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IntegerWidth`] when the array is longer than
    /// `u32` and [`PipelineError::IoUringSyscall`] when the kernel refuses it.
    pub(super) unsafe fn register_buffers(
        &self,
        iovecs: &[super::buffer::Iovec],
    ) -> Result<(), PipelineError> {
        let count = u32::try_from(iovecs.len()).map_err(|_| PipelineError::IntegerWidth {
            quantity: "registered buffer count",
            value: u128::try_from(iovecs.len()).unwrap_or(0),
            bits: 32,
            fix: "reduce the buffer registration count to fit within u32 bounds",
        })?;
        // SAFETY: `self.ring_fd` is open for the call, `iovecs` is a live
        // shared borrow so the array itself stays mapped, and the caller has
        // upheld that the ranges it describes stay writable for the kernel.
        let res = unsafe {
            libc::syscall(
                libc::SYS_io_uring_register,
                self.ring_fd,
                IORING_REGISTER_BUFFERS,
                iovecs.as_ptr().cast::<core::ffi::c_void>(),
                count,
            )
        };
        if res < 0 {
            return Err(PipelineError::IoUringSyscall {
                syscall: "io_uring_register(BUFFERS)",
                errno: get_errno(),
                fix: "check /proc/sys/vm/max_user_watches; EOPNOTSUPP means kernel < 5.1",
            });
        }
        Ok(())
    }

    /// Register a fixed descriptor set with `IORING_REGISTER_FILES`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IntegerWidth`] when the set is longer than
    /// `u32` and [`PipelineError::IoUringSyscall`] when the kernel refuses it,
    /// which is what a closed descriptor in `fds` reports as.
    pub(super) fn register_files(&self, fds: &[i32]) -> Result<(), PipelineError> {
        let count = u32::try_from(fds.len()).map_err(|_| PipelineError::IntegerWidth {
            quantity: "registered file count",
            value: u128::try_from(fds.len()).unwrap_or(0),
            bits: 32,
            fix: "reduce the file registration count to fit within u32 bounds",
        })?;
        // SAFETY: `self.ring_fd` is open for the call and `fds` is a live
        // shared borrow, so the array stays mapped while the kernel copies it.
        // A descriptor the caller has closed is reported as an errno, not a
        // memory error.
        let res = unsafe {
            libc::syscall(
                libc::SYS_io_uring_register,
                self.ring_fd,
                IORING_REGISTER_FILES,
                fds.as_ptr().cast::<core::ffi::c_void>(),
                count,
            )
        };
        if res < 0 {
            return Err(PipelineError::IoUringSyscall {
                syscall: "io_uring_register(FILES)",
                errno: get_errno(),
                fix: "ensure every fd is still open; ENOMEM means lower the fd set size",
            });
        }
        Ok(())
    }

    /// Byte offset of a `u32` word inside a region of `region_bytes`.
    ///
    /// The kernel reports these offsets from `io_uring_setup`. An offset past
    /// the region it names means the mapped layout disagrees with the ABI
    /// structs this module declares, which every subsequent access would read
    /// outside the mapping.
    fn word_offset(offset: usize, region_bytes: usize, region: &'static str) -> usize {
        assert!(
            offset
                .checked_add(mem::size_of::<u32>())
                .is_some_and(|end| end <= region_bytes),
            "{region} word offset {offset} leaves the {region_bytes}-byte mapping. \
             Fix: rebuild against the running kernel's io_uring ABI; the mapped \
             layout and the declared structs disagree"
        );
        offset
    }

    /// Atomically load the submission-ring header word at `offset`.
    pub(crate) fn sq_load(&self, offset: usize, order: Ordering) -> u32 {
        let offset = Self::word_offset(offset, self.sq_ring_size, "submission ring");
        // SAFETY: `map_regions` mapped `sq_ring_size` bytes at `sq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. The kernel writes the same word atomically.
        unsafe { (*(self.sq_ring_ptr.add(offset).cast::<AtomicU32>())).load(order) }
    }

    /// Atomically store `val` into the submission-ring header word at `offset`.
    pub(crate) fn sq_store(&self, offset: usize, val: u32, order: Ordering) {
        let offset = Self::word_offset(offset, self.sq_ring_size, "submission ring");
        // SAFETY: `map_regions` mapped `sq_ring_size` bytes at `sq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. The kernel reads the same word atomically.
        unsafe { (*(self.sq_ring_ptr.add(offset).cast::<AtomicU32>())).store(val, order) }
    }

    /// Read the submission-ring word at `offset` that only the kernel writes.
    pub(crate) fn sq_read(&self, offset: usize) -> u32 {
        let offset = Self::word_offset(offset, self.sq_ring_size, "submission ring");
        // SAFETY: `map_regions` mapped `sq_ring_size` bytes at `sq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. `ring_entries` and `ring_mask` are fixed for the ring's life,
        // so no concurrent write races this read.
        unsafe { *(self.sq_ring_ptr.add(offset).cast::<u32>()) }
    }

    /// Publish `val` into the submission-queue index array at `offset`.
    pub(crate) fn sq_write(&self, offset: usize, val: u32) {
        let offset = Self::word_offset(offset, self.sq_ring_size, "submission ring");
        // SAFETY: `map_regions` mapped `sq_ring_size` bytes at `sq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. The caller must uphold that the slot it names is one the tail
        // has not yet published, which the release store on the tail then
        // orders before the kernel can read it.
        unsafe { *(self.sq_ring_ptr.add(offset).cast::<u32>()) = val }
    }

    /// Atomically load the completion-ring header word at `offset`.
    pub(crate) fn cq_load(&self, offset: usize, order: Ordering) -> u32 {
        let offset = Self::word_offset(offset, self.cq_ring_size, "completion ring");
        // SAFETY: `map_regions` mapped `cq_ring_size` bytes at `cq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. The kernel writes the same word atomically.
        unsafe { (*(self.cq_ring_ptr.add(offset).cast::<AtomicU32>())).load(order) }
    }

    /// Atomically store `val` into the completion-ring header word at `offset`.
    pub(crate) fn cq_store(&self, offset: usize, val: u32, order: Ordering) {
        let offset = Self::word_offset(offset, self.cq_ring_size, "completion ring");
        // SAFETY: `map_regions` mapped `cq_ring_size` bytes at `cq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. The kernel reads the same word atomically.
        unsafe { (*(self.cq_ring_ptr.add(offset).cast::<AtomicU32>())).store(val, order) }
    }

    /// Read the completion-ring word at `offset` that only the kernel writes.
    pub(crate) fn cq_read(&self, offset: usize) -> u32 {
        let offset = Self::word_offset(offset, self.cq_ring_size, "completion ring");
        // SAFETY: `map_regions` mapped `cq_ring_size` bytes at `cq_ring_ptr`,
        // and `word_offset` has established that the whole word lies inside
        // them. `ring_mask` is fixed for the ring's life, so no concurrent
        // write races this read.
        unsafe { *(self.cq_ring_ptr.add(offset).cast::<u32>()) }
    }

    /// Borrow submission entry `index` for the duration of this mutable borrow.
    ///
    /// # Panics
    ///
    /// Panics when the entry leaves the mapped SQE table. The mapping is what
    /// makes the dereference below sound, so a caller that did not mask the
    /// index has no recoverable state to return to.
    pub(crate) fn sqe_mut(&mut self, index: usize) -> &mut io_uring_sqe {
        assert!(
            index
                .checked_add(1)
                .and_then(|count| count.checked_mul(mem::size_of::<io_uring_sqe>()))
                .is_some_and(|end| end <= self.sqes_size),
            "submission entry {index} leaves the {} byte SQE table. \
             Fix: mask the index with the kernel's ring_mask before submitting",
            self.sqes_size
        );
        // SAFETY: `map_regions` mapped `sqes_size` bytes of `io_uring_sqe`
        // records at `sqes_ptr`, and the assertion above has established that
        // the whole record lies inside them. The returned borrow lives no
        // longer than the `&mut self` it came from, so no second borrow of the
        // same slot exists while the caller fills it, and the kernel reads the
        // slot only after the tail store this borrow precedes.
        unsafe { &mut *self.sqes_ptr.cast::<io_uring_sqe>().add(index) }
    }

    /// Borrow completion entry `index` in the array at `offset`.
    ///
    /// # Panics
    ///
    /// Panics when the entry leaves the mapped completion ring, for the same
    /// reason [`Self::sqe_mut`] does.
    pub(crate) fn cqe(&self, offset: usize, index: usize) -> &io_uring_cqe {
        assert!(
            index
                .checked_add(1)
                .and_then(|count| count.checked_mul(mem::size_of::<io_uring_cqe>()))
                .and_then(|span| span.checked_add(offset))
                .is_some_and(|end| end <= self.cq_ring_size),
            "completion entry {index} at offset {offset} leaves the {} byte completion ring. \
             Fix: mask the index with the kernel's ring_mask before reading",
            self.cq_ring_size
        );
        // SAFETY: `map_regions` mapped `cq_ring_size` bytes at `cq_ring_ptr`,
        // and the assertion above has established that the whole record lies
        // inside them. The borrow lives no longer than the `&self` it came
        // from, and the acquire load on the completion tail that the caller
        // performed first orders the kernel's write to this record before it.
        unsafe {
            &*self
                .cq_ring_ptr
                .add(offset)
                .cast::<io_uring_cqe>()
                .add(index)
        }
    }
}

impl Drop for MappedRing {
    fn drop(&mut self) {
        // SAFETY: each address and length pair is what `map_regions` recorded
        // for that mapping. `drop` runs once and takes `&mut self`, so no
        // borrow of any region is outstanding, and a shared submission and
        // completion mapping is released once.
        unsafe {
            sys_munmap(self.sqes_ptr, self.sqes_size);
            if self.sq_ring_ptr != self.cq_ring_ptr {
                sys_munmap(self.cq_ring_ptr, self.cq_ring_size);
            }
            sys_munmap(self.sq_ring_ptr, self.sq_ring_size);
        }
        sys_close_fd(self.ring_fd);
    }
}

/// One host-visible byte range io_uring is allowed to transfer into.
#[derive(Debug)]
pub(crate) struct RawBufferPointer {
    ptr: *mut u8,
}

// SAFETY: the caller must uphold that the range this pointer names is not
// concurrently accessed through any other path. The type is not `Sync` and not
// `Clone`, so a value moved to another thread leaves no second handle behind,
// and the exclusive borrow `as_mut_slice` requires cannot be taken twice.
unsafe impl Send for RawBufferPointer {}

impl RawBufferPointer {
    /// Adopt one host-visible address.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `ptr` is the start of a host-visible range
    /// that stays mapped and unaliased for as long as this value lives, and
    /// that every length later passed to [`RawBufferPointer::as_mut_slice`]
    /// lies inside it.
    #[must_use]
    pub(crate) const unsafe fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    /// Name the address `offset` bytes into this range.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `offset` is inside the range this pointer
    /// names, so that the result carries the same obligation over a suffix of
    /// it rather than over unmapped memory.
    #[must_use]
    pub(crate) unsafe fn offset(&self, offset: usize) -> Self {
        Self {
            ptr: self.ptr.wrapping_add(offset),
        }
    }

    /// The host-visible address io_uring is given.
    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Borrow `len` bytes for the duration of this mutable borrow.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `len` bytes from this pointer are mapped
    /// and that no device transfer is in flight over them, because the returned
    /// slice is an exclusive borrow and a concurrent DMA write into it is a
    /// data race the type system cannot see.
    pub(crate) unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        // SAFETY: the caller has upheld that `len` bytes from `self.ptr` are
        // mapped and untouched by the device, and the borrow this returns lives
        // no longer than the `&mut self` it came from.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

/// Read the calling thread's `errno`.
#[must_use]
fn get_errno() -> i32 {
    // SAFETY: `__errno_location` returns the address of the calling thread's
    // own `errno`, so the read is thread-local and needs no caller obligation.
    unsafe { *libc::__errno_location() }
}

/// Create one io_uring, reporting the parameters the kernel granted it.
fn sys_io_uring_setup(entries: u32, params: &mut io_uring_params) -> Result<i32, PipelineError> {
    // SAFETY: `params` is a live exclusive borrow, so the kernel writes the
    // returned parameters into memory no other reference reaches.
    let ring_fd = unsafe {
        libc::syscall(
            libc::SYS_io_uring_setup,
            entries,
            params as *mut io_uring_params,
        )
    };
    if ring_fd < 0 {
        return Err(PipelineError::IoUringSyscall {
            syscall: "io_uring_setup",
            errno: get_errno(),
            fix: "check kernel version (>= 5.1 required), CAP_SYS_ADMIN for SQPOLL on < 5.13, and nofile ulimit",
        });
    }
    i32::try_from(ring_fd).map_err(|_| PipelineError::IoUringSyscall {
        syscall: "io_uring_setup",
        errno: libc::EOVERFLOW,
        fix: "io_uring_setup returned an fd outside i32; check libc/kernel ABI bindings",
    })
}

/// Map one of a ring's three kernel regions.
///
/// [`MappedRing::map_regions`] is the only caller. It derives every `size`
/// from the parameters the same ring reported, which is what keeps a mapping
/// from spanning past the region `offset` names and faulting on first access.
fn sys_mmap_ring(
    ring_fd: i32,
    size: usize,
    offset: u64,
    label: &'static str,
) -> Result<*mut libc::c_void, PipelineError> {
    // SAFETY: the caller must uphold that `ring_fd` is an open io_uring
    // descriptor and that `offset` is one of the kernel's `IORING_OFF_*`
    // constants. A null hint lets the kernel place the mapping, so no existing
    // mapping is replaced.
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            ring_fd,
            offset as libc::off_t,
        )
    };
    if ptr == libc::MAP_FAILED {
        Err(PipelineError::IoUringSyscall {
            syscall: label,
            errno: get_errno(),
            fix: "check /proc/sys/vm/max_map_count and process memory limits",
        })
    } else {
        Ok(ptr)
    }
}

/// Release one mapped region.
///
/// # Safety
///
/// The caller must uphold that `ptr` and `size` are the address and length one
/// earlier [`sys_mmap_ring`] returned, and that nothing still borrows the
/// region. A null or `MAP_FAILED` address is ignored rather than passed on.
unsafe fn sys_munmap(ptr: *mut libc::c_void, size: usize) {
    if !ptr.is_null() && ptr != libc::MAP_FAILED {
        // SAFETY: the caller has upheld that this address and length name one
        // live mapping it owns and that no borrow of it is outstanding.
        unsafe {
            libc::munmap(ptr, size);
        }
    }
}

/// Close one descriptor this crate opened.
fn sys_close_fd(fd: i32) {
    if fd >= 0 {
        // SAFETY: `fd` was opened by `sys_io_uring_setup` and is closed by the
        // single owner that value was handed to, so no other holder exists.
        unsafe {
            libc::close(fd);
        }
    }
}

/// Wait until the word at `host_visible_addr` stops reading `current`.
///
/// # Safety
///
/// The caller must uphold that `host_visible_addr` names a mapped, naturally
/// aligned `u32` that stays mapped for the whole wait, because the kernel
/// reads it after this call has parked the thread.
///
/// # Errors
///
/// Returns [`PipelineError::IoUringSyscall`] when the kernel refuses the wait.
/// A value that already changed reports `EAGAIN`, which is success.
pub(crate) unsafe fn sys_futex_waitv(
    host_visible_addr: *const u32,
    current: u32,
    timeout_ns: u64,
) -> Result<(), PipelineError> {
    let waitv = [futex_waitv {
        val: current as u64,
        uaddr: host_visible_addr as u64,
        flags: FUTEX2_SIZE_U32,
        __reserved: 0,
    }];

    let ts = Timespec {
        tv_sec: (timeout_ns / 1_000_000_000) as i64,
        tv_nsec: (timeout_ns % 1_000_000_000) as i64,
    };

    // SAFETY: the caller has upheld that `host_visible_addr` names a mapped,
    // naturally aligned `u32` that stays mapped for the wait. `waitv` and `ts`
    // are live locals, so the kernel reads them from this frame.
    let res = unsafe {
        libc::syscall(
            SYS_FUTEX_WAITV,
            waitv.as_ptr() as *const libc::c_void,
            1u32,
            0u32,
            &ts as *const Timespec,
            0u64,
        )
    };

    if res < 0 {
        let errno = get_errno();
        if errno == libc::EAGAIN {
            return Ok(());
        }
        return Err(PipelineError::IoUringSyscall {
            syscall: "futex_waitv",
            errno,
            fix: "kernel 5.16+ required; ETIMEDOUT means the value didn't change within timeout_ns",
        });
    }
    Ok(())
}
