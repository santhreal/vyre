//! Low-level platform FFI, mmap, futex, and raw syscall wrappers for io_uring.
//!
//! SAFETY: This is the ONLY module in `vyre-runtime` permitted to declare `unsafe` blocks
//! and libc/raw pointer interactions. Every unsafe operation is encapsulated behind
//! checked platform abstractions.

#![allow(unsafe_code)]
#![allow(non_camel_case_types)]
// Linux ABI io_uring structs and constants mirror the kernel interface definitions.
#![allow(dead_code)]
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

/// Encapsulated raw pointers for memory-mapped io_uring state.
#[derive(Debug)]
pub(crate) struct RawRingPointers {
    pub(crate) sq_ring_ptr: *mut libc::c_void,
    pub(crate) sq_ring_size: usize,
    pub(crate) cq_ring_ptr: *mut libc::c_void,
    pub(crate) cq_ring_size: usize,
    pub(crate) sqes_ptr: *mut libc::c_void,
    pub(crate) sqes_size: usize,
}

// SAFETY: Ring pointers access synchronized atomic headers and exclusive SQE slots.
unsafe impl Send for RawRingPointers {}
unsafe impl Sync for RawRingPointers {}

/// Encapsulated raw pointer for GPU-mapped buffer.
#[derive(Debug)]
pub(crate) struct RawBufferPointer {
    pub(crate) ptr: *mut u8,
}

// SAFETY: Exclusive buffer pointer can safely be moved across threads.
// Note: Intentionally NOT Sync, enforcing exclusive access at the type level.
unsafe impl Send for RawBufferPointer {}

impl RawBufferPointer {
    /// Construct from raw pointer.
    #[must_use]
    pub(crate) const fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    /// Offset raw pointer by `offset` bytes.
    #[must_use]
    pub(crate) fn offset(&self, offset: usize) -> Self {
        Self {
            ptr: self.ptr.wrapping_add(offset),
        }
    }

    /// Return raw pointer.
    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Convert to exclusive mutable slice.
    pub(crate) fn as_mut_slice<'a>(&mut self, len: usize) -> &'a mut [u8] {
        // SAFETY: caller holds exclusive reference to self and asserts len valid bytes.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

/// Safely construct a zeroed POD instance.
#[must_use]
pub(crate) fn zeroed_pod<T: Copy>() -> T {
    // SAFETY: caller guarantees T is POD without invalid bit-patterns (e.g. integer C-ABI structs).
    unsafe { mem::zeroed() }
}

/// Query current thread-local errno.
#[must_use]
pub(crate) fn get_errno() -> i32 {
    // SAFETY: __errno_location is always valid in the executing thread.
    unsafe { *libc::__errno_location() }
}

/// Execute `io_uring_setup` syscall.
pub(crate) fn sys_io_uring_setup(
    entries: u32,
    params: &mut io_uring_params,
) -> Result<i32, PipelineError> {
    // SAFETY: io_uring_setup receives a valid mutable io_uring_params pointer.
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

/// Execute `io_uring_enter` syscall.
pub(crate) fn sys_io_uring_enter(
    ring_fd: i32,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
) -> Result<i32, PipelineError> {
    // SAFETY: ring_fd is a valid open file descriptor; signal set is null.
    let res = unsafe {
        libc::syscall(
            libc::SYS_io_uring_enter,
            ring_fd,
            to_submit,
            min_complete,
            flags,
            ptr::null::<libc::sigset_t>(),
            0,
        )
    };
    if res < 0 {
        Err(PipelineError::IoUringSyscall {
            syscall: "io_uring_enter",
            errno: get_errno(),
            fix: "retry on EINTR/EBUSY; check SQPOLL thread health via /proc/<pid>/task/ on ENXIO",
        })
    } else {
        i32::try_from(res).map_err(|_| PipelineError::IoUringSyscall {
            syscall: "io_uring_enter",
            errno: libc::EOVERFLOW,
            fix: "io_uring_enter returned a completion count outside i32; check libc/kernel ABI bindings",
        })
    }
}

/// Execute `io_uring_register` for buffer arrays.
pub(crate) fn sys_io_uring_register_buffers(
    ring_fd: i32,
    iovecs: &[super::buffer::Iovec],
) -> Result<(), PipelineError> {
    let count = u32::try_from(iovecs.len()).map_err(|_| PipelineError::IntegerWidth {
        quantity: "registered buffer count",
        value: u128::try_from(iovecs.len()).unwrap_or(0),
        bits: 32,
        fix: "reduce the buffer registration count to fit within u32 bounds",
    })?;
    // SAFETY: ring_fd and iovec buffer slice are valid for the registration duration.
    let res = unsafe {
        libc::syscall(
            libc::SYS_io_uring_register,
            ring_fd,
            IORING_REGISTER_BUFFERS,
            iovecs.as_ptr() as *const core::ffi::c_void,
            count,
        )
    };
    if res < 0 {
        Err(PipelineError::IoUringSyscall {
            syscall: "io_uring_register(BUFFERS)",
            errno: get_errno(),
            fix: "check /proc/sys/vm/max_user_watches; EOPNOTSUPP means kernel < 5.1",
        })
    } else {
        Ok(())
    }
}

/// Execute `io_uring_register` for fixed files.
pub(crate) fn sys_io_uring_register_files(ring_fd: i32, fds: &[i32]) -> Result<(), PipelineError> {
    let count = u32::try_from(fds.len()).map_err(|_| PipelineError::IntegerWidth {
        quantity: "registered file count",
        value: u128::try_from(fds.len()).unwrap_or(0),
        bits: 32,
        fix: "reduce the file registration count to fit within u32 bounds",
    })?;
    // SAFETY: ring_fd and fd slice are valid for the registration duration.
    let res = unsafe {
        libc::syscall(
            libc::SYS_io_uring_register,
            ring_fd,
            IORING_REGISTER_FILES,
            fds.as_ptr() as *const core::ffi::c_void,
            count,
        )
    };
    if res < 0 {
        Err(PipelineError::IoUringSyscall {
            syscall: "io_uring_register(FILES)",
            errno: get_errno(),
            fix: "ensure every fd is still open; ENOMEM means lower the fd set size",
        })
    } else {
        Ok(())
    }
}

/// Memory-map an io_uring region.
pub(crate) fn sys_mmap_ring(
    ring_fd: i32,
    size: usize,
    offset: u64,
    label: &'static str,
) -> Result<*mut libc::c_void, PipelineError> {
    // SAFETY: ring_fd is valid and offset corresponds to kernel-defined ring offset.
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

/// Memory-unmap a region.
pub(crate) fn sys_munmap(ptr: *mut libc::c_void, size: usize) {
    if !ptr.is_null() && ptr != libc::MAP_FAILED {
        // SAFETY: ptr was previously returned by mmap with matching size.
        unsafe {
            libc::munmap(ptr, size);
        }
    }
}

/// Close a file descriptor.
pub(crate) fn sys_close_fd(fd: i32) {
    if fd >= 0 {
        // SAFETY: fd is a valid descriptor to close.
        unsafe {
            libc::close(fd);
        }
    }
}

/// Atomic load u32 from raw ring pointer with given ordering.
pub(crate) fn ring_atomic_u32_load(base: *mut libc::c_void, offset: usize, order: Ordering) -> u32 {
    // SAFETY: base + offset points to a valid mapped atomic word.
    unsafe {
        let atomic_ptr = (base.add(offset)) as *const AtomicU32;
        (*atomic_ptr).load(order)
    }
}

/// Atomic store u32 to raw ring pointer with given ordering.
pub(crate) fn ring_atomic_u32_store(
    base: *mut libc::c_void,
    offset: usize,
    val: u32,
    order: Ordering,
) {
    // SAFETY: base + offset points to a valid mapped atomic word.
    unsafe {
        let atomic_ptr = (base.add(offset)) as *const AtomicU32;
        (*atomic_ptr).store(val, order);
    }
}

/// Read a u32 from a raw ring pointer.
pub(crate) fn ring_read_u32(base: *mut libc::c_void, offset: usize) -> u32 {
    // SAFETY: base + offset points to a valid mapped u32.
    unsafe {
        let ptr = base.add(offset) as *const u32;
        *ptr
    }
}

/// Write a u32 to a raw ring pointer.
pub(crate) fn ring_write_u32(base: *mut libc::c_void, offset: usize, val: u32) {
    // SAFETY: base + offset points to a valid mapped u32.
    unsafe {
        let ptr = base.add(offset) as *mut u32;
        *ptr = val;
    }
}

/// Obtain a mutable reference to an SQE in the mapped SQE table.
pub(crate) fn ring_get_sqe_mut<'a>(base: *mut libc::c_void, index: usize) -> &'a mut io_uring_sqe {
    // SAFETY: base points to the mapped SQE table; index is within sq_entries.
    unsafe {
        let sqes = base as *mut io_uring_sqe;
        &mut *sqes.add(index)
    }
}

/// Obtain a shared reference to a CQE in the mapped CQ ring.
pub(crate) fn ring_get_cqe<'a>(
    base: *mut libc::c_void,
    offset: usize,
    index: usize,
) -> &'a io_uring_cqe {
    // SAFETY: base + offset points to mapped CQE array; index is within cq_entries.
    unsafe {
        let cqes = base.add(offset) as *const io_uring_cqe;
        &*cqes.add(index)
    }
}

/// Execute `futex_waitv` syscall on Linux kernel 5.16+.
pub(crate) fn sys_futex_waitv(
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

    // SAFETY: futex_waitv call parameters are valid POD pointers.
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
