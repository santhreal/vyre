//! Safe io_uring orchestrator and lifecycle coordinator.
//!
//! This module coordinates io_uring ring lifecycle, SQE submission, CQE reaping,
//! and buffer/file registration over safe platform abstractions provided by `raw_platform`.

use crate::PipelineError;
use core::mem;
use core::sync::atomic::Ordering;

use super::raw_platform::{
    io_uring_cqe, io_uring_params, io_uring_sqe, ring_atomic_u32_load, ring_atomic_u32_store,
    ring_get_cqe, ring_get_sqe_mut, ring_read_u32, ring_write_u32, sys_close_fd,
    sys_io_uring_enter, sys_io_uring_register_buffers, sys_io_uring_register_files,
    sys_io_uring_setup, sys_mmap_ring, sys_munmap, zeroed_pod, RawRingPointers,
    IORING_ENTER_SQ_WAKEUP, IORING_FEAT_SINGLE_MMAP, IORING_OFF_CQ_RING, IORING_OFF_SQES,
    IORING_OFF_SQ_RING, IORING_SETUP_SQPOLL, IORING_SQ_NEED_WAKEUP,
};

pub(crate) use super::raw_platform::IOSQE_FIXED_FILE;

/// Orchestrator for the `io_uring` ring.
///
/// Lifetime: owns an fd + three mmap'd regions (SQ ring, CQ ring,
/// SQEs array). `Drop` closes + unmaps in reverse order.
///
/// Thread-safety: `Send + Sync` is safe because every public method
/// takes `&mut self` OR uses atomic operations on the ring pointers.
pub struct IoUringState {
    ring_fd: i32,
    ptrs: RawRingPointers,
    params: io_uring_params,
}

impl IoUringState {
    /// Create an `IoUringState` with `entries` SQEs, SQPOLL enabled,
    /// and a 2-second kernel-thread idle timeout.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::IoUringSyscall`] if `io_uring_setup`
    ///   returns < 0.
    /// - [`PipelineError::IoUringSyscall`] if any of the three `mmap`
    ///   calls fail.
    pub fn new(entries: u32) -> Result<Self, PipelineError> {
        let mut params: io_uring_params = zeroed_pod();

        params.flags |= IORING_SETUP_SQPOLL;
        params.sq_thread_idle = 2000;

        let ring_fd = sys_io_uring_setup(entries, &mut params)?;

        let sq_ring_size = kernel_ring_span_usize(
            params.sq_off.array,
            params.sq_entries,
            mem::size_of::<u32>(),
            "SQ ring",
        )?;
        let cq_ring_size = kernel_ring_span_usize(
            params.cq_off.cqes,
            params.cq_entries,
            mem::size_of::<io_uring_cqe>(),
            "CQ ring",
        )?;

        let (sq_size, cq_size) = if (params.features & IORING_FEAT_SINGLE_MMAP) != 0 {
            let max_size = core::cmp::max(sq_ring_size, cq_ring_size);
            (max_size, max_size)
        } else {
            (sq_ring_size, cq_ring_size)
        };

        let sq_ring_ptr = match sys_mmap_ring(ring_fd, sq_size, IORING_OFF_SQ_RING, "mmap(sq_ring)")
        {
            Ok(ptr) => ptr,
            Err(err) => {
                sys_close_fd(ring_fd);
                return Err(err);
            }
        };

        let cq_ring_ptr = if (params.features & IORING_FEAT_SINGLE_MMAP) != 0 {
            sq_ring_ptr
        } else {
            match sys_mmap_ring(ring_fd, cq_size, IORING_OFF_CQ_RING, "mmap(cq_ring)") {
                Ok(ptr) => ptr,
                Err(err) => {
                    sys_munmap(sq_ring_ptr, sq_size);
                    sys_close_fd(ring_fd);
                    return Err(err);
                }
            }
        };

        let sqes_size = kernel_record_span_usize(
            params.sq_entries,
            mem::size_of::<io_uring_sqe>(),
            "SQE table",
        )?;
        let sqes_ptr = match sys_mmap_ring(ring_fd, sqes_size, IORING_OFF_SQES, "mmap(sqes)") {
            Ok(ptr) => ptr,
            Err(err) => {
                if (params.features & IORING_FEAT_SINGLE_MMAP) == 0 {
                    sys_munmap(cq_ring_ptr, cq_size);
                }
                sys_munmap(sq_ring_ptr, sq_size);
                sys_close_fd(ring_fd);
                return Err(err);
            }
        };

        Ok(Self {
            ring_fd,
            ptrs: RawRingPointers {
                sq_ring_ptr,
                sq_ring_size: sq_size,
                cq_ring_ptr,
                cq_ring_size: cq_size,
                sqes_ptr,
                sqes_size,
            },
            params,
        })
    }

    /// Enter the ring to submit items or wait for completions.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IoUringSyscall`] if the syscall fails.
    pub fn enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
    ) -> Result<i32, PipelineError> {
        sys_io_uring_enter(self.ring_fd, to_submit, min_complete, flags)
    }

    /// True when this ring was created with kernel-side SQ polling.
    #[must_use]
    pub fn uses_sqpoll(&self) -> bool {
        (self.params.flags & IORING_SETUP_SQPOLL) != 0
    }

    /// Submission entries the kernel allocated for this ring.
    #[must_use]
    pub fn submission_entries(&self) -> u32 {
        self.params.sq_entries
    }

    /// True when the SQPOLL thread has slept and must be explicitly woken.
    #[must_use]
    pub fn sq_needs_wakeup(&self) -> bool {
        let offset = match kernel_offset_usize(self.params.sq_off.flags) {
            Ok(off) => off,
            Err(_) => return false,
        };
        let flags = ring_atomic_u32_load(self.ptrs.sq_ring_ptr, offset, Ordering::Acquire);
        (flags & IORING_SQ_NEED_WAKEUP) != 0
    }

    /// Wake a sleeping SQPOLL thread so already-published SQEs make progress.
    pub fn wake_sqpoll(&self) -> Result<i32, PipelineError> {
        self.enter(0, 0, IORING_ENTER_SQ_WAKEUP)
    }

    /// Obtain a mutable reference to the next available SQE.
    pub(crate) fn get_sqe(&mut self) -> Option<&mut io_uring_sqe> {
        let head_off = kernel_offset_usize(self.params.sq_off.head).ok()?;
        let head = ring_atomic_u32_load(self.ptrs.sq_ring_ptr, head_off, Ordering::Acquire);

        let tail_off = kernel_offset_usize(self.params.sq_off.tail).ok()?;
        let tail = ring_atomic_u32_load(self.ptrs.sq_ring_ptr, tail_off, Ordering::Relaxed);

        let entries_off = kernel_offset_usize(self.params.sq_off.ring_entries).ok()?;
        let ring_entries = ring_read_u32(self.ptrs.sq_ring_ptr, entries_off);

        if tail.wrapping_sub(head) < ring_entries {
            let mask_off = kernel_offset_usize(self.params.sq_off.ring_mask).ok()?;
            let ring_mask = ring_read_u32(self.ptrs.sq_ring_ptr, mask_off);
            let idx = (tail & ring_mask) as usize;
            Some(ring_get_sqe_mut(self.ptrs.sqes_ptr, idx))
        } else {
            None
        }
    }

    /// Commit the currently acquired SQE and advance the SQ tail.
    pub fn commit_sqe(&mut self) {
        if let (Ok(tail_off), Ok(array_off), Ok(mask_off)) = (
            kernel_offset_usize(self.params.sq_off.tail),
            kernel_offset_usize(self.params.sq_off.array),
            kernel_offset_usize(self.params.sq_off.ring_mask),
        ) {
            let tail = ring_atomic_u32_load(self.ptrs.sq_ring_ptr, tail_off, Ordering::Relaxed);
            let ring_mask = ring_read_u32(self.ptrs.sq_ring_ptr, mask_off);
            let idx = tail & ring_mask;

            let elem_off = array_off + (idx as usize * mem::size_of::<u32>());
            ring_write_u32(self.ptrs.sq_ring_ptr, elem_off, idx);
            ring_atomic_u32_store(
                self.ptrs.sq_ring_ptr,
                tail_off,
                tail.wrapping_add(1),
                Ordering::Release,
            );
        }
    }

    /// Read the next available CQE from the completion queue.
    pub(crate) fn peek_cqe(&mut self) -> Option<&io_uring_cqe> {
        let head_off = kernel_offset_usize(self.params.cq_off.head).ok()?;
        let head = ring_atomic_u32_load(self.ptrs.cq_ring_ptr, head_off, Ordering::Relaxed);

        let tail_off = kernel_offset_usize(self.params.cq_off.tail).ok()?;
        let tail = ring_atomic_u32_load(self.ptrs.cq_ring_ptr, tail_off, Ordering::Acquire);

        if head != tail {
            let mask_off = kernel_offset_usize(self.params.cq_off.ring_mask).ok()?;
            let ring_mask = ring_read_u32(self.ptrs.cq_ring_ptr, mask_off);
            let idx = (head & ring_mask) as usize;
            let cqes_off = kernel_offset_usize(self.params.cq_off.cqes).ok()?;
            Some(ring_get_cqe(self.ptrs.cq_ring_ptr, cqes_off, idx))
        } else {
            None
        }
    }

    /// Register a set of buffers with the kernel via `IORING_REGISTER_BUFFERS`.
    pub fn register_buffers(&self, iovecs: &[super::buffer::Iovec]) -> Result<(), PipelineError> {
        sys_io_uring_register_buffers(self.ring_fd, iovecs)
    }

    /// Register fixed files via `IORING_REGISTER_FILES`.
    pub fn register_files(&self, fds: &[i32]) -> Result<(), PipelineError> {
        sys_io_uring_register_files(self.ring_fd, fds)
    }

    /// Advance the CQ head, acknowledging completion.
    pub fn advance_cq(&mut self) {
        if let Ok(head_off) = kernel_offset_usize(self.params.cq_off.head) {
            let head = ring_atomic_u32_load(self.ptrs.cq_ring_ptr, head_off, Ordering::Relaxed);
            ring_atomic_u32_store(
                self.ptrs.cq_ring_ptr,
                head_off,
                head.wrapping_add(1),
                Ordering::Release,
            );
        }
    }
}

impl Drop for IoUringState {
    fn drop(&mut self) {
        sys_munmap(self.ptrs.sqes_ptr, self.ptrs.sqes_size);
        if self.ptrs.sq_ring_ptr != self.ptrs.cq_ring_ptr {
            sys_munmap(self.ptrs.cq_ring_ptr, self.ptrs.cq_ring_size);
        }
        sys_munmap(self.ptrs.sq_ring_ptr, self.ptrs.sq_ring_size);
        sys_close_fd(self.ring_fd);
    }
}

fn kernel_ring_span_usize(
    base_offset: u32,
    entries: u32,
    record_bytes: usize,
    label: &'static str,
) -> Result<usize, PipelineError> {
    let base_usize = kernel_offset_usize(base_offset)?;
    let entries_usize = kernel_entries_usize(entries, label)?;
    let array_bytes = entries_usize
        .checked_mul(record_bytes)
        .ok_or_else(|| ring_span_overflow(label, base_usize, entries_usize, record_bytes))?;
    base_usize
        .checked_add(array_bytes)
        .ok_or_else(|| ring_span_overflow(label, base_usize, entries_usize, record_bytes))
}

fn kernel_record_span_usize(
    entries: u32,
    record_bytes: usize,
    label: &'static str,
) -> Result<usize, PipelineError> {
    let entries_usize = kernel_entries_usize(entries, label)?;
    entries_usize
        .checked_mul(record_bytes)
        .ok_or_else(|| ring_span_overflow(label, 0, entries_usize, record_bytes))
}

fn kernel_offset_usize(offset: u32) -> Result<usize, PipelineError> {
    usize::try_from(offset).map_err(|_| PipelineError::IntegerWidth {
        quantity: "io_uring kernel offset",
        value: u128::from(offset),
        bits: usize::BITS,
        fix: "check kernel header alignment on 32-bit platforms",
    })
}

fn kernel_entries_usize(entries: u32, label: &'static str) -> Result<usize, PipelineError> {
    usize::try_from(entries).map_err(|_| PipelineError::IntegerWidth {
        quantity: label,
        value: u128::from(entries),
        bits: usize::BITS,
        fix: "lower the ring entries count to fit the host address space",
    })
}

fn ring_span_overflow(
    label: &'static str,
    _base: usize,
    _entries: usize,
    _record_bytes: usize,
) -> PipelineError {
    PipelineError::IoUringSyscall {
        syscall: label,
        errno: libc::EOVERFLOW,
        fix: "ring size calculation overflowed host address space; reduce entries",
    }
}
