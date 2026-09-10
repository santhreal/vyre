//! io_uring lifecycle, SQE submission, CQE reaping, and registration.
//!
//! This module owns the submission and completion protocol. Every mapped
//! address that protocol reads or writes belongs to [`MappedRing`], so no raw
//! pointer appears here.

use crate::PipelineError;
use core::mem;
use core::sync::atomic::Ordering;

use super::raw_platform::{
    io_uring_cqe, io_uring_params, io_uring_sqe, MappedRing, IORING_ENTER_SQ_WAKEUP,
    IORING_SETUP_SQPOLL, IORING_SQ_NEED_WAKEUP,
};

pub(crate) use super::raw_platform::IOSQE_FIXED_FILE;

/// Orchestrator for the `io_uring` ring.
///
/// Lifetime: [`MappedRing`] owns the descriptor and the three mapped regions,
/// and releases them in reverse order when this value drops.
///
/// Thread-safety: `Send + Sync` holds because every ring header word is read
/// and written atomically and a submission entry is handed out only behind
/// `&mut self`.
pub struct IoUringState {
    ring: MappedRing,
    params: io_uring_params,
}

impl IoUringState {
    /// Create an `IoUringState` with `entries` SQEs, SQPOLL enabled,
    /// and a 2-second kernel-thread idle timeout.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::IoUringSyscall`] if `io_uring_setup` or any of the
    ///   three mappings is refused.
    /// - [`PipelineError::IntegerWidth`] if the kernel reports a ring the host
    ///   address space cannot span.
    pub fn new(entries: u32) -> Result<Self, PipelineError> {
        let (ring, params) = MappedRing::setup(entries, IORING_SETUP_SQPOLL, 2000)?;
        Ok(Self { ring, params })
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
        self.ring.enter(to_submit, min_complete, flags)
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
        let flags = self.ring.sq_load(offset, Ordering::Acquire);
        (flags & IORING_SQ_NEED_WAKEUP) != 0
    }

    /// Wake a sleeping SQPOLL thread so already-published SQEs make progress.
    pub fn wake_sqpoll(&self) -> Result<i32, PipelineError> {
        self.enter(0, 0, IORING_ENTER_SQ_WAKEUP)
    }

    /// Obtain a mutable reference to the next available SQE.
    pub(crate) fn get_sqe(&mut self) -> Option<&mut io_uring_sqe> {
        let head_off = kernel_offset_usize(self.params.sq_off.head).ok()?;
        let head = self.ring.sq_load(head_off, Ordering::Acquire);

        let tail_off = kernel_offset_usize(self.params.sq_off.tail).ok()?;
        let tail = self.ring.sq_load(tail_off, Ordering::Relaxed);

        let entries_off = kernel_offset_usize(self.params.sq_off.ring_entries).ok()?;
        let ring_entries = self.ring.sq_read(entries_off);

        if tail.wrapping_sub(head) < ring_entries {
            let mask_off = kernel_offset_usize(self.params.sq_off.ring_mask).ok()?;
            let ring_mask = self.ring.sq_read(mask_off);
            let idx = (tail & ring_mask) as usize;
            Some(self.ring.sqe_mut(idx))
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
            let tail = self.ring.sq_load(tail_off, Ordering::Relaxed);
            let ring_mask = self.ring.sq_read(mask_off);
            let idx = tail & ring_mask;

            let elem_off = array_off + (idx as usize * mem::size_of::<u32>());
            self.ring.sq_write(elem_off, idx);
            self.ring
                .sq_store(tail_off, tail.wrapping_add(1), Ordering::Release);
        }
    }

    /// Read the next available CQE from the completion queue.
    pub(crate) fn peek_cqe(&mut self) -> Option<&io_uring_cqe> {
        let head_off = kernel_offset_usize(self.params.cq_off.head).ok()?;
        let head = self.ring.cq_load(head_off, Ordering::Relaxed);

        let tail_off = kernel_offset_usize(self.params.cq_off.tail).ok()?;
        let tail = self.ring.cq_load(tail_off, Ordering::Acquire);

        if head != tail {
            let mask_off = kernel_offset_usize(self.params.cq_off.ring_mask).ok()?;
            let ring_mask = self.ring.cq_read(mask_off);
            let idx = (head & ring_mask) as usize;
            let cqes_off = kernel_offset_usize(self.params.cq_off.cqes).ok()?;
            Some(self.ring.cqe(cqes_off, idx))
        } else {
            None
        }
    }

    /// Register a set of buffers with the kernel via `IORING_REGISTER_BUFFERS`.
    ///
    /// # Safety
    ///
    /// The caller must uphold that every range in `iovecs` stays mapped and
    /// writable, and is untouched by the host, until this ring is torn down.
    /// The kernel writes transfer results into those ranges after submission,
    /// long after this call returns.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IntegerWidth`] when the array is longer than
    /// `u32` and [`PipelineError::IoUringSyscall`] when the kernel refuses it.
    #[allow(unsafe_code)]
    pub unsafe fn register_buffers(
        &self,
        iovecs: &[super::buffer::Iovec],
    ) -> Result<(), PipelineError> {
        // SAFETY: the obligation is restated verbatim on this function, so the
        // caller has already upheld what the ring's own registration requires.
        unsafe { self.ring.register_buffers(iovecs) }
    }

    /// Register fixed files via `IORING_REGISTER_FILES`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IntegerWidth`] when the set is longer than
    /// `u32` and [`PipelineError::IoUringSyscall`] when the kernel refuses it.
    pub fn register_files(&self, fds: &[i32]) -> Result<(), PipelineError> {
        self.ring.register_files(fds)
    }

    /// Advance the CQ head, acknowledging completion.
    pub fn advance_cq(&mut self) {
        if let Ok(head_off) = kernel_offset_usize(self.params.cq_off.head) {
            let head = self.ring.cq_load(head_off, Ordering::Relaxed);
            self.ring
                .cq_store(head_off, head.wrapping_add(1), Ordering::Release);
        }
    }
}

pub(super) fn kernel_ring_span_usize(
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

pub(super) fn kernel_record_span_usize(
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
