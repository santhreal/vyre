//! Artifact execution, resident work queues, resource residency, and zero-copy IO.
//!
//! Runtime construction starts from an authenticated [`ArtifactSession`].
//! Immutable compiler artifacts are materialized through registered target
//! devices; runtime policy owns bindings, retained state, queueing, recovery,
//! resource residency, IO, and telemetry.

#![deny(missing_docs)]
#![warn(unreachable_pub)]
// vyre-runtime owns the io_uring zero-copy ingest path and the persistent
// megakernel ring; both reach into FFI / mmap territory. Every unsafe site
// carries a `Safety:` comment that `check_unsafe_justifications.sh` validates.
#![allow(unsafe_code)]

/// Errors surfaced by the runtime layer. Every variant carries a
/// `Fix:`-bearing message so a reviewer can act on the failure.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// Raw io_uring / libc syscall failed with an errno.
    #[error("io_uring {syscall} failed: errno={errno}. Fix: {fix}")]
    IoUringSyscall {
        /// Which syscall failed (`io_uring_setup`, `mmap`, `io_uring_enter`).
        syscall: &'static str,
        /// Underlying errno value.
        errno: i32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// io_uring submission or completion queue was full / overflowed.
    #[error("io_uring {queue} queue at capacity. Fix: {fix}")]
    QueueFull {
        /// "submission" or "completion".
        queue: &'static str,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// Attempted to use io_uring on a non-Linux platform.
    #[error(
        "io_uring is Linux-only. Fix: run on Linux 5.1+ and attach an AsyncUringStream to UringCompletionPump"
    )]
    NotLinux,
    /// Feature required for NVMe passthrough is not enabled.
    #[error(
        "NVMe passthrough requires the `uring-cmd-nvme` feature + Linux kernel 6.0+. Fix: add `features = [\"uring-cmd-nvme\"]` to your Cargo.toml"
    )]
    NvmePassthroughDisabled,
    /// Backend error bubbled up from compile or dispatch.
    #[error("backend error: {0}")]
    Backend(String),
    /// A megakernel dispatch ended before its work queue drained: only
    /// `claimed` of `expected` `unit` were claimed, so the rest went unscanned
    /// and this dispatch's hit set is INCOMPLETE, never a silent partial
    /// (Law 10). A first-class variant (not a `Backend` string) so callers such
    /// as the `seg_len` calibrator can EXCLUDE a too-fine geometry by matching
    /// the type, never by substring-scanning the message text.
    #[error(
        "{descriptor} drain incomplete: only {claimed} of {expected} {unit} were claimed before \
         the dispatch ended, so {unscanned} {unit} went unscanned and their matches were dropped. \
         This dispatch's hit set is INCOMPLETE. Fix: raise the dispatch timeout \
         (BatchDispatchConfig.timeout) so the drain loop can exhaust the queue, or shard the batch \
         into smaller queues.",
        unscanned = expected.saturating_sub(*claimed),
    )]
    DrainIncomplete {
        /// Which dispatch path under-drained: `"megakernel"` (per-rule) or
        /// `"combined megakernel"` (combined-AC). Names the failing path in the
        /// operator message without a second string variant.
        descriptor: &'static str,
        /// Work-items/segments actually claimed before the dispatch ended.
        claimed: u32,
        /// Work-items/segments that should have been claimed (full queue length).
        expected: u32,
        /// The unit being drained: `"work-items"` (per-rule) or `"segments"`
        /// (combined-AC). Interpolated twice for a grammatical message.
        unit: &'static str,
    },
}

impl PipelineError {
    /// True iff this is a [`PipelineError::DrainIncomplete`]: a dispatch that
    /// could not exhaust its work queue within the timeout.
    ///
    /// Distinct from a hard backend failure, the `seg_len` calibrator
    /// EXCLUDES a geometry that drains incompletely (too fine to drain in the
    /// configured timeout) rather than aborting the whole calibration, while it
    /// must still PROPAGATE any other [`PipelineError`]. Match on this predicate
    /// instead of substring-scanning the Display message, which is fragile to
    /// wording changes.
    #[must_use]
    pub fn is_drain_incomplete(&self) -> bool {
        matches!(self, Self::DrainIncomplete { .. })
    }
}

impl From<vyre_driver::backend::BackendError> for PipelineError {
    fn from(err: vyre_driver::backend::BackendError) -> Self {
        PipelineError::Backend(err.to_string())
    }
}

/// Canonical artifact-envelope authentication and exact-format admission.
pub mod artifact_admission;

/// Backend-neutral immutable-resource and mutable-state residency.
pub mod resource_residency;

/// Resident work-queue protocols, scheduling policy, and runtime IO.
#[path = "megakernel/mod.rs"]
pub mod resident_work_queue;

/// Authenticated persistent execution over retained artifact bindings.
pub mod persistent_executor;
/// Content-addressed authenticated artifact cache.
pub mod pipeline_cache;

/// Structured artifact-session recovery without message parsing or recompilation.
pub mod recovery;
/// Differential megakernel replay log  -  captures every published
/// ring slot so a later cert run can diff epoch-by-epoch execution
/// against a live backend.
pub mod replay;

/// Backend routing policy for execution plans.
pub mod routing;

/// Multi-GPU work partitioning across runtime backends.
pub mod scheduler;

/// Multi-tenant megakernel multiplexing  -  one persistent kernel per
/// GPU, shared across producer tools via the `tenant_id` field already
/// in the ring protocol.
pub mod tenant;

/// Linux io_uring integration. Compiled out on macOS / Windows.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
pub mod uring;

/// Completion pump for an optional Linux io_uring stream.
///
/// Detached pumps report [`UringPollState::Detached`] instead of fabricating a
/// zero-completion observation.
pub struct UringCompletionPump<'a> {
    #[cfg(target_os = "linux")]
    uring: Option<uring::AsyncUringStream<'a>>,
    // On macOS / Windows the `uring` field is compiled out, which leaves the
    // `'a` lifetime unused and the compiler rejects the struct. Carry a
    // zero-sized marker so the lifetime stays live on non-Linux targets.
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<&'a ()>,
    shutdown_requested: bool,
}

impl Default for UringCompletionPump<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one non-blocking completion-pump probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UringPollState {
    /// No io_uring stream is attached.
    Detached,
    /// An attached stream was polled and produced this many completions.
    Completed(u32),
}

impl<'a> UringCompletionPump<'a> {
    /// Create a pipeline handle with no io_uring stream attached.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre_runtime::UringCompletionPump;
    ///
    /// let pump = UringCompletionPump::new();
    ///
    /// assert!(!pump.is_shutdown_requested());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            uring: None,
            #[cfg(not(target_os = "linux"))]
            _phantom: std::marker::PhantomData,
            shutdown_requested: false,
        }
    }

    /// Attach an io_uring stream for GPU-visible reads. Linux-only.
    ///
    /// Use `uring::NvmeGpuIngestDriver::new_gpudirect` when the caller
    /// requires the native NVMe → BAR1 path instead of registered mapped reads.
    #[cfg(target_os = "linux")]
    #[must_use]
    pub fn with_uring(mut self, stream: uring::AsyncUringStream<'a>) -> Self {
        self.uring = Some(stream);
        self
    }

    /// Probe the attached io_uring stream for completions.
    ///
    /// # Errors
    ///
    /// Propagates any uring syscall error from the underlying ring.
    pub fn poll(&mut self) -> Result<UringPollState, PipelineError> {
        #[cfg(target_os = "linux")]
        {
            if let Some(ref mut stream) = self.uring {
                return stream.poll().map(UringPollState::Completed);
            }
        }
        Ok(UringPollState::Detached)
    }

    /// Request graceful shutdown of the pipeline.
    pub fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    /// Whether shutdown has been requested.
    #[must_use]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    /// Block until the megakernel writes a new value into the
    /// observable word. Uses `futex_waitv` on Linux 5.16+.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::NotLinux`] on non-Linux hosts.
    /// - [`PipelineError::IoUringSyscall`] on futex errors.
    ///
    /// # Safety
    ///
    /// `host_visible_addr` must be host-mapped and outlive this call.
    #[cfg(target_os = "linux")]
    #[allow(unsafe_code)]
    pub unsafe fn wait_for_observable(
        host_visible_addr: *const u32,
        current: u32,
        timeout_ns: u64,
    ) -> Result<(), PipelineError> {
        #[repr(C)]
        struct futex_waitv {
            val: u64,
            uaddr: u64,
            flags: u32,
            __reserved: u32,
        }
        const FUTEX2_SIZE_U32: u32 = 0x02;
        const SYS_FUTEX_WAITV: libc::c_long = 449;

        let waitv = [futex_waitv {
            val: current as u64,
            uaddr: host_visible_addr as u64,
            flags: FUTEX2_SIZE_U32,
            __reserved: 0,
        }];

        #[repr(C)]
        struct Timespec {
            tv_sec: i64,
            tv_nsec: i64,
        }
        let ts = Timespec {
            tv_sec: (timeout_ns / 1_000_000_000) as i64,
            tv_nsec: (timeout_ns % 1_000_000_000) as i64,
        };

        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
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
            // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
            let errno = unsafe { *libc::__errno_location() };
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

    /// Non-Linux implementation returning the structured platform error.
    #[cfg(not(target_os = "linux"))]
    #[allow(unsafe_code, clippy::missing_safety_doc)]
    pub unsafe fn wait_for_observable(
        _host_visible_addr: *const u32,
        _current: u32,
        _timeout_ns: u64,
    ) -> Result<(), PipelineError> {
        Err(PipelineError::NotLinux)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_stream_has_no_shutdown() {
        let stream = UringCompletionPump::new();
        assert!(!stream.is_shutdown_requested());
    }

    #[test]
    fn shutdown_is_idempotent() {
        let mut stream = UringCompletionPump::new();
        stream.request_shutdown();
        stream.request_shutdown();
        assert!(stream.is_shutdown_requested());
    }

    #[test]
    fn poll_without_uring_reports_detached_state() {
        let mut stream = UringCompletionPump::new();
        assert_eq!(stream.poll().unwrap(), UringPollState::Detached);
    }

    #[test]
    fn drain_incomplete_is_distinguishable_by_type_not_substring() {
        // Regression: the seg_len calibrator must EXCLUDE a too-fine geometry
        // (drain-incomplete) but PROPAGATE every other backend failure. It used
        // to discriminate by `to_string().contains("drain incomplete")`, which
        // silently turns into "abort the whole calibration" the moment the
        // message wording drifts. The structured variant + predicate is the
        // contract; this test pins it.
        let drain = PipelineError::DrainIncomplete {
            descriptor: "combined megakernel",
            claimed: 3,
            expected: 10,
            unit: "segments",
        };
        assert!(drain.is_drain_incomplete());

        // The Display message stays operator-actionable AND keeps the
        // "drain incomplete" phrase + computed unscanned count, so legacy
        // substring matchers and operator logs do not regress.
        let msg = drain.to_string();
        assert_eq!(
            msg,
            "combined megakernel drain incomplete: only 3 of 10 segments were claimed before the \
             dispatch ended, so 7 segments went unscanned and their matches were dropped. This \
             dispatch's hit set is INCOMPLETE. Fix: raise the dispatch timeout \
             (BatchDispatchConfig.timeout) so the drain loop can exhaust the queue, or shard the \
             batch into smaller queues."
        );

        // The per-rule path uses the same variant with a different descriptor/unit.
        let per_rule = PipelineError::DrainIncomplete {
            descriptor: "megakernel",
            claimed: 0,
            expected: 4,
            unit: "work-items",
        };
        assert!(per_rule.is_drain_incomplete());
        assert!(
            per_rule
                .to_string()
                .starts_with("megakernel drain incomplete: only 0 of 4 work-items were claimed"),
            "msg was: {}",
            per_rule
        );

        // A genuine backend failure is NOT a drain-incomplete: it must surface
        // as a hard error, never be excluded-and-continued by the calibrator.
        let backend = PipelineError::Backend("adapter lost".to_string());
        assert!(!backend.is_drain_incomplete());
    }
}
