//! Completion pump for an optional Linux io_uring stream.

use crate::PipelineError;

/// Result of one non-blocking completion-pump probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UringPollState {
    /// No io_uring stream is attached.
    Detached,
    /// An attached stream was polled and produced this many completions.
    Completed(u32),
}

/// Completion pump for an optional Linux io_uring stream.
///
/// Detached pumps report [`UringPollState::Detached`] instead of fabricating a
/// zero-completion observation.
pub struct UringCompletionPump<'a> {
    #[cfg(target_os = "linux")]
    uring: Option<crate::uring::AsyncUringStream<'a>>,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<&'a ()>,
    shutdown_requested: bool,
}

impl Default for UringCompletionPump<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> UringCompletionPump<'a> {
    /// Create a pipeline handle with no io_uring stream attached.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre_runtime::uring_completion_pump::UringCompletionPump;
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
    pub fn with_uring(mut self, stream: crate::uring::AsyncUringStream<'a>) -> Self {
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
            if let Some(stream) = &mut self.uring {
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

    /// Block until the megakernel writes a value other than `current` into the
    /// observable word. Uses `futex_waitv` on Linux 5.16+.
    ///
    /// # Safety
    ///
    /// The caller must uphold that `host_visible_addr` names a mapped,
    /// naturally aligned `u32` that stays mapped for the whole wait. The
    /// kernel reads that address after this call has parked the thread, so an
    /// unmapping that races the wait is not observable from here.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::NotLinux`] on non-Linux hosts.
    /// - [`PipelineError::IoUringSyscall`] on futex errors.
    #[cfg(target_os = "linux")]
    #[allow(unsafe_code)]
    pub unsafe fn wait_for_observable(
        host_visible_addr: *const u32,
        current: u32,
        timeout_ns: u64,
    ) -> Result<(), PipelineError> {
        // SAFETY: the obligation is restated verbatim on this function, so the
        // caller has already upheld what the syscall wrapper requires.
        unsafe {
            crate::uring::raw_platform::sys_futex_waitv(host_visible_addr, current, timeout_ns)
        }
    }

    /// Non-Linux hosts report the structured platform error.
    ///
    /// # Safety
    ///
    /// The obligation matches the Linux arm so one call site compiles on both,
    /// even though this arm never reads the address.
    ///
    /// # Errors
    ///
    /// Always [`PipelineError::NotLinux`].
    #[cfg(not(target_os = "linux"))]
    #[allow(unsafe_code)]
    pub unsafe fn wait_for_observable(
        _host_visible_addr: *const u32,
        _current: u32,
        _timeout_ns: u64,
    ) -> Result<(), PipelineError> {
        Err(PipelineError::NotLinux)
    }
}
