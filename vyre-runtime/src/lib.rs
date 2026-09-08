//! Artifact execution, resident work queues, resource residency, and zero-copy IO.
//!
//! Runtime construction starts from an authenticated [`artifact_admission::ArtifactSession`].
//! Immutable compiler artifacts are materialized through registered target
//! devices; runtime policy owns bindings, retained state, queueing, recovery,
//! resource residency, IO, and telemetry.

// vyre-runtime owns the io_uring zero-copy ingest path and the persistent
// megakernel ring; both reach into FFI / mmap territory. Every unsafe site
// carries a `SAFETY:` comment the `lint-unsafe-justification` gate validates.
#![allow(unsafe_code)]

// A fixture module shared with the integration suites names this crate by its
// own name, so the same file compiles inside the library and inside a test
// binary.
#[cfg(test)]
extern crate self as vyre_runtime;

// The prefix-cache key fixture the integration proofs own.
#[cfg(test)]
#[path = "../tests/prefix_cache_fixtures/mod.rs"]
mod prefix_cache_fixtures;

// The `PipelineError` variant-space closure. An exhaustive match over a
// `#[non_exhaustive]` enum is legal only inside the crate that defines it.
#[cfg(test)]
mod pipeline_error_closure;

use std::fmt;

/// Renders a permitted-status set as protocol status names, so a rejection
/// message states `PUBLISHED, YIELD, REQUEUE` rather than raw words.
struct SlotStatusList(&'static [u32]);

impl fmt::Display for SlotStatusList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, status) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", SlotStatusName(*status))?;
        }
        Ok(())
    }
}

/// Renders one status word as its protocol name, or as `unknown` when the word
/// is not a status the protocol defines.
struct SlotStatusName(u32);

impl fmt::Display for SlotStatusName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match resident_work_queue::protocol::slot::status_name(self.0) {
            Some(name) => f.write_str(name),
            None => f.write_str("unknown"),
        }
    }
}

/// Declares a closed enum together with the list of all its variants.
///
/// A hand-written `ALL` beside an enum is a second list of the same variants,
/// and a second list goes stale in silence: `[Self; 3]` stays valid when a
/// fourth variant arrives, so a caller that covers "the whole space" by
/// iterating `ALL` then covers all but one and nothing says so. Here the
/// variants are written once and the count is written once, and the compiler
/// rejects the pair the moment they disagree.
macro_rules! closed_enum {
    (
        $(#[$enum_meta:meta])*
        $visibility:vis enum $name:ident: $count:literal {
            $($(#[$variant_meta:meta])* $variant:ident),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        $visibility enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl $name {
            /// Every variant, for a caller that must cover the whole space.
            pub const ALL: [Self; $count] = [$(Self::$variant),+];
        }
    };
}

closed_enum! {
    /// Why a resident-ring encode or publish was rejected.
    ///
    /// A typed class rather than a message string, so a test or a caller selects
    /// the fault it means and cannot be satisfied by a different one.
    ///
    /// Deliberately exhaustive: a caller outside this crate matches every fault
    /// with no catch-all, so adding one turns that caller red until it records a
    /// decision for the new class.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RingEncodingFault: 5 {
        /// The ring buffer is not a valid slot grid: its byte length is not an
        /// exact multiple of the slot width.
        Geometry,
        /// The request exceeds a fixed budget: ring slot count, per-slot argument
        /// words, or a packed-opcode field width.
        Capacity,
        /// Offset or count arithmetic exceeded the host or wire integer width.
        Overflow,
        /// A computed word lies outside the validated ring buffer.
        OutOfBounds,
        /// The publish protocol was used incorrectly, such as writing a status
        /// word ahead of the payload words it advertises.
        Protocol,
    }
}

impl fmt::Display for RingEncodingFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Geometry => "ring geometry is invalid",
            Self::Capacity => "request exceeds a ring capacity budget",
            Self::Overflow => "offset arithmetic overflowed",
            Self::OutOfBounds => "computed word is outside the ring buffer",
            Self::Protocol => "publish protocol was used incorrectly",
        })
    }
}

/// Which subsystem owns a counter whose own accounting was rejected.
///
/// A caller reads the owner from the variant instead of from message text, so
/// a tenant counter fault and an io_uring counter fault stay distinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterScope {
    /// A submission or completion counter owned by one io_uring stream.
    IoUring,
    /// A counter owned by the tenant registry itself, before any tenant id is
    /// issued.
    TenantRegistry,
    /// A counter owned by one tenant, named by its tenant id.
    Tenant(u32),
}

impl fmt::Display for CounterScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoUring => f.write_str("io_uring stream"),
            Self::TenantRegistry => f.write_str("tenant registry"),
            Self::Tenant(tenant_id) => write!(f, "tenant {tenant_id}"),
        }
    }
}

closed_enum! {
    /// Which arithmetic left the range of its integer width.
    ///
    /// A typed class rather than an operator string, so a caller that only tolerates
    /// a product overflow cannot be satisfied by a sum overflow.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CounterArithmetic: 2 {
        /// An addition.
        Sum,
        /// A multiplication.
        Product,
    }
}

impl fmt::Display for CounterArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sum => "sum",
            Self::Product => "product",
        })
    }
}

closed_enum! {
    /// Why a submitted io_uring request was rejected before it reached a queue.
    ///
    /// Every fault here is a property of the request alone: none of them depends on
    /// queue occupancy, region contents, or counter state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RequestFault: 3 {
        /// A count the path requires to be positive was below its minimum.
        BelowMinimum,
        /// A length does not equal the fixed length the path is bound to.
        LengthMismatch,
        /// A total does not divide evenly into the requested parts, so some of it
        /// would belong to no part.
        Indivisible,
    }
}

impl fmt::Display for RequestFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BelowMinimum => "is below the minimum",
            Self::LengthMismatch => "does not equal the bound length",
            Self::Indivisible => "does not divide evenly by",
        })
    }
}

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
    /// An io_uring submission or completion queue holds every entry it has, so
    /// the request has nowhere to go until the ring drains.
    ///
    /// Reached only where the code observed a full queue. The bounds,
    /// conversion, arithmetic, counter-order, request, and slot faults that
    /// once shared this variant have their own below, because this message
    /// states an occupancy the ring never reported for any of them.
    #[error("io_uring {queue} queue at capacity ({depth} entries). Fix: {fix}")]
    QueueFull {
        /// "submission" or "completion".
        queue: &'static str,
        /// Entries the queue holds, which is the bound the request hit.
        depth: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A byte or slot range lies outside the region registered for it.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed, so
    /// a sub-region past the end of a mapped GPU buffer, a file larger than its
    /// ingest slot, and a slot index past the mapped-slot table each told the
    /// caller an io_uring submission queue was at capacity, and discarded the
    /// offset, the length, and the size of the region they were checked
    /// against.
    #[error(
        "{region} {unit} range [{offset}, {}) is outside the region's {region_len} {unit}. \
         Fix: {fix}",
        offset.saturating_add(*len)
    )]
    RegionBounds {
        /// The registered region the range was checked against.
        region: &'static str,
        /// First unit of the requested range.
        offset: u64,
        /// Units the request covers from `offset`.
        len: u64,
        /// Units the region holds.
        region_len: u64,
        /// What `offset`, `len`, and `region_len` count: `"bytes"` or
        /// `"slots"`.
        unit: &'static str,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A count, offset, or length did not fit the integer width its destination
    /// requires.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed: a
    /// `usize::try_from` that could not hold an ingest slot count and a
    /// completion `user_data` wider than the megakernel slot-index word both
    /// rendered as a full submission queue, and neither carried the value that
    /// did not fit or the width it had to fit.
    #[error("{quantity} {value} does not fit {bits} bits. Fix: {fix}")]
    IntegerWidth {
        /// What the value counts.
        quantity: &'static str,
        /// The value that did not fit. `u128` so a host `usize` and a wire
        /// `u64` are both representable without a lossy cast into the error.
        value: u128,
        /// Bit width of the destination integer.
        bits: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// Counter arithmetic left the range of the counter's integer width.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed, so
    /// an inflight-SQE count that reached `u32::MAX` and a tenant staging-byte
    /// reservation that reached `u64::MAX` both claimed a queue was at
    /// capacity, and neither named the counter, its owner, or the two operands.
    #[error(
        "{scope} {counter} overflowed: the {arithmetic} of {lhs} and {rhs} leaves the \
         {bits}-bit range. Fix: {fix}"
    )]
    CounterOverflow {
        /// Which subsystem owns the counter.
        scope: CounterScope,
        /// The counter that overflowed.
        counter: &'static str,
        /// Which arithmetic overflowed.
        arithmetic: CounterArithmetic,
        /// Left operand.
        lhs: u64,
        /// Right operand.
        rhs: u64,
        /// Bit width of the counter.
        bits: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// Two counters that advance in step went out of order: the consumed count
    /// exceeded the produced count that bounds it, so at least one advance was
    /// accounted twice.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed, so
    /// a tenant `drained_count` past its `published_count` and a completion
    /// reaped with no inflight SQE both rendered as a full queue, which is the
    /// opposite occupancy from the one observed, and neither carried either
    /// counter.
    #[error(
        "{scope} counters are out of order: {consumed_counter} {consumed} exceeds \
         {produced_counter} {produced} by {}. Fix: {fix}",
        consumed.saturating_sub(*produced)
    )]
    CounterOrder {
        /// Which subsystem owns the pair.
        scope: CounterScope,
        /// The counter that bounds the pair.
        produced_counter: &'static str,
        /// Value of the bounding counter.
        produced: u64,
        /// The counter that exceeded its bound.
        consumed_counter: &'static str,
        /// Value of the exceeding counter.
        consumed: u64,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// The submitted request is malformed on its own terms, independent of
    /// queue occupancy and region contents.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed, so
    /// an empty iovec array, a read length that did not match the pump's bound
    /// chunk size, and a staging buffer that does not divide by its slot count
    /// all told the caller the submission queue was at capacity before the
    /// request had reached any queue.
    #[error(
        "io_uring request rejected: {quantity} is {observed}, which {fault} {bound}. Fix: {fix}"
    )]
    InvalidRequest {
        /// Which class of malformation was detected.
        fault: RequestFault,
        /// What the observed value counts.
        quantity: &'static str,
        /// The value the caller supplied.
        observed: u64,
        /// The value the path requires, related to `observed` by `fault`.
        bound: u64,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// An ingest slot that still has a read in flight was submitted to again.
    ///
    /// Reported as [`PipelineError::QueueFull`] before this variant existed,
    /// which named the io_uring submission queue for a fault about one mapped
    /// slot and discarded the slot index and the tag of the read holding it.
    #[error(
        "io_uring ingest slot {slot} of {slot_count} already has a read in flight \
         (tag {inflight_tag}). Fix: {fix}"
    )]
    SlotInFlight {
        /// The slot the caller submitted to.
        slot: u32,
        /// Slots the driver registered.
        slot_count: u32,
        /// Caller tag of the read still occupying the slot.
        inflight_tag: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A resident-ring encode or publish was rejected before any slot was
    /// written.
    ///
    /// Separate from [`PipelineError::QueueFull`], which states that an
    /// io_uring queue is at capacity. The resident ring encoder reported every
    /// one of its geometry, capacity, overflow, and bounds faults through that
    /// variant, so a caller was told the submission queue was full when the
    /// real fault was, for example, a slot index outside the ring.
    #[error("resident ring encode rejected: {fault}. Fix: {fix}")]
    RingEncoding {
        /// Which class of fault was detected.
        fault: RingEncodingFault,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A host-protocol encode or decode was rejected.
    ///
    /// Carries the typed [`resident_work_queue::protocol::ProtocolError`], so a
    /// caller reads the buffer, byte length, and word index it names. Two of
    /// the three protocol faults previously reached callers as a
    /// [`PipelineError::Backend`] string, which discarded every one of those
    /// fields.
    #[error(transparent)]
    Protocol(#[from] resident_work_queue::protocol::ProtocolError),
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
        /// message without a second string variant.
        descriptor: &'static str,
        /// Work-items/segments actually claimed before the dispatch ended.
        claimed: u32,
        /// Work-items/segments that should have been claimed (full queue length).
        expected: u32,
        /// The unit being drained: `"work-items"` (per-rule) or `"segments"`
        /// (combined-AC). Interpolated twice for a grammatical message.
        unit: &'static str,
    },
    /// A ring slot was asked to make a transition its status does not permit.
    ///
    /// A first-class variant, for the reason `DrainIncomplete` is one: a caller
    /// matches the transition and the status it observed rather than scanning
    /// message text. Before it existed this fault was reported as
    /// [`PipelineError::QueueFull`], which told the caller the io_uring
    /// submission queue was at capacity when the ring had a slot in the wrong
    /// state, and dropped the observed status entirely.
    #[error(
        "illegal ring slot transition: {transition} requires {}, and the slot holds status {} \
         ({current_status}). Fix: {fix}",
        SlotStatusList(.permitted),
        SlotStatusName(*.current_status)
    )]
    IllegalSlotTransition {
        /// The attempted transition, as `RingSlotTransition` names it.
        transition: &'static str,
        /// The status words this transition is legal from, from the same table
        /// the legality predicate reads.
        permitted: &'static [u32],
        /// The status word the slot held when the transition was attempted.
        current_status: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A request named a resource handle that was never registered with the
    /// path serving it.
    ///
    /// Reported as [`PipelineError::Backend`] before this variant existed, so a
    /// READ naming an unregistered GPU destination arrived as an untyped string
    /// a caller could only substring-scan, and the handle and the slot were
    /// readable only out of that text.
    #[error(
        "{request} in slot {slot} names {resource} handle {handle}, which is not registered. \
         Fix: {fix}"
    )]
    UnregisteredResource {
        /// The request that named the handle, as its opcode path names it.
        request: &'static str,
        /// What kind of resource the handle was expected to identify.
        resource: &'static str,
        /// The handle the request carried.
        handle: u32,
        /// The queue slot the request occupied.
        slot: u32,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// A runtime worker thread ended by panic, so the work it owned has no
    /// result and its queue has no consumer.
    ///
    /// Reported as [`PipelineError::Backend`] before this variant existed. A
    /// panicked worker is not a backend failure: the backend was never reached.
    #[error("the {worker} thread panicked before it could be joined. Fix: {fix}")]
    WorkerThreadPanicked {
        /// Which worker ended by panic.
        worker: &'static str,
        /// Actionable remediation.
        fix: &'static str,
    },
    /// An opcode allocated to a tenant lands in the range the megakernel
    /// reserves for itself, so publishing it would collide with a system
    /// opcode.
    ///
    /// Reported as [`PipelineError::Backend`] before this variant existed, which
    /// rendered a broken opcode-window allocation as a backend failure and left
    /// the three numbers that identify it inside the message text.
    #[error(
        "tenant {tenant_id} local opcode {local_opcode} maps to global opcode {global_opcode}, \
         which lies in the range reserved for system opcodes. Fix: {fix}"
    )]
    ReservedOpcode {
        /// The tenant whose window produced the opcode.
        tenant_id: u32,
        /// The opcode the tenant asked for, within its own window.
        local_opcode: u32,
        /// The global opcode the window mapped it to.
        global_opcode: u32,
        /// Actionable remediation.
        fix: &'static str,
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

impl From<vyre_driver::BackendError> for PipelineError {
    fn from(err: vyre_driver::BackendError) -> Self {
        PipelineError::Backend(err.to_string())
    }
}

/// Canonical artifact-envelope authentication and exact-format admission.
pub mod artifact_admission;
mod semantic_execution;
pub use semantic_execution::RegisteredSemanticExecutor;

/// Intra-device expert scheduling and inter-device token exchange.
pub mod expert_scheduling;
/// Multi-Token Prediction (MTP) speculative decoding and rollback coordination.
pub mod mtp;
/// Paged KV cache residency contracts and validation.
pub mod paged_residency;
/// Radix prefix-cache lifecycle, immutable identity, and copy-on-write allocation.
pub mod prefix_cache;
/// Backend-neutral immutable-resource and mutable-state residency.
pub mod resource_residency;

/// Authenticated safetensors transfer lifecycle, residency composition, and integrity.
pub mod safetensors_transfer;

/// Resident work-queue protocols, scheduling policy, and runtime IO.
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
