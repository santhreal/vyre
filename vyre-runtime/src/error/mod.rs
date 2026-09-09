//! Pipeline and runtime error types and structured diagnostics.

mod diagnostics;

use std::fmt;

/// Renders a permitted-status set as protocol status names, so a rejection
/// message states `PUBLISHED, YIELD, REQUEUE` rather than raw words.
pub(crate) struct SlotStatusList(pub(crate) &'static [u32]);

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
pub(crate) struct SlotStatusName(pub(crate) u32);

impl fmt::Display for SlotStatusName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match crate::resident_work_queue::protocol::slot::status_name(self.0) {
            Some(name) => f.write_str(name),
            None => f.write_str("unknown"),
        }
    }
}

/// Declares a closed enum together with the list of all its variants.
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
    /// Carries the typed [`crate::resident_work_queue::protocol::ProtocolError`], so a
    /// caller reads the buffer, byte length, and word index it names. Two of
    /// the three protocol faults previously reached callers as a
    /// [`PipelineError::Backend`] string, which discarded every one of those
    /// fields.
    #[error(transparent)]
    Protocol(#[from] crate::resident_work_queue::protocol::ProtocolError),
    /// Attempted to use io_uring on a non-platform platform.
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

vyre_foundation::diagnostic_conversions!(PipelineError, diagnostic);
