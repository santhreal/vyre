//! Substrate-neutral memory model contracts and closed effect representations.
//!
//! Concurrency and memory semantics are modeled using separate, closed, orthogonal types
//! for atomic ordering, memory scope, execution scope, storage domain, fence semantics,
//! barrier participation, asynchronous transaction lifecycle, collective communication groups,
//! and failure/cancellation behavior.
//!
//! There are no default orderings or scopes.

/// Closed atomic memory ordering for atomic read-modify-write and load/store operations.
///
/// Synchronization intent must be explicitly stated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AtomicOrdering {
    /// No synchronization beyond single-location atomicity.
    Relaxed,
    /// Acquires prior writes released by another thread.
    Acquire,
    /// Releases prior writes to acquiring threads.
    Release,
    /// Both acquire and release semantics on the same memory location.
    AcqRel,
    /// Sequentially consistent total order across all threads in the relevant memory scope.
    SeqCst,
}

impl AtomicOrdering {
    /// Stable wire tag for atomic ordering.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Relaxed => 0,
            Self::Acquire => 1,
            Self::Release => 2,
            Self::AcqRel => 3,
            Self::SeqCst => 4,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `AtomicOrdering`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Relaxed),
            1 => Ok(Self::Acquire),
            2 => Ok(Self::Release),
            3 => Ok(Self::AcqRel),
            4 => Ok(Self::SeqCst),
            other => Err(format!(
                "InvalidDiscriminant: atomic ordering tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this atomic ordering.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Relaxed => "relaxed",
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::SeqCst => "seq_cst",
        }
    }

    /// Whether this ordering incorporates acquire synchronization.
    #[must_use]
    #[inline]
    pub const fn is_acquire(self) -> bool {
        match self {
            Self::Acquire | Self::AcqRel | Self::SeqCst => true,
            Self::Relaxed | Self::Release => false,
        }
    }

    /// Whether this ordering incorporates release synchronization.
    #[must_use]
    #[inline]
    pub const fn is_release(self) -> bool {
        match self {
            Self::Release | Self::AcqRel | Self::SeqCst => true,
            Self::Relaxed | Self::Acquire => false,
        }
    }

    /// Join two atomic orderings into the weakest ordering satisfying both.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        use AtomicOrdering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
        match (self, other) {
            (SeqCst, _) | (_, SeqCst) => SeqCst,
            (AcqRel, _) | (_, AcqRel) | (Acquire, Release) | (Release, Acquire) => AcqRel,
            (Acquire, Acquire) => Acquire,
            (Release, Release) => Release,
            (Relaxed, ordering) | (ordering, Relaxed) => ordering,
        }
    }
}

/// Closed memory visibility and coherence scope across execution hierarchy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryScope {
    /// Single thread / invocation visibility only.
    Thread,
    /// Subgroup / warp / wavefront coherent visibility.
    Subgroup,
    /// Workgroup / threadblock coherent visibility.
    Workgroup,
    /// Cluster of cooperating workgroups.
    Cluster,
    /// Full device (GPU / accelerator) global memory coherence.
    Device,
    /// Cross-device / host-heterogeneous system coherence.
    System,
}

impl MemoryScope {
    /// Stable wire tag for memory scope.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Thread => 0,
            Self::Subgroup => 1,
            Self::Workgroup => 2,
            Self::Cluster => 3,
            Self::Device => 4,
            Self::System => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `MemoryScope`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Thread),
            1 => Ok(Self::Subgroup),
            2 => Ok(Self::Workgroup),
            3 => Ok(Self::Cluster),
            4 => Ok(Self::Device),
            5 => Ok(Self::System),
            other => Err(format!(
                "InvalidDiscriminant: memory scope tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this memory scope.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Thread => "thread",
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::Device => "device",
            Self::System => "system",
        }
    }

    /// Whether this scope crosses workgroup boundaries.
    #[must_use]
    #[inline]
    pub const fn is_cross_workgroup(self) -> bool {
        match self {
            Self::Cluster | Self::Device | Self::System => true,
            Self::Thread | Self::Subgroup | Self::Workgroup => false,
        }
    }

    /// Whether this scope encompasses the entire device.
    #[must_use]
    #[inline]
    pub const fn is_device_wide(self) -> bool {
        match self {
            Self::Device | Self::System => true,
            Self::Thread | Self::Subgroup | Self::Workgroup | Self::Cluster => false,
        }
    }

    /// Widen two memory scopes to the minimum scope enclosing both.
    #[must_use]
    pub const fn widen(self, other: Self) -> Self {
        if self.wire_tag() >= other.wire_tag() {
            self
        } else {
            other
        }
    }
}

/// Closed execution synchronization scope defining participant domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ExecutionScope {
    /// Single thread execution.
    Thread,
    /// Subgroup / warp execution rendezvous.
    Subgroup,
    /// Workgroup / threadblock execution rendezvous.
    Workgroup,
    /// Cluster execution rendezvous across cooperative workgroups.
    Cluster,
    /// Grid / dispatch execution rendezvous across the entire launch.
    Grid,
    /// Distributed device mesh rendezvous across multiple physical devices.
    DeviceMesh,
}

impl ExecutionScope {
    /// Stable wire tag for execution scope.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Thread => 0,
            Self::Subgroup => 1,
            Self::Workgroup => 2,
            Self::Cluster => 3,
            Self::Grid => 4,
            Self::DeviceMesh => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `ExecutionScope`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Thread),
            1 => Ok(Self::Subgroup),
            2 => Ok(Self::Workgroup),
            3 => Ok(Self::Cluster),
            4 => Ok(Self::Grid),
            5 => Ok(Self::DeviceMesh),
            other => Err(format!(
                "InvalidDiscriminant: execution scope tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this execution scope.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Thread => "thread",
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::Grid => "grid",
            Self::DeviceMesh => "device_mesh",
        }
    }

    /// Whether this execution scope crosses thread block boundaries.
    #[must_use]
    #[inline]
    pub const fn is_cross_block(self) -> bool {
        match self {
            Self::Cluster | Self::Grid | Self::DeviceMesh => true,
            Self::Thread | Self::Subgroup | Self::Workgroup => false,
        }
    }
}

/// Closed physical and logical storage tier / memory space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum StorageDomain {
    /// Register file / private thread-local storage.
    Register,
    /// Workgroup shared memory / scratchpad (L1/shared SRAM).
    Scratchpad,
    /// Workgroup-local memory tier.
    WorkgroupLocal,
    /// Device global High Bandwidth Memory (HBM/VRAM).
    DeviceGlobal,
    /// Host pinned / zero-copy system memory.
    HostPinned,
    /// Host paged / virtual system memory.
    HostPaged,
    /// Constant / read-only uniform cache storage.
    Constant,
    /// Texture / surface / specialized hardware cache domain.
    Texture,
}

impl StorageDomain {
    /// Stable wire tag for storage domain.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Register => 0,
            Self::Scratchpad => 1,
            Self::WorkgroupLocal => 2,
            Self::DeviceGlobal => 3,
            Self::HostPinned => 4,
            Self::HostPaged => 5,
            Self::Constant => 6,
            Self::Texture => 7,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `StorageDomain`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Register),
            1 => Ok(Self::Scratchpad),
            2 => Ok(Self::WorkgroupLocal),
            3 => Ok(Self::DeviceGlobal),
            4 => Ok(Self::HostPinned),
            5 => Ok(Self::HostPaged),
            6 => Ok(Self::Constant),
            7 => Ok(Self::Texture),
            other => Err(format!(
                "InvalidDiscriminant: storage domain tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for this storage domain.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Register => "register",
            Self::Scratchpad => "scratchpad",
            Self::WorkgroupLocal => "workgroup_local",
            Self::DeviceGlobal => "device_global",
            Self::HostPinned => "host_pinned",
            Self::HostPaged => "host_paged",
            Self::Constant => "constant",
            Self::Texture => "texture",
        }
    }

    /// Whether this storage domain is shared across multiple threads in a workgroup.
    #[must_use]
    #[inline]
    pub const fn is_shared_across_threads(self) -> bool {
        match self {
            Self::Scratchpad
            | Self::WorkgroupLocal
            | Self::DeviceGlobal
            | Self::HostPinned
            | Self::HostPaged
            | Self::Constant
            | Self::Texture => true,
            Self::Register => false,
        }
    }

    /// Whether this storage domain is directly accessible by host CPU.
    #[must_use]
    #[inline]
    pub const fn is_host_accessible(self) -> bool {
        match self {
            Self::HostPinned | Self::HostPaged => true,
            Self::Register
            | Self::Scratchpad
            | Self::WorkgroupLocal
            | Self::DeviceGlobal
            | Self::Constant
            | Self::Texture => false,
        }
    }
}

/// Closed memory fence synchronization semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum FenceSemantics {
    /// Acquire fence: orders subsequent reads after prior acquire operations.
    Acquire,
    /// Release fence: orders prior writes before subsequent release operations.
    Release,
    /// Acquire-Release fence: bi-directional memory ordering barrier.
    AcqRel,
    /// Sequentially consistent fence: total order across all participant operations.
    SequentiallyConsistent,
}

impl FenceSemantics {
    /// Stable wire tag for fence semantics.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Acquire => 0,
            Self::Release => 1,
            Self::AcqRel => 2,
            Self::SequentiallyConsistent => 3,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `FenceSemantics`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Acquire),
            1 => Ok(Self::Release),
            2 => Ok(Self::AcqRel),
            3 => Ok(Self::SequentiallyConsistent),
            other => Err(format!(
                "InvalidDiscriminant: fence semantics tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for fence semantics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::AcqRel => "acq_rel",
            Self::SequentiallyConsistent => "sequentially_consistent",
        }
    }
}

/// Closed barrier participant discipline and divergence contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum BarrierParticipation {
    /// All participants in the execution scope participate unconditionally.
    Uniform,
    /// Converged control flow: participants currently active at the rendezvous point.
    Converged,
    /// Exactly one elected participant executes; others observe completion.
    ElectOne,
    /// Dynamically masked participants explicitly tracked via active bitmask.
    DynamicMask,
    /// Participation restricted strictly to subgroup scope.
    SubgroupOnly,
    /// Participation restricted strictly to workgroup scope.
    WorkgroupOnly,
}

impl BarrierParticipation {
    /// Stable wire tag for barrier participation.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Uniform => 0,
            Self::Converged => 1,
            Self::ElectOne => 2,
            Self::DynamicMask => 3,
            Self::SubgroupOnly => 4,
            Self::WorkgroupOnly => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `BarrierParticipation`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Uniform),
            1 => Ok(Self::Converged),
            2 => Ok(Self::ElectOne),
            3 => Ok(Self::DynamicMask),
            4 => Ok(Self::SubgroupOnly),
            5 => Ok(Self::WorkgroupOnly),
            other => Err(format!(
                "InvalidDiscriminant: barrier participation tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for barrier participation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Converged => "converged",
            Self::ElectOne => "elect_one",
            Self::DynamicMask => "dynamic_mask",
            Self::SubgroupOnly => "subgroup_only",
            Self::WorkgroupOnly => "workgroup_only",
        }
    }

    /// Whether this participation discipline requires uniform control flow.
    #[must_use]
    #[inline]
    pub const fn requires_uniform_control_flow(self) -> bool {
        match self {
            Self::Uniform => true,
            Self::Converged
            | Self::ElectOne
            | Self::DynamicMask
            | Self::SubgroupOnly
            | Self::WorkgroupOnly => false,
        }
    }
}

/// Closed state machine lifecycle for asynchronous memory transactions / DMA / pipeline stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AsyncTransactionLifecycle {
    /// Transaction descriptor submitted to hardware / queue.
    Submitted,
    /// Transaction in flight on memory / copy engine.
    InFlight,
    /// Transaction arrived at completion barrier / stage counter incremented.
    Arrived,
    /// Transaction committed and data visible in destination storage domain.
    Committed,
    /// Transaction failed due to bus error / out of bounds / hardware fault.
    Failed,
    /// Transaction cancelled or aborted before commit.
    Aborted,
}

impl AsyncTransactionLifecycle {
    /// Stable wire tag for async transaction lifecycle.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Submitted => 0,
            Self::InFlight => 1,
            Self::Arrived => 2,
            Self::Committed => 3,
            Self::Failed => 4,
            Self::Aborted => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to an `AsyncTransactionLifecycle`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Submitted),
            1 => Ok(Self::InFlight),
            2 => Ok(Self::Arrived),
            3 => Ok(Self::Committed),
            4 => Ok(Self::Failed),
            5 => Ok(Self::Aborted),
            other => Err(format!(
                "InvalidDiscriminant: async transaction lifecycle tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for async transaction lifecycle.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::InFlight => "in_flight",
            Self::Arrived => "arrived",
            Self::Committed => "committed",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }

    /// Whether this lifecycle state represents a terminal state.
    #[must_use]
    #[inline]
    pub const fn is_terminal(self) -> bool {
        match self {
            Self::Committed | Self::Failed | Self::Aborted => true,
            Self::Submitted | Self::InFlight | Self::Arrived => false,
        }
    }

    /// Whether this transaction has successfully completed and committed.
    #[must_use]
    #[inline]
    pub const fn is_successful_commit(self) -> bool {
        match self {
            Self::Committed => true,
            Self::Submitted | Self::InFlight | Self::Arrived | Self::Failed | Self::Aborted => {
                false
            }
        }
    }
}

/// Closed collective communication group and topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum CollectiveGroup {
    /// Subgroup collective (warp shuffle / vote).
    Subgroup,
    /// Workgroup collective (shared memory reduction / scan).
    Workgroup,
    /// Cluster collective across cooperating workgroups.
    Cluster,
    /// Device mesh collective across grid partition.
    DeviceMesh,
    /// Cross-device ring topology collective.
    CrossDeviceRing,
    /// Custom topology collective with explicit participant mapping.
    CustomTopology,
}

impl CollectiveGroup {
    /// Stable wire tag for collective group.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Subgroup => 0,
            Self::Workgroup => 1,
            Self::Cluster => 2,
            Self::DeviceMesh => 3,
            Self::CrossDeviceRing => 4,
            Self::CustomTopology => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `CollectiveGroup`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Subgroup),
            1 => Ok(Self::Workgroup),
            2 => Ok(Self::Cluster),
            3 => Ok(Self::DeviceMesh),
            4 => Ok(Self::CrossDeviceRing),
            5 => Ok(Self::CustomTopology),
            other => Err(format!(
                "InvalidDiscriminant: collective group tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for collective group.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Cluster => "cluster",
            Self::DeviceMesh => "device_mesh",
            Self::CrossDeviceRing => "cross_device_ring",
            Self::CustomTopology => "custom_topology",
        }
    }

    /// Whether this collective group spans across physical devices.
    #[must_use]
    #[inline]
    pub const fn is_inter_device(self) -> bool {
        match self {
            Self::CrossDeviceRing => true,
            Self::Subgroup
            | Self::Workgroup
            | Self::Cluster
            | Self::DeviceMesh
            | Self::CustomTopology => false,
        }
    }
}

/// Closed fault, trap, and cancellation policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum FailureCancellationBehavior {
    /// Hard execution trap halting the kernel / thread.
    Trap,
    /// Mark destination buffer / register as poisoned with propagating error token.
    Poison,
    /// Propagate error status code to caller / runtime return slot.
    Propagate,
    /// Abort the entire dispatch / launch immediately.
    AbortKernel,
    /// Ignore fault / silently drop invalid transaction.
    Ignore,
}

impl FailureCancellationBehavior {
    /// Stable wire tag for failure cancellation behavior.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Trap => 0,
            Self::Poison => 1,
            Self::Propagate => 2,
            Self::AbortKernel => 3,
            Self::Ignore => 4,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an error message when `tag` does not correspond to a `FailureCancellationBehavior`.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Trap),
            1 => Ok(Self::Poison),
            2 => Ok(Self::Propagate),
            3 => Ok(Self::AbortKernel),
            4 => Ok(Self::Ignore),
            other => Err(format!(
                "InvalidDiscriminant: failure cancellation behavior tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Canonical string identifier for failure cancellation behavior.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Trap => "trap",
            Self::Poison => "poison",
            Self::Propagate => "propagate",
            Self::AbortKernel => "abort_kernel",
            Self::Ignore => "ignore",
        }
    }

    /// Whether this failure policy unconditionally halts kernel execution.
    #[must_use]
    #[inline]
    pub const fn halts_execution(self) -> bool {
        match self {
            Self::Trap | Self::AbortKernel => true,
            Self::Poison | Self::Propagate | Self::Ignore => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Exhaustive compile-time checks without catch-all arms.
// Adding a variant to any closed type without updating these functions
// causes a build failure.
// ---------------------------------------------------------------------------

/// Exhaustive compile-time check for [`AtomicOrdering`].
#[must_use]
pub const fn exhaustiveness_check_atomic_ordering(val: AtomicOrdering) -> &'static str {
    match val {
        AtomicOrdering::Relaxed => "Relaxed",
        AtomicOrdering::Acquire => "Acquire",
        AtomicOrdering::Release => "Release",
        AtomicOrdering::AcqRel => "AcqRel",
        AtomicOrdering::SeqCst => "SeqCst",
    }
}

/// Exhaustive compile-time check for [`MemoryScope`].
#[must_use]
pub const fn exhaustiveness_check_memory_scope(val: MemoryScope) -> &'static str {
    match val {
        MemoryScope::Thread => "Thread",
        MemoryScope::Subgroup => "Subgroup",
        MemoryScope::Workgroup => "Workgroup",
        MemoryScope::Cluster => "Cluster",
        MemoryScope::Device => "Device",
        MemoryScope::System => "System",
    }
}

/// Exhaustive compile-time check for [`ExecutionScope`].
#[must_use]
pub const fn exhaustiveness_check_execution_scope(val: ExecutionScope) -> &'static str {
    match val {
        ExecutionScope::Thread => "Thread",
        ExecutionScope::Subgroup => "Subgroup",
        ExecutionScope::Workgroup => "Workgroup",
        ExecutionScope::Cluster => "Cluster",
        ExecutionScope::Grid => "Grid",
        ExecutionScope::DeviceMesh => "DeviceMesh",
    }
}

/// Exhaustive compile-time check for [`StorageDomain`].
#[must_use]
pub const fn exhaustiveness_check_storage_domain(val: StorageDomain) -> &'static str {
    match val {
        StorageDomain::Register => "Register",
        StorageDomain::Scratchpad => "Scratchpad",
        StorageDomain::WorkgroupLocal => "WorkgroupLocal",
        StorageDomain::DeviceGlobal => "DeviceGlobal",
        StorageDomain::HostPinned => "HostPinned",
        StorageDomain::HostPaged => "HostPaged",
        StorageDomain::Constant => "Constant",
        StorageDomain::Texture => "Texture",
    }
}

/// Exhaustive compile-time check for [`FenceSemantics`].
#[must_use]
pub const fn exhaustiveness_check_fence_semantics(val: FenceSemantics) -> &'static str {
    match val {
        FenceSemantics::Acquire => "Acquire",
        FenceSemantics::Release => "Release",
        FenceSemantics::AcqRel => "AcqRel",
        FenceSemantics::SequentiallyConsistent => "SequentiallyConsistent",
    }
}

/// Exhaustive compile-time check for [`BarrierParticipation`].
#[must_use]
pub const fn exhaustiveness_check_barrier_participation(val: BarrierParticipation) -> &'static str {
    match val {
        BarrierParticipation::Uniform => "Uniform",
        BarrierParticipation::Converged => "Converged",
        BarrierParticipation::ElectOne => "ElectOne",
        BarrierParticipation::DynamicMask => "DynamicMask",
        BarrierParticipation::SubgroupOnly => "SubgroupOnly",
        BarrierParticipation::WorkgroupOnly => "WorkgroupOnly",
    }
}

/// Exhaustive compile-time check for [`AsyncTransactionLifecycle`].
#[must_use]
pub const fn exhaustiveness_check_async_transaction_lifecycle(
    val: AsyncTransactionLifecycle,
) -> &'static str {
    match val {
        AsyncTransactionLifecycle::Submitted => "Submitted",
        AsyncTransactionLifecycle::InFlight => "InFlight",
        AsyncTransactionLifecycle::Arrived => "Arrived",
        AsyncTransactionLifecycle::Committed => "Committed",
        AsyncTransactionLifecycle::Failed => "Failed",
        AsyncTransactionLifecycle::Aborted => "Aborted",
    }
}

/// Exhaustive compile-time check for [`CollectiveGroup`].
#[must_use]
pub const fn exhaustiveness_check_collective_group(val: CollectiveGroup) -> &'static str {
    match val {
        CollectiveGroup::Subgroup => "Subgroup",
        CollectiveGroup::Workgroup => "Workgroup",
        CollectiveGroup::Cluster => "Cluster",
        CollectiveGroup::DeviceMesh => "DeviceMesh",
        CollectiveGroup::CrossDeviceRing => "CrossDeviceRing",
        CollectiveGroup::CustomTopology => "CustomTopology",
    }
}

/// Exhaustive compile-time check for [`FailureCancellationBehavior`].
#[must_use]
pub const fn exhaustiveness_check_failure_cancellation_behavior(
    val: FailureCancellationBehavior,
) -> &'static str {
    match val {
        FailureCancellationBehavior::Trap => "Trap",
        FailureCancellationBehavior::Poison => "Poison",
        FailureCancellationBehavior::Propagate => "Propagate",
        FailureCancellationBehavior::AbortKernel => "AbortKernel",
        FailureCancellationBehavior::Ignore => "Ignore",
    }
}

// ---------------------------------------------------------------------------
// Legacy combined MemoryOrdering (no Default implementation)
// ---------------------------------------------------------------------------

/// Memory ordering attached to atomic and barrier operations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryOrdering {
    /// No synchronization beyond atomicity of the operation.
    Relaxed,
    /// Subsequent reads observe writes released by another participant.
    Acquire,
    /// Prior writes become visible to acquiring participants.
    Release,
    /// Acquire and release semantics in one operation.
    AcqRel,
    /// Single total order across sequentially consistent operations
    /// within the issuing thread's workgroup.
    SeqCst,
    /// Cross-grid synchronization. Every thread in the dispatch waits
    /// here, and every prior write is globally visible after the
    /// barrier returns.
    GridSync,
}

impl MemoryOrdering {
    /// Stable wire tag for this ordering.
    #[must_use]
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Relaxed => 0,
            Self::Acquire => 1,
            Self::Release => 2,
            Self::AcqRel => 3,
            Self::SeqCst => 4,
            Self::GridSync => 5,
        }
    }

    /// Decode a stable wire tag.
    ///
    /// # Errors
    ///
    /// Returns an actionable error when `tag` is not assigned to a memory
    /// ordering in this schema.
    #[inline]
    pub fn from_wire_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Relaxed),
            1 => Ok(Self::Acquire),
            2 => Ok(Self::Release),
            3 => Ok(Self::AcqRel),
            4 => Ok(Self::SeqCst),
            5 => Ok(Self::GridSync),
            other => Err(format!(
                "InvalidDiscriminant: memory ordering tag {other} is unknown. Fix: reserialize with a compatible VYRE wire schema."
            )),
        }
    }

    /// Whether this ordering is valid for an atomic RMW operation.
    /// `GridSync` is barrier-only and not a valid atomic ordering.
    #[must_use]
    #[inline]
    pub const fn is_valid_for_atomic_rmw(self) -> bool {
        matches!(
            self,
            Self::Relaxed | Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst
        )
    }

    /// Whether this ordering is valid for a barrier.
    #[must_use]
    #[inline]
    pub const fn is_valid_for_barrier(self) -> bool {
        matches!(
            self,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst | Self::GridSync
        )
    }

    /// Whether this ordering requires cross-grid synchronization.
    #[must_use]
    #[inline]
    pub const fn requires_grid_sync(self) -> bool {
        matches!(self, Self::GridSync)
    }

    /// Convert to canonical closed [`AtomicOrdering`] if valid for atomic operations.
    #[must_use]
    pub const fn to_atomic_ordering(self) -> Option<AtomicOrdering> {
        match self {
            Self::Relaxed => Some(AtomicOrdering::Relaxed),
            Self::Acquire => Some(AtomicOrdering::Acquire),
            Self::Release => Some(AtomicOrdering::Release),
            Self::AcqRel => Some(AtomicOrdering::AcqRel),
            Self::SeqCst => Some(AtomicOrdering::SeqCst),
            Self::GridSync => None,
        }
    }

    /// Primary execution synchronization scope implied by this ordering.
    #[must_use]
    pub const fn execution_scope(self) -> ExecutionScope {
        match self {
            Self::Relaxed => ExecutionScope::Thread,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst => ExecutionScope::Workgroup,
            Self::GridSync => ExecutionScope::Grid,
        }
    }

    /// Primary memory coherence scope implied by this ordering.
    #[must_use]
    pub const fn memory_scope(self) -> MemoryScope {
        match self {
            Self::Relaxed => MemoryScope::Thread,
            Self::Acquire | Self::Release | Self::AcqRel | Self::SeqCst => MemoryScope::Workgroup,
            Self::GridSync => MemoryScope::Device,
        }
    }

    /// Join two orderings to the weakest ordering that satisfies both.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        use MemoryOrdering::{AcqRel, Acquire, GridSync, Relaxed, Release, SeqCst};
        match (self, other) {
            (GridSync, _) | (_, GridSync) => GridSync,
            (SeqCst, _) | (_, SeqCst) => SeqCst,
            (AcqRel, _) | (_, AcqRel) | (Acquire, Release) | (Release, Acquire) => AcqRel,
            (Acquire, Acquire) => Acquire,
            (Release, Release) => Release,
            (Relaxed, ordering) | (ordering, Relaxed) => ordering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_memory_ordering_variants_are_distinct() {
        let variants = [
            MemoryOrdering::Relaxed,
            MemoryOrdering::Acquire,
            MemoryOrdering::Release,
            MemoryOrdering::AcqRel,
            MemoryOrdering::SeqCst,
            MemoryOrdering::GridSync,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn grid_sync_round_trips() {
        let tag = MemoryOrdering::GridSync.wire_tag();
        assert_eq!(tag, 5);
        assert_eq!(
            MemoryOrdering::from_wire_tag(tag).unwrap(),
            MemoryOrdering::GridSync
        );
        assert!(MemoryOrdering::GridSync.is_valid_for_barrier());
        assert!(!MemoryOrdering::GridSync.is_valid_for_atomic_rmw());
        assert!(MemoryOrdering::GridSync.requires_grid_sync());
        assert!(!MemoryOrdering::SeqCst.requires_grid_sync());
    }

    #[test]
    fn clone_eq() {
        let a = MemoryOrdering::AcqRel;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn exhaustive_checks_exercise_every_variant() {
        let orderings = [
            AtomicOrdering::Relaxed,
            AtomicOrdering::Acquire,
            AtomicOrdering::Release,
            AtomicOrdering::AcqRel,
            AtomicOrdering::SeqCst,
        ];
        for o in orderings {
            assert!(!exhaustiveness_check_atomic_ordering(o).is_empty());
            assert_eq!(AtomicOrdering::from_wire_tag(o.wire_tag()).unwrap(), o);
        }

        let mem_scopes = [
            MemoryScope::Thread,
            MemoryScope::Subgroup,
            MemoryScope::Workgroup,
            MemoryScope::Cluster,
            MemoryScope::Device,
            MemoryScope::System,
        ];
        for s in mem_scopes {
            assert!(!exhaustiveness_check_memory_scope(s).is_empty());
            assert_eq!(MemoryScope::from_wire_tag(s.wire_tag()).unwrap(), s);
        }

        let exec_scopes = [
            ExecutionScope::Thread,
            ExecutionScope::Subgroup,
            ExecutionScope::Workgroup,
            ExecutionScope::Cluster,
            ExecutionScope::Grid,
            ExecutionScope::DeviceMesh,
        ];
        for e in exec_scopes {
            assert!(!exhaustiveness_check_execution_scope(e).is_empty());
            assert_eq!(ExecutionScope::from_wire_tag(e.wire_tag()).unwrap(), e);
        }

        let storage_domains = [
            StorageDomain::Register,
            StorageDomain::Scratchpad,
            StorageDomain::WorkgroupLocal,
            StorageDomain::DeviceGlobal,
            StorageDomain::HostPinned,
            StorageDomain::HostPaged,
            StorageDomain::Constant,
            StorageDomain::Texture,
        ];
        for d in storage_domains {
            assert!(!exhaustiveness_check_storage_domain(d).is_empty());
            assert_eq!(StorageDomain::from_wire_tag(d.wire_tag()).unwrap(), d);
        }

        let fences = [
            FenceSemantics::Acquire,
            FenceSemantics::Release,
            FenceSemantics::AcqRel,
            FenceSemantics::SequentiallyConsistent,
        ];
        for f in fences {
            assert!(!exhaustiveness_check_fence_semantics(f).is_empty());
            assert_eq!(FenceSemantics::from_wire_tag(f.wire_tag()).unwrap(), f);
        }

        let parts = [
            BarrierParticipation::Uniform,
            BarrierParticipation::Converged,
            BarrierParticipation::ElectOne,
            BarrierParticipation::DynamicMask,
            BarrierParticipation::SubgroupOnly,
            BarrierParticipation::WorkgroupOnly,
        ];
        for p in parts {
            assert!(!exhaustiveness_check_barrier_participation(p).is_empty());
            assert_eq!(BarrierParticipation::from_wire_tag(p.wire_tag()).unwrap(), p);
        }

        let lifecycles = [
            AsyncTransactionLifecycle::Submitted,
            AsyncTransactionLifecycle::InFlight,
            AsyncTransactionLifecycle::Arrived,
            AsyncTransactionLifecycle::Committed,
            AsyncTransactionLifecycle::Failed,
            AsyncTransactionLifecycle::Aborted,
        ];
        for l in lifecycles {
            assert!(!exhaustiveness_check_async_transaction_lifecycle(l).is_empty());
            assert_eq!(
                AsyncTransactionLifecycle::from_wire_tag(l.wire_tag()).unwrap(),
                l
            );
        }

        let collectives = [
            CollectiveGroup::Subgroup,
            CollectiveGroup::Workgroup,
            CollectiveGroup::Cluster,
            CollectiveGroup::DeviceMesh,
            CollectiveGroup::CrossDeviceRing,
            CollectiveGroup::CustomTopology,
        ];
        for c in collectives {
            assert!(!exhaustiveness_check_collective_group(c).is_empty());
            assert_eq!(CollectiveGroup::from_wire_tag(c.wire_tag()).unwrap(), c);
        }

        let cancellations = [
            FailureCancellationBehavior::Trap,
            FailureCancellationBehavior::Poison,
            FailureCancellationBehavior::Propagate,
            FailureCancellationBehavior::AbortKernel,
            FailureCancellationBehavior::Ignore,
        ];
        for fail in cancellations {
            assert!(!exhaustiveness_check_failure_cancellation_behavior(fail).is_empty());
            assert_eq!(
                FailureCancellationBehavior::from_wire_tag(fail.wire_tag()).unwrap(),
                fail
            );
        }
    }
}
